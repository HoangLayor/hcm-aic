import os
import json
import logging
import torch
import torch.multiprocessing as mp
import queue
from pathlib import Path
from transformers import AutoModelForMultimodalLM, AutoProcessor

from src.config import config

logger = logging.getLogger("SegmentCaptioner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

PROMPT_TEMPLATE = """
Analyze this short news video clip and generate one concise English caption.

Describe only visually supported information.

Focus on:
- people
- actions
- objects
- scene or location
- visible text
- important visual events

Do not infer names, locations, or events unless they are clearly supported by the video.
"""

def _caption_worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue):
    """Worker function that runs on a specific GPU."""
    # Isolate this process to only see one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Must re-setup logger for child process
    worker_logger = logging.getLogger(f"Worker-GPU{gpu_id}")
    worker_logger.setLevel(logging.INFO)
    if not worker_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
        worker_logger.addHandler(ch)
        
    worker_logger.info(f"Initializing Qwen VLM ({config.VLM_MODEL_ID}) on GPU {gpu_id}...")
    try:
        device = "cuda"
        # Load model directly to cuda (which is now just the one GPU we exposed)
        model = AutoModelForMultimodalLM.from_pretrained(
            config.VLM_MODEL_ID,
            dtype=torch.float16,
            device_map=device,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(config.VLM_MODEL_ID)
        
        if hasattr(processor, "video_processor") and processor.video_processor is not None:
            processor.video_processor.max_frames = config.VLM_MAX_VIDEO_FRAMES
            processor.video_processor.size = {
                "shortest_edge": 4_096,
                "longest_edge": config.VLM_MAX_VIDEO_PIXELS,
            }
        worker_logger.info(f"Model initialized successfully on GPU {gpu_id}.")
        
        while True:
            try:
                # Use timeout to allow checking for interrupt, but we rely on sentinel 'None'
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if task is None:
                worker_logger.info(f"Received shutdown signal. Exiting GPU {gpu_id}.")
                break
                
            video_path = task['video_path']
            segment_id = task['segment_id']
            
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": str(video_path)},
                            {"type": "text", "text": PROMPT_TEMPLATE.strip()},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    fps=config.VLM_VIDEO_FPS,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )

                generated_ids_trimmed = [
                    output_ids[len(input_ids):]
                    for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                result_queue.put({"status": "success", "segment_id": segment_id, "caption": output_text[0]})
            except Exception as e:
                worker_logger.error(f"Error processing {segment_id} on GPU {gpu_id}: {e}")
                result_queue.put({"status": "error", "segment_id": segment_id, "error": str(e)})
                
    except Exception as e:
        worker_logger.error(f"Failed to load model on GPU {gpu_id}: {e}")
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'processor' in locals():
            del processor
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class SegmentCaptioner:
    def __init__(self):
        logger.info("Initializing Multiprocessing SegmentCaptioner...")
        # Get number of GPUs available before we fork/spawn
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            logger.warning(f"Only {self.num_gpus} GPU(s) found. Multiprocessing will spawn {max(1, self.num_gpus)} worker(s).")
            self.num_gpus = max(1, self.num_gpus)
            
        # Use 'spawn' to prevent CUDA initialization context errors in child processes
        ctx = mp.get_context('spawn')
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.workers = []
        
        logger.info(f"Spawning {self.num_gpus} VLM workers (One per GPU)... This takes time.")
        for i in range(self.num_gpus):
            p = ctx.Process(target=_caption_worker, args=(i, self.task_queue, self.result_queue))
            p.daemon = True
            p.start()
            self.workers.append(p)
            
    def free_memory(self):
        """Cleanly shuts down worker processes."""
        logger.info("Shutting down VLM workers and freeing memory...")
        # Send sentinel value for each worker
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
            
        for p in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not terminate gracefully, forcing terminate.")
                p.terminate()
                
        logger.info("All VLM workers terminated.")

    def process_video(self, video_id: str, force: bool = False):
        """Generates captions for all segments of a video using workers."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for video {video_id}: {manifest_path}")
            return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Gathering tasks for {len(segments)} segments in {video_id}...")
            
            captions_dir = video_dir / "captions"
            captions_dir.mkdir(parents=True, exist_ok=True)
            
            all_captions = []
            pending_tasks = 0
            
            for seg_meta in segments:
                segment_id = seg_meta['segment_id']
                segment_file = Path(seg_meta['file_path'])
                cap_file = captions_dir / f"{segment_id}_caption.json"

                # Checkpoint check
                if not force and cap_file.exists():
                    try:
                        with open(cap_file, 'r', encoding='utf-8') as f:
                            cap_data = json.load(f)
                        if cap_data.get("caption"):
                            logger.info(f"  -> [Skip] Caption for {segment_id} (already generated).")
                            all_captions.append(cap_data)
                            continue
                    except Exception:
                        pass
                
                # Push to queue
                self.task_queue.put({
                    'video_path': str(segment_file),
                    'segment_id': segment_id
                })
                pending_tasks += 1
                
            if pending_tasks > 0:
                logger.info(f"Dispatched {pending_tasks} tasks to GPU workers. Waiting for results...")
                
            # Collect results
            completed_tasks = 0
            while completed_tasks < pending_tasks:
                result = self.result_queue.get()
                completed_tasks += 1
                segment_id = result['segment_id']
                
                if result['status'] == 'success':
                    caption_text = result['caption']
                    logger.info(f"  -> [{completed_tasks}/{pending_tasks}] Caption generated for {segment_id}: {caption_text[:50]}...")
                    
                    # Find meta to attach global_start_frame
                    meta = next((s for s in segments if s['segment_id'] == segment_id), {})
                    cap_data = {
                        "segment_id": segment_id,
                        "global_start_frame": meta.get('global_start_frame', 0),
                        "caption": caption_text
                    }
                    
                    cap_file = captions_dir / f"{segment_id}_caption.json"
                    with open(cap_file, 'w', encoding='utf-8') as f:
                        json.dump(cap_data, f, indent=2, ensure_ascii=False)
                        
                    all_captions.append(cap_data)
                else:
                    logger.error(f"  -> [{completed_tasks}/{pending_tasks}] Failed on {segment_id}: {result.get('error')}")
            
            # Sort all_captions by global_start_frame to maintain order
            all_captions.sort(key=lambda x: x.get('global_start_frame', 0))
            
            # Save aggregated captions
            with open(video_dir / "all_captions.json", 'w', encoding='utf-8') as f:
                json.dump({"video_id": video_id, "captions": all_captions}, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully finished Captioning for {video_id}.")
        except Exception as e:
            logger.error(f"Caption generation failed for {video_id}: {e}")
            raise

# Context manager to ensure model cleanup
class QwenCaptionerContext:
    def __enter__(self):
        self.captioner = SegmentCaptioner()
        return self.captioner
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.captioner.free_memory()

# Quick Testing
if __name__ == "__main__":
    pass
