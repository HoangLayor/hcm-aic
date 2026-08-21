import os
import json
import logging
import re
import torch
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
Describe this short news video using exactly one concise English sentence.

Only describe visually supported people, actions, objects, locations,
visible text, and important visual events.
Do not infer information that is not clearly visible.
Do not provide analysis, reasoning, bullet points, headings, or explanations.

Return only:
<caption>Your caption here.</caption>
"""

class SegmentCaptioner:
    def __init__(self):
        logger.info(f"Loading Qwen VLM ({config.VLM_MODEL_ID})... This may take a while and consume ~8GB VRAM.")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForMultimodalLM.from_pretrained(
                config.VLM_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(config.VLM_MODEL_ID, trust_remote_code=True)
            
            # Constrain video length and resolution strictly to prevent OOM
            if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
                self.processor.video_processor.max_frames = config.VLM_MAX_VIDEO_FRAMES
                self.processor.video_processor.size = {
                    "shortest_edge": 4_096,
                    "longest_edge": 8_388_608, # Max pixel budget
                }
            logger.info("Qwen VLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Qwen VLM: {e}")
            raise

    def free_memory(self):
        """Explicitly clear Qwen VLM from VRAM."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen VLM memory freed.")

    @staticmethod
    def extract_caption(text: str) -> str:
        """Extract one final caption and reject reasoning-style model output."""
        text = (text or "").strip()
        if not text:
            return ""

        # Some Qwen templates can still emit a thinking block even when thinking is disabled.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()

        match = re.search(
            r"<caption>\s*(.*?)\s*</caption>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            text = match.group(1)
        else:
            text = re.sub(
                r"^(caption|description)\s*:\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            reasoning_markers = (
                "identify the main",
                "identify their",
                "synthesize the description",
                "the user wants",
            )
            looks_like_reasoning = (
                any(marker in text.lower() for marker in reasoning_markers)
                or bool(re.search(r"(^|\n)\s*\d+[.)]\s+", text))
                or bool(re.search(r"(^|\n)\s*[-*]\s+", text))
            )
            if looks_like_reasoning:
                return ""

        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text.split()) <= 80 else ""

    def generate_caption(self, video_path: Path) -> str:
        """Takes a video path and generates a caption."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": str(video_path)},
                    {"type": "text", "text": PROMPT_TEMPLATE.strip()},
                ],
            }
        ]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                fps=config.VLM_VIDEO_FPS,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )

            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                    do_sample=False, # Use greedy decoding for factual descriptions
                    temperature=0.0,
                )

            # Trim the prompt tokens from the output
            generated_ids_trimmed = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            caption = self.extract_caption(output_text)
            if not caption:
                raise ValueError(
                    f"Model did not return a valid caption. Raw output: {output_text[:500]}"
                )
            return caption
        except Exception as e:
            logger.error(f"Error generating caption for {video_path.name}: {e}")
            raise

    def process_video(self, video_id: str, force: bool = False):
        """Generates captions for all segments of a video with checkpoint support."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for video {video_id}: {manifest_path}")
            return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Generating captions for {len(segments)} segments in {video_id}...")
            
            captions_dir = video_dir / "captions"
            captions_dir.mkdir(parents=True, exist_ok=True)
            
            all_captions = []
            
            for seg_meta in segments:
                segment_id = seg_meta['segment_id']
                segment_file = Path(seg_meta['file_path'])
                cap_file = captions_dir / f"{segment_id}_caption.json"

                # Checkpoint check: skip generation if caption file already exists
                if not force and cap_file.exists():
                    try:
                        with open(cap_file, 'r', encoding='utf-8') as f:
                            cap_data = json.load(f)
                        if cap_data.get("caption"):
                            logger.info(f"  -> [Skip] Caption for {segment_id} (already generated: {cap_data['caption'][:40]}...).")
                            all_captions.append(cap_data)
                            continue
                    except Exception:
                        pass
                
                logger.info(f"Processing segment {segment_id} with Qwen VLM...")
                caption_text = self.generate_caption(segment_file)
                
                # Save individual caption
                cap_data = {
                    "segment_id": segment_id,
                    "global_start_frame": seg_meta.get('global_start_frame', 0),
                    "caption": caption_text
                }
                
                with open(cap_file, 'w', encoding='utf-8') as f:
                    json.dump(cap_data, f, indent=2, ensure_ascii=False)
                    
                all_captions.append(cap_data)
                logger.info(f"  -> Caption generated: {caption_text[:50]}...")
            
            # Save aggregated captions for easy loading by the embedder
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
