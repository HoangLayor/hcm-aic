import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger("MultiModalEmbedder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class MultiModalEmbedder:
    def __init__(self):
        logger.info(f"Loading Qwen3-VL-Embedder ({config.EMBEDDER_MODEL_ID})...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(config.EMBEDDER_MODEL_ID, device=self.device)
            # Use fp16 to save memory and speed up inference if on CUDA
            if self.device == "cuda":
                self.model.half()
            logger.info("Embedder loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Embedder: {e}")
            raise

    def free_memory(self):
        """Explicitly clear Embedder from VRAM."""
        if hasattr(self, 'model'):
            del self.model
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Embedder memory freed.")

    def embed_texts(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([])
        vectors = self.model.encode(
            texts, 
            batch_size=config.EMBEDDER_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vectors.astype(np.float32)

    def embed_images(self, image_paths: list) -> np.ndarray:
        if not image_paths:
            return np.array([])
            
        all_vectors = []
        # Chunking to prevent CPU RAM explosion on massive videos
        chunk_size = 500
        for i in range(0, len(image_paths), chunk_size):
            chunk_paths = image_paths[i:i+chunk_size]
            resized_images = []
            
            for path in chunk_paths:
                with Image.open(path) as img:
                    img = img.convert('RGB')
                    img.thumbnail((config.IMAGE_RESIZE_PX, config.IMAGE_RESIZE_PX))
                    # We have to keep a reference to the image in memory while encoding
                    resized_images.append(img.copy())
                    
            vectors = self.model.encode(
                resized_images,
                batch_size=config.EMBEDDER_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_vectors.append(vectors.astype(np.float32))
            
            # Explicitly clear PIL images from memory
            for img in resized_images:
                img.close()
            del resized_images
            
        return np.concatenate(all_vectors, axis=0) if all_vectors else np.array([])

    def process_video(self, video_id: str, force: bool = False):
        """Embeds all keyframes and captions for a given video with checkpoint support."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        embed_dir = video_dir / "embeddings"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for video {video_id}.")
            return

        # Checkpoint check: skip if embeddings already exist on disk
        if not force and embed_dir.exists():
            kf_vec = embed_dir / "keyframe_vectors.pt"
            kf_meta = embed_dir / "keyframe_metadata.json"
            cap_vec = embed_dir / "caption_vectors.pt"
            cap_meta = embed_dir / "caption_metadata.json"
            has_kf = kf_vec.exists() and kf_meta.exists() and kf_vec.stat().st_size > 0
            has_cap = cap_vec.exists() and cap_meta.exists() and cap_vec.stat().st_size > 0
            if has_kf or has_cap:
                logger.info(f"  -> [Skip] Embeddings for {video_id} (already exist on disk).")
                return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Processing Embeddings for {video_id}...")
            
            all_keyframe_metadata = []
            all_keyframe_paths = []
            
            all_caption_metadata = []
            all_caption_texts = []
            
            for seg_meta in segments:
                segment_id = seg_meta['segment_id']
                
                # 1. Gather Keyframes
                kf_meta_file = video_dir / "keyframes" / f"{segment_id}_meta.json"
                if kf_meta_file.exists():
                    with open(kf_meta_file, 'r', encoding='utf-8') as f:
                        kf_data = json.load(f)
                    for grp in kf_data.get('groups', []):
                        all_keyframe_paths.append(grp['image_path'])
                        all_keyframe_metadata.append({
                            "point_type": "keyframe",
                            "video_id": video_id,
                            "segment_id": segment_id,
                            "frame_index": grp['global_keyframe'],
                            "start_time_sec": seg_meta['start_time_sec'],
                            "end_time_sec": seg_meta['end_time_sec'],
                            "image_path": grp['image_path']
                        })
                        
                # 2. Gather Captions
                cap_meta_file = video_dir / "captions" / f"{segment_id}_caption.json"
                if cap_meta_file.exists():
                    with open(cap_meta_file, 'r', encoding='utf-8') as f:
                        cap_data = json.load(f)
                    all_caption_texts.append(cap_data['caption'])
                    all_caption_metadata.append({
                        "point_type": "caption",
                        "video_id": video_id,
                        "segment_id": segment_id,
                        "frame_index": cap_data['global_start_frame'], # Approximation
                        "start_time_sec": seg_meta['start_time_sec'],
                        "end_time_sec": seg_meta['end_time_sec'],
                        "caption": cap_data['caption']
                    })

            # Create Embeddings output directory
            embed_dir = video_dir / "embeddings"
            embed_dir.mkdir(parents=True, exist_ok=True)

            # Embed Images
            if all_keyframe_paths:
                logger.info(f"Embedding {len(all_keyframe_paths)} keyframes...")
                img_vectors = self.embed_images(all_keyframe_paths)
                torch.save(torch.from_numpy(img_vectors), embed_dir / "keyframe_vectors.pt")
                with open(embed_dir / "keyframe_metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(all_keyframe_metadata, f, indent=2, ensure_ascii=False)
            
            # Embed Captions
            if all_caption_texts:
                logger.info(f"Embedding {len(all_caption_texts)} captions...")
                txt_vectors = self.embed_texts(all_caption_texts)
                torch.save(torch.from_numpy(txt_vectors), embed_dir / "caption_vectors.pt")
                with open(embed_dir / "caption_metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(all_caption_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved embeddings for {video_id}.")
            
        except Exception as e:
            logger.error(f"Embedding failed for {video_id}: {e}")

# Context manager to ensure model cleanup
class EmbedderContext:
    def __enter__(self):
        self.embedder = MultiModalEmbedder()
        return self.embedder
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.embedder.free_memory()

# Quick Testing
if __name__ == "__main__":
    pass
