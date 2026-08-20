import os
import json
import logging
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from src.config import config

logger = logging.getLogger("KeyframeExtractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class KeyframeExtractor:
    def __init__(self):
        logger.info(f"Loading DINOv2 ({config.DINO_MODEL_ID})...")
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_entrypoint = config.DINO_MODEL_ID.split('/')[-1]
            self.dino = torch.hub.load('facebookresearch/dinov2', model_entrypoint).eval().to(self.device)
            # Standard ImageNet norm values used by DINOv2
            self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(3, 1, 1)
            logger.info("DINOv2 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DINOv2: {e}")
            raise

    def free_memory(self):
        """Explicitly clear DINOv2 from VRAM."""
        if hasattr(self, 'dino'):
            del self.dino
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DINOv2 memory freed.")

    def read_frames(self, path: Path) -> list:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
            
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            logger.warning(f"No frames extracted from {path}")
        return frames

    def encode_dino(self, frames: list) -> np.ndarray:
        output = []
        for start in range(0, len(frames), config.DINO_BATCH_SIZE):
            tensors = []
            for frame in frames[start:start + config.DINO_BATCH_SIZE]:
                # Resize and color space conversion
                rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
                tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
                
            batch = (torch.stack(tensors).to(self.device) - self.mean) / self.std
            with torch.inference_mode():
                # Normalize features for cosine similarity
                embeddings = F.normalize(self.dino(batch), dim=1).cpu()
                output.append(embeddings)
                
        return torch.cat(output).numpy() if output else np.array([])

    def group_and_select(self, embeddings: np.ndarray):
        """Groups sequential similar frames and selects the medoid as keyframe."""
        if len(embeddings) == 0:
            return [], [], []
            
        if len(embeddings) == 1:
            return [[0]], [0], []
            
        scores = np.sum(embeddings[:-1] * embeddings[1:], axis=1).astype(float).tolist()
        groups, current = [], [0]
        
        for index, score in enumerate(scores, start=1):
            if score < config.DINO_SIMILARITY_THRESHOLD:
                groups.append(current)
                current = [index]
            else:
                current.append(index)
        groups.append(current)
        
        keyframes = []
        for group in groups:
            # Sub-sample candidates if group is too large
            positions = np.linspace(0, len(group) - 1, min(config.DINO_MAX_CANDIDATES, len(group)), dtype=int)
            candidates = [group[pos] for pos in positions]
            
            if len(candidates) == 1:
                keyframes.append(candidates[0])
                continue
                
            # Select medoid
            vectors = embeddings[candidates]
            matrix = vectors @ vectors.T
            medoid_idx = int(np.argmax((matrix.sum(1) - 1) / (len(candidates) - 1)))
            keyframes.append(candidates[medoid_idx])
            
        return groups, keyframes, scores

    def process_segment(self, seg_meta: dict, video_dir: Path, force: bool = False):
        """Extracts keyframes for a single segment and saves grouping metadata."""
        segment_id = seg_meta['segment_id']
        segment_file = Path(seg_meta['file_path'])
        global_start_frame = seg_meta.get('global_start_frame', 0)
        
        meta_file = video_dir / "keyframes" / f"{segment_id}_meta.json"

        # Checkpoint check: skip if meta file and keyframe images already exist
        if not force and meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                cached_groups = cached_data.get("groups", [])
                if cached_groups and all(Path(g["image_path"]).exists() for g in cached_groups):
                    logger.info(f"  -> [Skip] Keyframes for {segment_id} ({len(cached_groups)} keyframes already exist).")
                    return
            except Exception:
                pass
        
        logger.info(f"Processing segment {segment_id} (Global Start: {global_start_frame})...")
        
        frames = self.read_frames(segment_file)
        if not frames:
            return

        embeddings = self.encode_dino(frames)
        groups, keyframes, scores = self.group_and_select(embeddings)
        
        # Save Keyframes
        keyframe_dir = video_dir / "keyframes" / segment_id
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        
        group_metadata = []
        for i, (group, local_kf_idx) in enumerate(zip(groups, keyframes), start=1):
            # Calculate absolute global frame index safely
            global_kf_idx = global_start_frame + local_kf_idx
            
            frame_img = frames[local_kf_idx]
            kf_filename = f"keyframe_{global_kf_idx:08d}.jpg"
            kf_path = keyframe_dir / kf_filename
            cv2.imwrite(str(kf_path), frame_img)
            
            group_metadata.append({
                "group_id": i,
                "local_start_frame": int(group[0]),
                "local_end_frame": int(group[-1]),
                "local_keyframe": int(local_kf_idx),
                "global_start_frame": global_start_frame + int(group[0]),
                "global_end_frame": global_start_frame + int(group[-1]),
                "global_keyframe": global_kf_idx,
                "num_frames": len(group),
                "image_path": str(kf_path)
            })

        # Save segment's grouping metadata
        data = {
            "segment_id": segment_id,
            "global_start_frame": global_start_frame,
            "num_frames_extracted": len(frames),
            "n_groups": len(groups),
            "groups": group_metadata
        }
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"  -> {segment_id}: {len(groups)} groups, {len(keyframes)} keyframes saved.")

    def process_video(self, video_id: str, force: bool = False):
        """Processes all segments of a video defined in manifest_vad.json with resume support."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        manifest_path = video_dir / "manifest_vad.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found for video {video_id}: {manifest_path}")
            return
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            segments = manifest.get('segments', [])
            logger.info(f"Extracting keyframes for {len(segments)} segments in {video_id}...")
            
            for seg_meta in segments:
                self.process_segment(seg_meta, video_dir, force=force)
                
            logger.info(f"Successfully finished Keyframe Extraction for {video_id}.")
        except Exception as e:
            logger.error(f"Keyframe extraction failed for {video_id}: {e}")

# Context manager to ensure model cleanup
class DINOv2Context:
    def __enter__(self):
        self.extractor = KeyframeExtractor()
        return self.extractor
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.extractor.free_memory()

# Quick Testing
if __name__ == "__main__":
    # Test context manager behavior
    with DINOv2Context() as ext:
        logger.info("Extractor instantiated inside context.")
        # ext.process_video("vid_sample_001")
    logger.info("Extractor destroyed and memory freed.")
