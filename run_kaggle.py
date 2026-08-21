import os
import gc
import sys
import logging
import argparse
from pathlib import Path
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KaggleOrchestrator")

def free_memory():
    """Utility to aggressively free VRAM and RAM between heavy model stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Memory freed.")

def find_raw_videos(raw_dir_path: str, limit: int = None) -> list:
    """Find all video files in raw_dir_path (supports .mp4, .mkv, .avi, .mov and subdirectories)."""
    raw_dir = Path(raw_dir_path)
    if not raw_dir.exists():
        logger.warning(f"RAW_DIR does not exist: {raw_dir.resolve()}")
        return []
    
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MKV"}
    video_files = [
        f for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix in video_extensions
    ]
    video_files = sorted(video_files)
    if limit and limit > 0:
        video_files = video_files[:limit]
    return video_files

def run_staged_pipeline(dry_run: bool = False, raw_dir: str = None, limit: int = None, stage: str = "all", force: bool = False):
    """
    Staged Execution Pipeline designed for Kaggle/Colab (VRAM < 16GB).
    
    Args:
        dry_run (bool): If True, only loads and unloads models to test environment/VRAM without processing videos.
        raw_dir (str): Custom directory containing raw video files.
        limit (int): Maximum number of videos to process (useful for quick testing).
        stage (str): 'all', '1'/'vad', '1.5'/'transcript', '2'/'dino',
            '3'/'vlm', '4'/'embed', '5'/'qdrant'
        force (bool): If True, forces re-processing even if checkpoints/outputs already exist.
    """
    target_raw_dir = raw_dir if raw_dir else config.RAW_DIR
    stage = stage.lower()

    if dry_run:
        logger.info("==================================================")
        logger.info("🔍 RUNNING IN DRY-RUN MODE (Model & Memory Test)")
        logger.info("==================================================")
        
        # 1. DRY RUN VAD
        logger.info("--- STAGE 1: VAD Splitting (Dry-Run) ---")
        from src.audio.vad_splitter import VadVideoSplitter
        splitter = VadVideoSplitter()
        free_memory()

        # 1.5. DRY RUN TRANSCRIPT
        logger.info("--- STAGE 1.5: Transcript Extraction (PhoASR & Pyannote) (Dry-Run) ---")
        from src.audio.transcript_extractor import TranscriptContext
        with TranscriptContext() as transcriber:
            pass
        free_memory()

        # 2. DRY RUN DINO
        logger.info("--- STAGE 2: Keyframe Extraction (DINOv2) (Dry-Run) ---")
        from src.vision.keyframe_extractor import DINOv2Context
        with DINOv2Context() as extractor:
            pass
        free_memory()

        # 3. DRY RUN VLM
        logger.info("--- STAGE 3: Dense Captioning (Qwen VLM) (Dry-Run) ---")
        from src.vlm.segment_captioner import QwenCaptionerContext
        with QwenCaptionerContext() as captioner:
            pass
        free_memory()

        # 4. DRY RUN EMBEDDER
        logger.info("--- STAGE 4: Multi-modal Embedding (Qwen3-VL-Embedder) (Dry-Run) ---")
        from src.embedding.qwen3_embedder import EmbedderContext
        with EmbedderContext() as embedder:
            pass
        free_memory()

        # 5. DRY RUN QDRANT
        logger.info("--- STAGE 5: Vector DB Ingestion (Qdrant) (Dry-Run) ---")
        from src.storage.qdrant_manager import QdrantManager
        db = QdrantManager()
        
        logger.info("==================================================")
        logger.info("✅ DRY-RUN COMPLETED SUCCESSFULLY! Models & VRAM verified.")
        logger.info("==================================================")
        return

    # ================= REAL EXECUTION MODE =================
    logger.info("==================================================")
    logger.info("🚀 RUNNING REAL PIPELINE EXECUTION")
    logger.info(f"Target RAW_DIR: {target_raw_dir}")
    if force:
        logger.info("Mode: Force re-processing (--force enabled)")
    else:
        logger.info("Mode: Smart Checkpointing & Resume (Skipping existing files)")
    logger.info("==================================================")

    # Discover videos
    raw_videos = find_raw_videos(target_raw_dir, limit=limit)
    if not raw_videos:
        logger.error(f"No video files found in RAW_DIR ('{target_raw_dir}').")
        logger.error("Please provide valid video files or use --raw-dir / AIC_RAW_DIR.")
        logger.error("Example: python run_kaggle.py --raw-dir /kaggle/input/dataset")
        sys.exit(1)

    logger.info(f"Found {len(raw_videos)} video(s) to process: {[v.name for v in raw_videos]}")
    video_ids = [v.stem for v in raw_videos]

    # 1. STAGE 1: VAD SPLITTING
    if stage in ["all", "1", "vad"]:
        logger.info("--- STAGE 1: VAD Splitting ---")
        from src.audio.vad_splitter import VadVideoSplitter
        splitter = VadVideoSplitter()
        for idx, v_path in enumerate(raw_videos, 1):
            logger.info(f"[{idx}/{len(raw_videos)}] VAD Splitting: {v_path.name} ...")
            splitter.process_video(str(v_path), force=force)
        free_memory()

    # 1.5. STAGE 1.5: TRANSCRIPT EXTRACTION
    transcript_enabled = stage in ["1.5", "transcript"] or (
        stage == "all" and config.USE_TRANSCRIPT_BRANCH
    )
    if transcript_enabled:
        logger.info("--- STAGE 1.5: Transcript Extraction (PhoASR & Pyannote) ---")
        from src.audio.transcript_extractor import TranscriptContext
        with TranscriptContext() as transcriber:
            for idx, vid in enumerate(video_ids, 1):
                logger.info(f"[{idx}/{len(video_ids)}] Extracting transcript for: {vid} ...")
                transcriber.process_video(vid, force=force)
        free_memory()

    # 2. STAGE 2: KEYFRAME EXTRACTION (DINOv2)
    if stage in ["all", "2", "dino", "keyframe"]:
        logger.info("--- STAGE 2: Keyframe Extraction (DINOv2) ---")
        from src.vision.keyframe_extractor import DINOv2Context
        with DINOv2Context() as extractor:
            for idx, vid in enumerate(video_ids, 1):
                logger.info(f"[{idx}/{len(video_ids)}] Extracting keyframes for: {vid} ...")
                extractor.process_video(vid, force=force)
        free_memory()

    # 3. STAGE 3: DENSE CAPTIONING (QWEN VLM)
    if stage in ["all", "3", "vlm", "caption"]:
        logger.info("--- STAGE 3: Dense Captioning (Qwen VLM) ---")
        from src.vlm.segment_captioner import QwenCaptionerContext
        with QwenCaptionerContext() as captioner:
            for idx, vid in enumerate(video_ids, 1):
                logger.info(f"[{idx}/{len(video_ids)}] Generating captions for: {vid} ...")
                captioner.process_video(vid, force=force)
        free_memory()

    # 4. STAGE 4: EMBEDDING (QWEN3-VL-EMBEDDER)
    if stage in ["all", "4", "embed", "embedding"]:
        logger.info("--- STAGE 4: Multi-modal Embedding (Qwen3-VL-Embedder) ---")
        from src.embedding.qwen3_embedder import EmbedderContext
        with EmbedderContext() as embedder:
            for idx, vid in enumerate(video_ids, 1):
                logger.info(f"[{idx}/{len(video_ids)}] Generating embeddings for: {vid} ...")
                embedder.process_video(vid, force=force)
        free_memory()

    # 5. STAGE 5: VECTOR DB INGESTION (QDRANT)
    if stage in ["all", "5", "qdrant", "db"]:
        logger.info("--- STAGE 5: Vector DB Ingestion (Qdrant) ---")
        from src.storage.qdrant_manager import QdrantManager
        db = QdrantManager()
        for idx, vid in enumerate(video_ids, 1):
            logger.info(f"[{idx}/{len(video_ids)}] Ingesting vectors to Qdrant for: {vid} ...")
            db.upsert_video_embeddings(vid)

    logger.info("==================================================")
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY FOR ALL VIDEOS!")
    logger.info("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle/Colab Orchestrator for AIC Video Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (tests model load/unload only)")
    parser.add_argument("--force", action="store_true", help="Force re-processing and overwrite existing checkpoints")
    parser.add_argument("--raw-dir", type=str, default=None, help="Custom path to raw videos directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all", "1", "1.5", "2", "3", "4", "5",
            "vad", "transcript", "dino", "vlm", "embed", "qdrant",
        ],
        help="Run a specific stage only",
    )

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DB_DIR, exist_ok=True)

    run_staged_pipeline(
        dry_run=args.dry_run,
        raw_dir=args.raw_dir,
        limit=args.limit,
        stage=args.stage,
        force=args.force
    )
