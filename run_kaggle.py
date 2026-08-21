import os
import gc
import sys
import logging
import argparse
import shutil
import subprocess
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

def chunk_list(items: list, chunk_size: int):
    """Yield successive chunks from items. If chunk_size <= 0, yields all items at once."""
    if chunk_size is None or chunk_size <= 0:
        yield items
        return
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def upload_batch_to_google_drive(video_ids: list, drive_destination: str, delete_local: bool):
    """Copy each video's artifacts with rclone and optionally delete verified copies.

    ``drive_destination`` must be an rclone remote path, e.g. ``gdrive:aic-output``.
    Local video directories are removed only after both commands complete for
    every video in the batch.
    """
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone was not found. Install and configure rclone before using "
            "--drive-destination. See README.md."
        )

    output_root = Path(config.OUTPUT_DIR).resolve()
    remote_root = drive_destination.rstrip("/")
    if not remote_root or ":" not in remote_root:
        raise ValueError(
            "--drive-destination must be an rclone remote path, for example "
            "gdrive:aic-output"
        )

    local_remote_pairs = []
    for video_id in video_ids:
        local_video_dir = output_root / video_id
        if not local_video_dir.is_dir():
            raise FileNotFoundError(f"Output directory not found for {video_id}: {local_video_dir}")

        remote_video_dir = f"{remote_root}/{video_id}"
        logger.info(f"☁️ Uploading {local_video_dir} to {remote_video_dir} ...")
        subprocess.run(
            ["rclone", "copy", str(local_video_dir), remote_video_dir, "--progress"],
            check=True,
        )
        local_remote_pairs.append((video_id, local_video_dir, remote_video_dir))

    # Do not trust a copy exit code alone before deleting any data in the batch.
    for video_id, local_video_dir, remote_video_dir in local_remote_pairs:
        subprocess.run(
            ["rclone", "check", str(local_video_dir), remote_video_dir, "--one-way"],
            check=True,
        )
        logger.info(f"✅ Upload verified for {video_id}.")

    if delete_local:
        for _, local_video_dir, _ in local_remote_pairs:
            shutil.rmtree(local_video_dir)
            logger.info(f"🗑️ Deleted verified local output: {local_video_dir}")

def run_staged_pipeline(
    dry_run: bool = False,
    raw_dir: str = None,
    limit: int = None,
    stage: str = "all",
    force: bool = False,
    video_batch_size: int = None,
    skip_transcript: bool = False,
    drive_destination: str = None,
    delete_local_after_upload: bool = False,
):
    """
    Staged Execution Pipeline designed for Kaggle/Colab (VRAM < 16GB).
    
    Args:
        dry_run (bool): If True, only loads and unloads models to test environment/VRAM without processing videos.
        raw_dir (str): Custom directory containing raw video files.
        limit (int): Maximum number of videos to process (useful for quick testing).
        stage (str): 'all', '1'/'vad', '1.5'/'transcript', '2'/'dino',
            '3'/'vlm', '4'/'embed', '5'/'qdrant'
        force (bool): If True, forces re-processing even if checkpoints/outputs already exist.
        video_batch_size (int): Number of videos to run end-to-end per batch (defaults to config.VIDEO_BATCH_SIZE).
        skip_transcript (bool): If True, explicitly skips transcript extraction stage.
        drive_destination (str): rclone remote destination for each completed batch.
        delete_local_after_upload (bool): Delete each local video output after verified upload.
    """
    target_raw_dir = raw_dir if raw_dir else config.RAW_DIR
    stage = stage.lower()
    batch_size = (
        video_batch_size
        if video_batch_size is not None
        else getattr(config, "VIDEO_BATCH_SIZE", 5)
    )
    use_transcript = getattr(config, "USE_TRANSCRIPT_BRANCH", True) and not skip_transcript

    if (drive_destination or delete_local_after_upload) and stage != "all":
        raise ValueError("Google Drive upload is supported only with --stage all (per completed batch).")
    if delete_local_after_upload and not drive_destination:
        raise ValueError("--delete-local-after-upload requires --drive-destination.")

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
        if use_transcript:
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

    # Discover videos
    raw_videos = find_raw_videos(target_raw_dir, limit=limit)
    if not raw_videos:
        logger.error(f"No video files found in RAW_DIR ('{target_raw_dir}').")
        logger.error("Please provide valid video files or use --raw-dir / AIC_RAW_DIR.")
        logger.error("Example: python run_kaggle.py --raw-dir /kaggle/input/dataset")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("🚀 RUNNING REAL PIPELINE EXECUTION")
    logger.info(f"Target RAW_DIR: {target_raw_dir}")
    logger.info(f"Total videos to process: {len(raw_videos)}")
    logger.info(f"Video Batch Size: {batch_size if batch_size > 0 else 'All at once'}")
    if force:
        logger.info("Mode: Force re-processing (--force enabled)")
    else:
        logger.info("Mode: Smart Checkpointing & Resume (Skipping existing files)")
    logger.info("==================================================")

    video_ids = [v.stem for v in raw_videos]

    # ================= 1. BATCHED END-TO-END EXECUTION (STAGE == 'ALL') =================
    if stage == "all":
        import math
        total_batches = math.ceil(len(raw_videos) / batch_size) if batch_size > 0 else 1
        
        for batch_idx, batch_videos in enumerate(chunk_list(raw_videos, batch_size), 1):
            batch_vids = [v.stem for v in batch_videos]
            logger.info("==================================================")
            logger.info(f"📦 BATCH {batch_idx}/{total_batches} ({len(batch_videos)} video(s)): {[v.name for v in batch_videos]}")
            logger.info("==================================================")

            # STAGE 1: VAD SPLITTING
            logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 1: VAD Splitting ---")
            from src.audio.vad_splitter import VadVideoSplitter
            splitter = VadVideoSplitter()
            for idx, v_path in enumerate(batch_videos, 1):
                logger.info(f"[{idx}/{len(batch_videos)}] VAD Splitting: {v_path.name} ...")
                splitter.process_video(str(v_path), force=force)
            free_memory()

            # STAGE 1.5: TRANSCRIPT EXTRACTION
            if use_transcript:
                logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 1.5: Transcript Extraction (PhoASR & Pyannote) ---")
                from src.audio.transcript_extractor import TranscriptContext
                with TranscriptContext() as transcriber:
                    for idx, vid in enumerate(batch_vids, 1):
                        logger.info(f"[{idx}/{len(batch_vids)}] Extracting transcript for: {vid} ...")
                        transcriber.process_video(vid, force=force)
                free_memory()

            # STAGE 2: KEYFRAME EXTRACTION (DINOv2)
            logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 2: Keyframe Extraction (DINOv2) ---")
            from src.vision.keyframe_extractor import DINOv2Context
            with DINOv2Context() as extractor:
                for idx, vid in enumerate(batch_vids, 1):
                    logger.info(f"[{idx}/{len(batch_vids)}] Extracting keyframes for: {vid} ...")
                    extractor.process_video(vid, force=force)
            free_memory()

            # STAGE 3: DENSE CAPTIONING (QWEN VLM)
            logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 3: Dense Captioning (Qwen VLM) ---")
            from src.vlm.segment_captioner import QwenCaptionerContext
            with QwenCaptionerContext() as captioner:
                for idx, vid in enumerate(batch_vids, 1):
                    logger.info(f"[{idx}/{len(batch_vids)}] Generating captions for: {vid} ...")
                    captioner.process_video(vid, force=force)
            free_memory()

            # STAGE 4: EMBEDDING (QWEN3-VL-EMBEDDER)
            logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 4: Multi-modal Embedding (Qwen3-VL-Embedder) ---")
            from src.embedding.qwen3_embedder import EmbedderContext
            with EmbedderContext() as embedder:
                for idx, vid in enumerate(batch_vids, 1):
                    logger.info(f"[{idx}/{len(batch_vids)}] Generating embeddings for: {vid} ...")
                    embedder.process_video(vid, force=force)
            free_memory()

            # STAGE 5: VECTOR DB INGESTION (QDRANT)
            logger.info(f"--- [Batch {batch_idx}/{total_batches}] STAGE 5: Vector DB Ingestion (Qdrant) ---")
            from src.storage.qdrant_manager import QdrantManager
            db = QdrantManager()
            for idx, vid in enumerate(batch_vids, 1):
                logger.info(f"[{idx}/{len(batch_vids)}] Ingesting vectors to Qdrant for: {vid} ...")
                db.upsert_video_embeddings(vid)

            logger.info(f"✨ Batch {batch_idx}/{total_batches} completed and ingested into DB successfully!")

            if drive_destination:
                upload_batch_to_google_drive(
                    batch_vids,
                    drive_destination=drive_destination,
                    delete_local=delete_local_after_upload,
                )

    # ================= 2. SINGLE-STAGE TARGETED EXECUTION =================
    else:
        # STAGE 1: VAD SPLITTING
        if stage in ["1", "vad"]:
            logger.info("--- STAGE 1: VAD Splitting ---")
            from src.audio.vad_splitter import VadVideoSplitter
            splitter = VadVideoSplitter()
            for idx, v_path in enumerate(raw_videos, 1):
                logger.info(f"[{idx}/{len(raw_videos)}] VAD Splitting: {v_path.name} ...")
                splitter.process_video(str(v_path), force=force)
            free_memory()

        # STAGE 1.5: TRANSCRIPT EXTRACTION
        if stage in ["1.5", "transcript"]:
            logger.info("--- STAGE 1.5: Transcript Extraction (PhoASR & Pyannote) ---")
            from src.audio.transcript_extractor import TranscriptContext
            with TranscriptContext() as transcriber:
                for idx, vid in enumerate(video_ids, 1):
                    logger.info(f"[{idx}/{len(video_ids)}] Extracting transcript for: {vid} ...")
                    transcriber.process_video(vid, force=force)
            free_memory()

        # STAGE 2: KEYFRAME EXTRACTION (DINOv2)
        if stage in ["2", "dino", "keyframe"]:
            logger.info("--- STAGE 2: Keyframe Extraction (DINOv2) ---")
            from src.vision.keyframe_extractor import DINOv2Context
            with DINOv2Context() as extractor:
                for idx, vid in enumerate(video_ids, 1):
                    logger.info(f"[{idx}/{len(video_ids)}] Extracting keyframes for: {vid} ...")
                    extractor.process_video(vid, force=force)
            free_memory()

        # STAGE 3: DENSE CAPTIONING (QWEN VLM)
        if stage in ["3", "vlm", "caption"]:
            logger.info("--- STAGE 3: Dense Captioning (Qwen VLM) ---")
            from src.vlm.segment_captioner import QwenCaptionerContext
            with QwenCaptionerContext() as captioner:
                for idx, vid in enumerate(video_ids, 1):
                    logger.info(f"[{idx}/{len(video_ids)}] Generating captions for: {vid} ...")
                    captioner.process_video(vid, force=force)
            free_memory()

        # STAGE 4: EMBEDDING (QWEN3-VL-EMBEDDER)
        if stage in ["4", "embed", "embedding"]:
            logger.info("--- STAGE 4: Multi-modal Embedding (Qwen3-VL-Embedder) ---")
            from src.embedding.qwen3_embedder import EmbedderContext
            with EmbedderContext() as embedder:
                for idx, vid in enumerate(video_ids, 1):
                    logger.info(f"[{idx}/{len(video_ids)}] Generating embeddings for: {vid} ...")
                    embedder.process_video(vid, force=force)
            free_memory()

        # STAGE 5: VECTOR DB INGESTION (QDRANT)
        if stage in ["5", "qdrant", "db"]:
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
        "--video-batch-size", "--batch-size",
        type=int,
        default=None,
        help="Number of videos to process end-to-end per batch (default: 5, 0 to process all together)",
    )
    parser.add_argument(
        "--no-transcript", "--skip-transcript",
        action="store_true",
        help="Skip Stage 1.5 transcript extraction in the pipeline",
    )
    parser.add_argument(
        "--drive-destination",
        type=str,
        default=None,
        help="rclone destination, e.g. gdrive:aic-output. Upload each completed batch.",
    )
    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help="Delete output/<video_id> only after its Google Drive upload is verified.",
    )
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
        force=args.force,
        video_batch_size=args.video_batch_size,
        skip_transcript=args.no_transcript,
        drive_destination=args.drive_destination,
        delete_local_after_upload=args.delete_local_after_upload,
    )
