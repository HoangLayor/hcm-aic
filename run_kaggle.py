import os
import gc
import torch
import logging
from src.config import config

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KaggleOrchestrator")

def free_memory():
    """Utility to aggressively free VRAM and RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory freed.")

def run_staged_pipeline():
    """
    Staged Execution Pipeline designed for Kaggle/Colab (VRAM < 16GB).
    Runs one heavy model at a time, clears memory, then proceeds.
    """
    logger.info("=== STARTING KAGGLE PIPELINE ===")
    
    # 1. AUDIO VAD SPLITTER STAGE
    logger.info("--- STAGE 1: VAD Splitting ---")
    from src.audio.vad_splitter import VadVideoSplitter
    splitter = VadVideoSplitter()
    # Replace with logic to loop over multiple raw files
    # splitter.process_video(raw_video_path)
    free_memory()

    # 2. KEYFRAME EXTRACTION STAGE
    logger.info("--- STAGE 2: Keyframe Extraction (DINOv2) ---")
    from src.vision.keyframe_extractor import DINOv2Context
    with DINOv2Context() as extractor:
        # extractor.process_video(video_id)
        pass
    free_memory()

    # 3. DENSE CAPTIONING STAGE
    logger.info("--- STAGE 3: Dense Captioning (Qwen3.5-2B) ---")
    from src.vlm.segment_captioner import QwenCaptionerContext
    with QwenCaptionerContext() as captioner:
        # captioner.process_video(video_id)
        pass
    free_memory()

    # 4. EMBEDDING STAGE
    logger.info("--- STAGE 4: Multi-modal Embedding (Qwen3-VL-Embedder) ---")
    from src.embedding.qwen3_embedder import EmbedderContext
    with EmbedderContext() as embedder:
        # embedder.process_video(video_id)
        pass
    free_memory()

    # 5. QDRANT INGESTION STAGE
    logger.info("--- STAGE 5: Vector DB Ingestion (Qdrant) ---")
    from src.storage.qdrant_manager import QdrantManager
    db = QdrantManager()
    # db.upsert_video_embeddings(video_id)
    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    # Ensures the necessary output directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DB_DIR, exist_ok=True)
    
    run_staged_pipeline()
