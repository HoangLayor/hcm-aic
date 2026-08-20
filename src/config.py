import os
from pydantic import BaseSettings, Field

class AppConfig(BaseSettings):
    # Global Pipeline Settings
    TARGET_FPS: int = Field(default=30, description="Target FPS to avoid frame drift.")
    USE_TRANSCRIPT_BRANCH: bool = Field(default=False, description="Enable ASR transcript collection.")
    
    # Storage Paths
    RAW_DIR: str = Field(default="./raw", description="Directory containing raw MP4 video files.")
    OUTPUT_DIR: str = Field(default="./output", description="Directory to store intermediate artifacts.")
    DB_DIR: str = Field(default="./qdrant_db", description="Directory for Qdrant local storage.")
    
    # 1. VAD & Splitter Configuration
    VAD_MAX_SEGMENT_DURATION_SEC: int = Field(default=30, description="Maximum duration per segment in seconds.")
    VAD_MIN_SILENCE_MS: int = Field(default=1000, description="Minimum silence duration to trigger split.")
    VAD_THRESHOLD: float = Field(default=0.5, description="Silence detection threshold.")
    VAD_MIN_SPEECH_DURATION_MS: int = Field(default=250, description="Minimum speech duration.")
    
    # 2. Keyframe Extraction (DINOv2) Configuration
    DINO_MODEL_ID: str = "facebookresearch/dinov2_vitb14"
    DINO_BATCH_SIZE: int = Field(default=16, description="Batch size for DINOv2.")
    DINO_SIMILARITY_THRESHOLD: float = Field(default=0.65, description="Cosine threshold to group frames.")
    DINO_MAX_CANDIDATES: int = Field(default=10, description="Max candidate frames per group.")
    
    # 3. Dense Captioning (Qwen3.5-2B) Configuration
    VLM_MODEL_ID: str = "Qwen/Qwen3.5-2B"
    VLM_VIDEO_FPS: float = Field(default=1.0, description="Sampling rate (frames per sec) for VLM.")
    VLM_MAX_VIDEO_FRAMES: int = Field(default=32, description="Max frames allowed in VLM input.")
    VLM_MAX_NEW_TOKENS: int = Field(default=256, description="Max caption length.")
    
    # 4. Multi-modal Embedding (Qwen3-VL-Embedding)
    EMBEDDER_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
    EMBEDDER_BATCH_SIZE: int = Field(default=8, description="Batch size for Embedder.")
    IMAGE_RESIZE_PX: int = Field(default=512, description="Resize keyframes before embedding to save VRAM.")

    class Config:
        env_prefix = "AIC_"
        case_sensitive = True

# Global singleton configuration object
config = AppConfig()
