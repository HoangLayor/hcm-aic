import os
from pydantic import Field

try:
    # 1. Try Pydantic v2 (pydantic-settings package)
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class AppConfig(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="AIC_", case_sensitive=True, extra="ignore")

        # Global Pipeline Settings
        TARGET_FPS: int = Field(default=30, description="Target FPS to avoid frame drift.")
        USE_TRANSCRIPT_BRANCH: bool = Field(default=False, description="Enable ASR transcript collection.")
        
        # Storage Paths
        RAW_DIR: str = Field(default="./data", description="Directory containing raw MP4 video files.")
        OUTPUT_DIR: str = Field(default="./output", description="Directory to store intermediate artifacts.")
        DB_DIR: str = Field(default="./qdrant_db", description="Directory for Qdrant local storage.")
        
        QDRANT_URL: str = Field(default="https://4ae329d5-5ea2-466b-a1a4-ff1d8754a68a.sa-east-1-0.aws.cloud.qdrant.io", description="Qdrant Cloud URL.")
        QDRANT_API_KEY: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDYyZjRkYmItYjU4Yi00ODhhLTk4ZDEtY2FiNzAwNmNmOTJjIn0.RDCtpTqEcEH5BnV3FKiRQo3J1SjzOYj0KtM00yK_H44", description="Qdrant Cloud API Key.")
        QDRANT_KEYFRAME_COLLECTION: str = Field(default="keyframes", description="Target Qdrant Collection for Keyframes.")
        QDRANT_CAPTION_COLLECTION: str = Field(default="captions", description="Target Qdrant Collection for Captions.")
        
        # 1. VAD & Splitter Configuration
        VAD_MAX_SEGMENT_DURATION_SEC: int = Field(default=30, description="Maximum duration per segment in seconds.")
        VAD_MIN_SILENCE_MS: int = Field(default=350, description="Minimum silence duration to trigger split.")
        VAD_THRESHOLD: float = Field(default=0.5, description="Silence detection threshold.")
        VAD_MIN_SPEECH_DURATION_MS: int = Field(default=250, description="Minimum speech duration.")
        VAD_SPEECH_PAD_MS: int = Field(default=300, description="Pad speech to avoid cutting words.")
        
        # 1.5. Transcript & Diarization Configuration
        ASR_MODEL_ID: str = "Qualcomm-AI-Research/PhoASR-whisper-small"
        DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-community-1"
        DIARIZATION_MIN_SPEAKERS: int = Field(default=1, description="Min speakers.")
        DIARIZATION_MAX_SPEAKERS: int = Field(default=6, description="Max speakers.")
        HF_TOKEN: str = Field(default="", description="HuggingFace Token for Pyannote")
        
        # 2. Keyframe Extraction (DINOv2) Configuration
        DINO_MODEL_ID: str = "dinov2_vitb14"
        DINO_BATCH_SIZE: int = Field(default=16, description="Batch size for DINOv2.")
        DINO_SIMILARITY_THRESHOLD: float = Field(default=0.65, description="Cosine threshold to group frames.")
        DINO_MAX_CANDIDATES: int = Field(default=10, description="Max candidate frames per group.")
        
        # 3. Dense Captioning (Qwen3.5-4B) Configuration
        VLM_MODEL_ID: str = "Qwen/Qwen3.5-4B"
        VLM_VIDEO_FPS: float = Field(default=1.0, description="Sampling rate (frames per sec) for VLM.")
        VLM_MAX_VIDEO_FRAMES: int = Field(default=32, description="Max frames allowed in VLM input.")
        VLM_MAX_NEW_TOKENS: int = Field(default=256, description="Max caption length.")
        
        # 4. Multi-modal Embedding (Qwen3-VL-Embedding)
        EMBEDDER_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
        EMBEDDER_BATCH_SIZE: int = Field(default=8, description="Batch size for Embedder.")
        IMAGE_RESIZE_PX: int = Field(default=512, description="Resize keyframes before embedding to save VRAM.")
        VECTOR_DIM: int = Field(default=2048, description="Output embedding dimension for Qdrant vector collection.")

except ImportError:
    try:
        # 2. Try Pydantic v1
        from pydantic import BaseSettings

        class AppConfig(BaseSettings):
            TARGET_FPS: int = Field(default=30, description="Target FPS to avoid frame drift.")
            USE_TRANSCRIPT_BRANCH: bool = Field(default=False, description="Enable ASR transcript collection.")
            
            RAW_DIR: str = Field(default="./data", description="Directory containing raw MP4 video files.")
            OUTPUT_DIR: str = Field(default="./output", description="Directory to store intermediate artifacts.")
            DB_DIR: str = Field(default="./qdrant_db", description="Directory for Qdrant local storage.")
            
            QDRANT_URL: str = Field(default="https://4ae329d5-5ea2-466b-a1a4-ff1d8754a68a.sa-east-1-0.aws.cloud.qdrant.io", description="Qdrant Cloud URL.")
            QDRANT_API_KEY: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDYyZjRkYmItYjU4Yi00ODhhLTk4ZDEtY2FiNzAwNmNmOTJjIn0.RDCtpTqEcEH5BnV3FKiRQo3J1SjzOYj0KtM00yK_H44", description="Qdrant Cloud API Key.")
            QDRANT_KEYFRAME_COLLECTION: str = Field(default="keyframes", description="Target Qdrant Collection for Keyframes.")
            QDRANT_CAPTION_COLLECTION: str = Field(default="captions", description="Target Qdrant Collection for Captions.")
            
            VAD_MAX_SEGMENT_DURATION_SEC: int = Field(default=30, description="Maximum duration per segment in seconds.")
            VAD_MIN_SILENCE_MS: int = Field(default=1000, description="Minimum silence duration to trigger split.")
            VAD_THRESHOLD: float = Field(default=0.5, description="Silence detection threshold.")
            VAD_MIN_SPEECH_DURATION_MS: int = Field(default=250, description="Minimum speech duration.")
            
            DIARIZATION_MIN_SPEAKERS: int = Field(default=1, description="Min speakers.")
            DIARIZATION_MAX_SPEAKERS: int = Field(default=6, description="Max speakers.")
            
            DINO_MODEL_ID: str = "dinov2_vitb14"
            DINO_BATCH_SIZE: int = Field(default=16, description="Batch size for DINOv2.")
            DINO_SIMILARITY_THRESHOLD: float = Field(default=0.65, description="Cosine threshold to group frames.")
            DINO_MAX_CANDIDATES: int = Field(default=10, description="Max candidate frames per group.")
            
            VLM_MODEL_ID: str = "Qwen/Qwen3.5-4B"
            VLM_VIDEO_FPS: float = Field(default=1.0, description="Sampling rate (frames per sec) for VLM.")
            VLM_MAX_VIDEO_FRAMES: int = Field(default=32, description="Max frames allowed in VLM input.")
            VLM_MAX_NEW_TOKENS: int = Field(default=256, description="Max caption length.")
            
            EMBEDDER_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
            EMBEDDER_BATCH_SIZE: int = Field(default=8, description="Batch size for Embedder.")
            IMAGE_RESIZE_PX: int = Field(default=512, description="Resize keyframes before embedding to save VRAM.")
            VECTOR_DIM: int = Field(default=2048, description="Output embedding dimension for Qdrant vector collection.")

            class Config:
                env_prefix = "AIC_"
                case_sensitive = True

    except Exception:
        # 3. Fallback to standard BaseModel with os.getenv
        from pydantic import BaseModel

        class AppConfig(BaseModel):
            TARGET_FPS: int = int(os.getenv("AIC_TARGET_FPS", 30))
            USE_TRANSCRIPT_BRANCH: bool = os.getenv("AIC_USE_TRANSCRIPT_BRANCH", "False").lower() in ("true", "1")
            
            RAW_DIR: str = os.getenv("AIC_RAW_DIR", "./data")
            OUTPUT_DIR: str = os.getenv("AIC_OUTPUT_DIR", "./output")
            DB_DIR: str = os.getenv("AIC_DB_DIR", "./qdrant_db")
            
            QDRANT_URL: str = os.getenv("AIC_QDRANT_URL", "https://4ae329d5-5ea2-466b-a1a4-ff1d8754a68a.sa-east-1-0.aws.cloud.qdrant.io")
            QDRANT_API_KEY: str = os.getenv("AIC_QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDYyZjRkYmItYjU4Yi00ODhhLTk4ZDEtY2FiNzAwNmNmOTJjIn0.RDCtpTqEcEH5BnV3FKiRQo3J1SjzOYj0KtM00yK_H44")
            QDRANT_KEYFRAME_COLLECTION: str = os.getenv("AIC_QDRANT_KEYFRAME_COLLECTION", "keyframes")
            QDRANT_CAPTION_COLLECTION: str = os.getenv("AIC_QDRANT_CAPTION_COLLECTION", "captions")
            
            VAD_MAX_SEGMENT_DURATION_SEC: int = int(os.getenv("AIC_VAD_MAX_SEGMENT_DURATION_SEC", 30))
            VAD_MIN_SILENCE_MS: int = int(os.getenv("AIC_VAD_MIN_SILENCE_MS", 1000))
            VAD_THRESHOLD: float = float(os.getenv("AIC_VAD_THRESHOLD", 0.5))
            VAD_MIN_SPEECH_DURATION_MS: int = int(os.getenv("AIC_VAD_MIN_SPEECH_DURATION_MS", 250))
            
            DINO_MODEL_ID: str = os.getenv("AIC_DINO_MODEL_ID", "dinov2_vitb14")
            DINO_BATCH_SIZE: int = int(os.getenv("AIC_DINO_BATCH_SIZE", 16))
            DINO_SIMILARITY_THRESHOLD: float = float(os.getenv("AIC_DINO_SIMILARITY_THRESHOLD", 0.65))
            DINO_MAX_CANDIDATES: int = int(os.getenv("AIC_DINO_MAX_CANDIDATES", 10))
            
            VLM_MODEL_ID: str = os.getenv("AIC_VLM_MODEL_ID", "Qwen/Qwen3.5-4B")
            VLM_VIDEO_FPS: float = float(os.getenv("AIC_VLM_VIDEO_FPS", 1.0))
            VLM_MAX_VIDEO_FRAMES: int = int(os.getenv("AIC_VLM_MAX_VIDEO_FRAMES", 32))
            VLM_MAX_NEW_TOKENS: int = int(os.getenv("AIC_VLM_MAX_NEW_TOKENS", 256))
            
            EMBEDDER_MODEL_ID: str = os.getenv("AIC_EMBEDDER_MODEL_ID", "Qwen/Qwen3-VL-Embedding-2B")
            EMBEDDER_BATCH_SIZE: int = int(os.getenv("AIC_EMBEDDER_BATCH_SIZE", 8))
            IMAGE_RESIZE_PX: int = int(os.getenv("AIC_IMAGE_RESIZE_PX", 512))
            VECTOR_DIM: int = int(os.getenv("AIC_VECTOR_DIM", 2048))

# Global singleton configuration object
config = AppConfig()
