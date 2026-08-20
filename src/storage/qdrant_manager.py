import os
import uuid
import json
import logging
import torch
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.config import config

logger = logging.getLogger("QdrantManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class QdrantManager:
    def __init__(self, collection_name=None, vector_dim=None):
        self.collection_name = collection_name if collection_name is not None else config.QDRANT_COLLECTION_NAME
        self.vector_dim = vector_dim if vector_dim is not None else config.VECTOR_DIM
        
        if hasattr(config, "QDRANT_URL") and config.QDRANT_URL:
            logger.info(f"Connecting to Qdrant Cloud (URL: {config.QDRANT_URL})...")
            self.client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=60
            )
        else:
            # Local In-Memory / File-based Storage (Kaggle friendly)
            db_path = Path(config.DB_DIR)
            db_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Connecting to Qdrant (Local Path: {db_path})...")
            self.client = QdrantClient(path=str(db_path))
        
        # Initialize collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Creating Collection '{self.collection_name}' with dim={self.vector_dim}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            logger.info("Collection created.")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def upsert_video_embeddings(self, video_id: str):
        """Reads embeddings.pt and metadata from disk and upserts into Qdrant."""
        video_dir = Path(config.OUTPUT_DIR) / video_id
        embed_dir = video_dir / "embeddings"
        
        if not embed_dir.exists():
            logger.warning(f"No embeddings directory found for {video_id}.")
            return
            
        points = []
        
        # 1. Upsert Keyframes
        kf_meta_file = embed_dir / "keyframe_metadata.json"
        kf_vec_file = embed_dir / "keyframe_vectors.pt"
        if kf_meta_file.exists() and kf_vec_file.exists():
            logger.info(f"Loading keyframe embeddings for {video_id}...")
            vectors = torch.load(kf_vec_file).numpy()
            with open(kf_meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            if len(vectors) == len(metadata):
                for vec, meta in zip(vectors, metadata):
                    point_key = f"{video_id}_{meta.get('segment_id', '')}_kf_{meta.get('frame_index', 0)}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key))
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vec.tolist(),
                            payload=meta
                        )
                    )
            else:
                logger.error(f"Mismatch in keyframe vectors ({len(vectors)}) and metadata ({len(metadata)}).")

        # 2. Upsert Captions
        cap_meta_file = embed_dir / "caption_metadata.json"
        cap_vec_file = embed_dir / "caption_vectors.pt"
        if cap_meta_file.exists() and cap_vec_file.exists():
            logger.info(f"Loading caption embeddings for {video_id}...")
            vectors = torch.load(cap_vec_file).numpy()
            with open(cap_meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            if len(vectors) == len(metadata):
                for vec, meta in zip(vectors, metadata):
                    point_key = f"{video_id}_{meta.get('segment_id', '')}_cap_{meta.get('frame_index', 0)}"
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_key))
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vec.tolist(),
                            payload=meta
                        )
                    )
            else:
                logger.error(f"Mismatch in caption vectors ({len(vectors)}) and metadata ({len(metadata)}).")

        # 3. Batch Upsert to Qdrant
        if points:
            logger.info(f"Upserting {len(points)} points to Qdrant...")
            try:
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True,
                    points=points
                )
                logger.info(f"Upsert successful: {operation_info.status.name}")
            except Exception as e:
                logger.error(f"Failed to upsert points: {e}")
        else:
            logger.info("No points to upsert.")

# Quick Testing
if __name__ == "__main__":
    db = QdrantManager()
    # db.upsert_video_embeddings("vid_sample_001")
