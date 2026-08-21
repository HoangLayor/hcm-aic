import os
import sys
import uuid
import logging

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MergeCollections")

# Initialize Qdrant Client (Prefer Cloud if configured, else Local)
if getattr(config, "QDRANT_URL", None) and getattr(config, "QDRANT_API_KEY", None):
    logger.info(f"Connecting to Qdrant Cloud (URL: {config.QDRANT_URL})...")
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=120
    )
else:
    logger.info(f"Connecting to Qdrant Local (Path: {config.DB_DIR})...")
    client = QdrantClient(path=str(config.DB_DIR))


def copy_collection(source: str, target: str, version: str):
    logger.info(f"Starting to copy from '{source}' to '{target}' as version '{version}'")
    
    if not client.collection_exists(source):
        logger.warning(f"Source collection '{source}' does not exist. Skipping.")
        return

    offset = None
    batch_size = 256
    total_points = 0
    
    try:
        info = client.get_collection(source)
        total_expected = info.points_count
    except Exception:
        total_expected = "Unknown"

    while True:
        points, offset = client.scroll(
            collection_name=source,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        new_points = []
        for point in points:
            payload = point.payload or {}
            payload["version"] = version
            payload["original_id"] = str(point.id)

            # Generate a valid UUID using uuid5 based on the version and original ID
            new_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{version}_{point.id}"))

            new_points.append(
                PointStruct(
                    id=new_id,
                    vector=point.vector,
                    payload=payload,
                )
            )

        # Upsert the batch
        client.upsert(
            collection_name=target,
            points=new_points,
        )
        
        total_points += len(new_points)
        if total_points % 2560 == 0:
            logger.info(f"Copied {total_points} / {total_expected} points from '{source}'...")

        if offset is None:
            break
            
    logger.info(f"Successfully copied {total_points} points from '{source}' to '{target}'.")


def merge_collections(source_v1: str, source_v2: str, target: str, vector_size: int = 2048):
    logger.info("=" * 50)
    logger.info(f"Merging '{source_v1}' and '{source_v2}' into '{target}'")
    logger.info("=" * 50)
    
    # 1. Create target collection if not exists
    if not client.collection_exists(target):
        logger.info(f"Creating target collection '{target}' (vector_size={vector_size})...")
        client.create_collection(
            collection_name=target,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        # Create useful indexes for filtering
        logger.info("Creating payload indexes for 'video_id' and 'point_type'...")
        client.create_payload_index(target, "video_id", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(target, "point_type", field_schema=PayloadSchemaType.KEYWORD)
    else:
        logger.info(f"Target collection '{target}' already exists. Points will be appended/upserted.")

    # 2. Copy v1
    copy_collection(source_v1, target, version="v1")

    # 3. Copy v2
    copy_collection(source_v2, target, version="v2")
    
    # 4. Verify Final Count
    try:
        final_info = client.get_collection(target)
        logger.info(f"Merge Complete! Collection '{target}' now has {final_info.points_count} points.\n")
    except Exception as e:
        logger.warning(f"Could not verify final points count: {e}")


def main():
    VECTOR_SIZE = getattr(config, "VECTOR_DIM", 2048)
    
    # Merge Keyframes
    merge_collections(
        source_v1="keyframes",
        source_v2="keyframes_v2",
        target="keyframes_merged",
        vector_size=VECTOR_SIZE
    )
    
    # Merge Captions
    merge_collections(
        source_v1="captions",
        source_v2="captions_v2",
        target="captions_merged",
        vector_size=VECTOR_SIZE
    )
    
    logger.info("All merge operations finished successfully.")

if __name__ == "__main__":
    main()
