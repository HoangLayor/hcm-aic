import os
import sys
import json

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from qdrant_client import QdrantClient

def print_collections_info(client, db_type):
    print("==================================================")
    print(f"Analyzing {db_type} Qdrant Database")
    print("==================================================")
    
    try:
        collections_response = client.get_collections()
        collections = collections_response.collections
        
        if not collections:
            print(f"No collections found in the {db_type} database.\n")
            return
            
        print(f"Found {len(collections)} collection(s).\n")
        
        for collection in collections:
            collection_name = collection.name
            print(f"--- Collection: {collection_name} ---")
            
            # Get detailed info about the collection
            info = client.get_collection(collection_name)
            print(f"  Status:       {info.status}")
            print(f"  Points count: {info.points_count}")
            if info.config and info.config.params and info.config.params.vectors:
                # Handle cases where vectors is a dictionary of named vectors or a single vector config
                if isinstance(info.config.params.vectors, dict):
                    print(f"  Vectors config:")
                    for v_name, v_conf in info.config.params.vectors.items():
                        print(f"    - {v_name}: size={v_conf.size}, distance={v_conf.distance}")
                else:
                    print(f"  Vector size:  {info.config.params.vectors.size}")
                    print(f"  Distance:     {info.config.params.vectors.distance}")
            
            # Fetch one point to inspect its payload schema
            points, _ = client.scroll(collection_name=collection_name, limit=1, with_payload=True)
            if points:
                payload = points[0].payload
                print(f"  Payload keys: {list(payload.keys()) if payload else 'None'}")
                print("  Sample point payload:")
                print("    " + json.dumps(payload, ensure_ascii=False, indent=2).replace("\n", "\n    "))
            else:
                print("  Payload:      No points available to inspect payload.")
            print("")
                
    except Exception as e:
        print(f"Error accessing {db_type} Database: {e}\n")

def analyze_qdrant_db():
    # 1. Test Local DB
    print(f"Testing Local Qdrant at: {config.DB_DIR}")
    if os.path.exists(config.DB_DIR):
        local_client = QdrantClient(path=str(config.DB_DIR))
        print_collections_info(local_client, "Local")
    else:
        print(f"Local Qdrant DB directory '{config.DB_DIR}' does not exist.\n")

    # 2. Test Cloud DB (from config)
    if config.QDRANT_URL and config.QDRANT_API_KEY:
        print(f"Testing Cloud Qdrant at: {config.QDRANT_URL}")
        try:
            cloud_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY
            )
            print_collections_info(cloud_client, "Cloud")
        except Exception as e:
            print(f"Failed to connect to Cloud Qdrant: {e}")

if __name__ == "__main__":
    analyze_qdrant_db()
