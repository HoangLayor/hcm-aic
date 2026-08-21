import logging
import argparse
import torch
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger("SearchRetrieverKaggle")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class SearchRetriever:
    def __init__(self, kf_collection="keyframes_merged", cap_collection="captions_merged"):
        self.kf_collection = kf_collection
        self.cap_collection = cap_collection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Connect to Qdrant (Prefer Cloud for Kaggle)
        if getattr(config, "QDRANT_URL", None) and getattr(config, "QDRANT_API_KEY", None):
            logger.info(f"Connecting to Qdrant Cloud (URL: {config.QDRANT_URL})...")
            self.qdrant = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=60
            )
        else:
            logger.info(f"Connecting to local Qdrant ({config.DB_DIR})...")
            self.qdrant = QdrantClient(path=str(config.DB_DIR))
        
        # 2. Load Embedder
        logger.info(f"Loading query embedder ({config.EMBEDDER_MODEL_ID})...")
        try:
            self.embedder = SentenceTransformer(config.EMBEDDER_MODEL_ID, device=self.device)
            if self.device == "cuda":
                self.embedder.half()
        except Exception as e:
            logger.error(f"Failed to load query embedder: {e}")
            raise

    def free_memory(self):
        """Free GPU memory if this class is destroyed."""
        if hasattr(self, 'embedder'):
            del self.embedder
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SearchRetriever memory freed.")

    def _encode_text(self, text: str) -> list:
        """Helper to encode a single text query."""
        vec = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.tolist()

    def kis_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Known Item Search (KIS):
        Searches the query against Keyframes AND Captions (from separated collections),
        then merges and sorts the results by score.
        """
        logger.info(f"KIS Search: '{query}'")
        query_vector = self._encode_text(query)
        
        combined_results = []
        
        # Search Keyframes
        try:
            kf_results = self.qdrant.search(
                collection_name=self.kf_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            combined_results.extend(kf_results)
        except Exception as e:
            logger.error(f"Failed to search {self.kf_collection}: {e}")

        # Search Captions
        try:
            cap_results = self.qdrant.search(
                collection_name=self.cap_collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            combined_results.extend(cap_results)
        except Exception as e:
            logger.error(f"Failed to search {self.cap_collection}: {e}")

        # Merge, Sort and Slice top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        final_results = combined_results[:top_k]
            
        formatted_results = []
        for rank, pt in enumerate(final_results, start=1):
            formatted_results.append({
                "rank": rank,
                "score": round(pt.score, 4),
                "video_id": pt.payload.get("video_id"),
                "frame_index": pt.payload.get("frame_index"),
                "point_type": pt.payload.get("point_type", "unknown"),
                "image_path": pt.payload.get("image_path"),
                "text_caption": pt.payload.get("text") or pt.payload.get("caption")
            })
            
        return formatted_results

    def trake_search(self, queries: List[str], top_k_per_query: int = 10) -> List[Dict]:
        """
        Temporal Event Tracking (TRAKE):
        Takes a list of chronological events and queries only Keyframes.
        """
        logger.info(f"TRAKE Search for {len(queries)} sequential events.")
        
        event_candidates = []
        for i, q in enumerate(queries):
            vec = self._encode_text(q)
            try:
                results = self.qdrant.search(
                    collection_name=self.kf_collection,
                    query_vector=vec,
                    limit=top_k_per_query,
                    with_payload=True
                )
                event_candidates.append(results)
            except Exception as e:
                logger.error(f"Failed to search {self.kf_collection} for TRAKE: {e}")
                event_candidates.append([])
            
        valid_paths = []
        logger.info("Running Temporal Alignment on candidates...")
        # ... logic to find valid (E1 < E2 < E3) frame sequences within same video ...
        
        return valid_paths

    def qa_search_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Video Question Answering (QA):
        Finds the top_k most relevant video segments that might contain the answer.
        """
        logger.info(f"QA Search Context Gathering: '{question}'")
        return self.kis_search(question, top_k=top_k)

# Quick Testing via CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Search Retriever")
    parser.add_argument("--query", "-q", type=str, required=True, help="Text query to search for")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    retriever = SearchRetriever()
    results = retriever.kis_search(args.query, top_k=args.top_k)
    
    print("\n" + "="*50)
    print(f"RESULTS FOR: '{args.query}'")
    print("="*50)
    
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Video: {r['video_id']} | Frame: {r['frame_index']} | Type: {r['point_type']}")
