import logging
import torch
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.config import config

logger = logging.getLogger("SearchRetriever")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(ch)

class SearchRetriever:
    def __init__(self, collection_name="aic2026_video_retrieval"):
        self.collection_name = collection_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Connecting to local Qdrant ({config.DB_DIR})...")
        self.qdrant = QdrantClient(path=str(config.DB_DIR))
        
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
        Searches the query against Keyframes and Captions.
        """
        logger.info(f"KIS Search: '{query}'")
        query_vector = self._encode_text(query)
        
        try:
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            formatted_results = []
            for rank, pt in enumerate(results, start=1):
                formatted_results.append({
                    "rank": rank,
                    "score": round(pt.score, 4),
                    "video_id": pt.payload.get("video_id"),
                    "frame_index": pt.payload.get("frame_index"),
                    "point_type": pt.payload.get("point_type"), # 'keyframe' or 'caption'
                    "image_path": pt.payload.get("image_path"),
                    "text_caption": pt.payload.get("caption")
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"KIS Search failed: {e}")
            return []

    def trake_search(self, queries: List[str], top_k_per_query: int = 10) -> List[Dict]:
        """
        Temporal Event Tracking (TRAKE):
        Takes a list of chronological events (E1, E2, E3...) and finds a matching
        sequence within the SAME video where frame_E1 < frame_E2 < frame_E3.
        """
        logger.info(f"TRAKE Search for {len(queries)} sequential events.")
        
        # 1. Fetch top candidates for each event
        event_candidates = []
        for i, q in enumerate(queries):
            vec = self._encode_text(q)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vec,
                limit=top_k_per_query,
                query_filter=Filter(
                    must=[FieldCondition(key="point_type", match=MatchValue(value="keyframe"))]
                ),
                with_payload=True
            )
            event_candidates.append(results)
            
        # 2. Temporal Alignment Logic (Sliding Window Algorithm)
        # Note: Implementing a simplified dynamic programming or sliding window here.
        # For Kaggle simplicity, we can group by video_id and check valid chronological paths.
        
        valid_paths = []
        # --- Simplified Temporal Alignment Mock for Skeleton ---
        logger.info("Running Temporal Alignment on candidates...")
        # ... logic to find valid (E1 < E2 < E3) frame sequences within same video ...
        
        return valid_paths

    def qa_search_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Video Question Answering (QA):
        Finds the top_k most relevant video segments that might contain the answer.
        The result should be passed to a VLM (like Qwen3.5-2B) to generate the final text answer.
        """
        logger.info(f"QA Search Context Gathering: '{question}'")
        return self.kis_search(question, top_k=top_k)

# Quick Testing
if __name__ == "__main__":
    # retriever = SearchRetriever()
    # res = retriever.kis_search("Đĩa gỏi cuốn chay hoa pansy", top_k=5)
    # for r in res:
    #     print(f"Rank {r['rank']} | Score: {r['score']} | Video: {r['video_id']} | Frame: {r['frame_index']}")
    pass
