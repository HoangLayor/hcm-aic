import logging
import torch
from typing import List, Dict, Any, Optional
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

RRF_K = 60


def _split_collections(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(c).strip() for c in raw if str(c).strip()]
    return [c.strip() for c in str(raw).split(",") if c.strip()]


class SearchRetriever:
    def __init__(self, keyframe_collections=None, caption_collections=None):
        """Search engine over the Qdrant collections written by Stage 5.

        Connects to Qdrant Cloud when config.QDRANT_URL is set (same rule as
        QdrantManager), otherwise falls back to the local file-based DB.
        """
        self.kf_collections = _split_collections(
            keyframe_collections if keyframe_collections is not None
            else getattr(config, "SEARCH_KEYFRAME_COLLECTIONS", config.QDRANT_KEYFRAME_COLLECTION)
        )
        self.cap_collections = _split_collections(
            caption_collections if caption_collections is not None
            else getattr(config, "SEARCH_CAPTION_COLLECTIONS", config.QDRANT_CAPTION_COLLECTION)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if getattr(config, "QDRANT_URL", None):
            logger.info(f"Connecting to Qdrant Cloud ({config.QDRANT_URL})...")
            self.qdrant = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=60
            )
        else:
            logger.info(f"Connecting to local Qdrant ({config.DB_DIR})...")
            self.qdrant = QdrantClient(path=str(config.DB_DIR))

        # Drop collections that do not exist so a partial DB does not break search.
        self.kf_collections = self._existing(self.kf_collections)
        self.cap_collections = self._existing(self.cap_collections)
        logger.info(f"Keyframe collections: {self.kf_collections} | Caption collections: {self.cap_collections}")

        logger.info(f"Loading query embedder ({config.EMBEDDER_MODEL_ID})...")
        try:
            self.embedder = SentenceTransformer(config.EMBEDDER_MODEL_ID, device=self.device)
            if self.device == "cuda":
                self.embedder.half()
        except Exception as e:
            logger.error(f"Failed to load query embedder: {e}")
            raise

    def _existing(self, names: List[str]) -> List[str]:
        alive = []
        for name in names:
            try:
                if self.qdrant.collection_exists(collection_name=name):
                    alive.append(name)
                else:
                    logger.warning(f"Collection '{name}' not found, skipping.")
            except Exception as e:
                logger.warning(f"Could not check collection '{name}': {e}")
        return alive

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

    def _query(self, collection: str, vector: list, limit: int, query_filter=None) -> List[Any]:
        """Single-collection vector search (query_points API, search() fallback)."""
        try:
            response = self.qdrant.query_points(
                collection_name=collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(response.points)
        except AttributeError:
            return list(self.qdrant.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            ))
        except Exception as e:
            logger.error(f"Search on '{collection}' failed: {e}")
            return []

    @staticmethod
    def _video_filter(video_id: Optional[str]):
        if not video_id:
            return None
        return Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])

    @staticmethod
    def _segment_key(payload: dict) -> tuple:
        return (payload.get("video_id"), payload.get("segment_id"))

    def kis_search(self, query: str, top_k: int = 100, video_id: Optional[str] = None,
                   candidates_per_collection: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Known Item Search (KIS):
        Queries every keyframe and caption collection, then fuses the two modalities
        with Reciprocal Rank Fusion at the (video_id, segment_id) level.
        """
        logger.info(f"KIS Search: '{query}'")
        query_vector = self._encode_text(query)
        flt = self._video_filter(video_id)
        per_collection = candidates_per_collection or max(top_k, 50)

        fused: Dict[tuple, Dict[str, Any]] = {}
        for collection in self.kf_collections + self.cap_collections:
            points = self._query(collection, query_vector, per_collection, flt)
            for rank, pt in enumerate(points, start=1):
                payload = pt.payload or {}
                key = self._segment_key(payload)
                entry = fused.setdefault(key, {
                    "rrf_score": 0.0,
                    "video_id": payload.get("video_id"),
                    "segment_id": payload.get("segment_id"),
                    "start_time_sec": payload.get("start_time_sec"),
                    "end_time_sec": payload.get("end_time_sec"),
                    "frame_index": payload.get("frame_index"),
                    "image_path": None,
                    "text": None,
                    "keyframe_score": None,
                    "caption_score": None,
                    "sources": [],
                })
                entry["rrf_score"] += 1.0 / (RRF_K + rank)
                entry["sources"].append(collection)

                if payload.get("point_type") == "caption" or payload.get("text"):
                    if entry["caption_score"] is None or pt.score > entry["caption_score"]:
                        entry["caption_score"] = pt.score
                        entry["text"] = payload.get("text")
                else:
                    if entry["keyframe_score"] is None or pt.score > entry["keyframe_score"]:
                        entry["keyframe_score"] = pt.score
                        entry["image_path"] = payload.get("image_path")
                        entry["frame_index"] = payload.get("frame_index")

        ranked = sorted(fused.values(), key=lambda e: e["rrf_score"], reverse=True)[:top_k]
        results = []
        for rank, entry in enumerate(ranked, start=1):
            modality_scores = [s for s in (entry["keyframe_score"], entry["caption_score"]) if s is not None]
            entry["rank"] = rank
            entry["rrf_score"] = round(entry["rrf_score"], 6)
            entry["score"] = round(max(modality_scores), 4) if modality_scores else 0.0
            entry["sources"] = sorted(set(entry["sources"]))
            results.append(entry)
        logger.info(f"KIS returned {len(results)} fused segments.")
        return results

    def trake_search(self, queries: List[str], top_k_per_query: int = 50,
                     top_k: int = 20) -> List[Dict]:
        """
        Temporal Event Tracking (TRAKE):
        Takes a list of chronological events (E1, E2, E3...) and finds a matching
        sequence within the SAME video where frame_E1 < frame_E2 < frame_E3.
        """
        if not queries:
            return []
        logger.info(f"TRAKE Search for {len(queries)} sequential events.")

        # 1. Fetch top keyframe candidates for each event, grouped per video.
        per_video: Dict[str, List[List[Dict[str, Any]]]] = {}
        for step, q in enumerate(queries):
            vec = self._encode_text(q)
            points = []
            for collection in self.kf_collections:
                points.extend(self._query(collection, vec, top_k_per_query))
            for pt in points:
                payload = pt.payload or {}
                vid = payload.get("video_id")
                if vid is None or payload.get("frame_index") is None:
                    continue
                steps = per_video.setdefault(vid, [[] for _ in queries])
                steps[step].append({
                    "frame_index": payload.get("frame_index"),
                    "segment_id": payload.get("segment_id"),
                    "image_path": payload.get("image_path"),
                    "start_time_sec": payload.get("start_time_sec"),
                    "end_time_sec": payload.get("end_time_sec"),
                    "score": pt.score,
                })

        # 2. Temporal alignment: best strictly-increasing chain per video (DP over steps).
        valid_paths = []
        for vid, steps in per_video.items():
            if any(not c for c in steps):
                continue  # this video misses at least one event
            for c in steps:
                c.sort(key=lambda x: x["frame_index"])

            # best = list of (total_score, chain) for chains ending at the current step
            best = [(c["score"], [c]) for c in steps[0]]
            for step in range(1, len(steps)):
                new_best = []
                for cand in steps[step]:
                    feasible = [b for b in best if b[1][-1]["frame_index"] < cand["frame_index"]]
                    if not feasible:
                        continue
                    prev_score, prev_chain = max(feasible, key=lambda b: b[0])
                    new_best.append((prev_score + cand["score"], prev_chain + [cand]))
                best = new_best
                if not best:
                    break
            if not best:
                continue

            total, chain = max(best, key=lambda b: b[0])
            valid_paths.append({
                "video_id": vid,
                "score": round(total / len(queries), 4),
                "total_score": round(total, 4),
                "events": [
                    dict({"event_index": i, "query": queries[i]}, **c)
                    for i, c in enumerate(chain)
                ],
            })

        valid_paths.sort(key=lambda p: p["total_score"], reverse=True)
        valid_paths = valid_paths[:top_k]
        for rank, p in enumerate(valid_paths, start=1):
            p["rank"] = rank
        logger.info(f"TRAKE found {len(valid_paths)} valid chronological paths.")
        return valid_paths

    def qa_search_context(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Video Question Answering (QA):
        Finds the top_k most relevant video segments that might contain the answer.
        The result should be passed to a VLM (like Qwen3.5-2B) to generate the final text answer.
        """
        logger.info(f"QA Search Context Gathering: '{question}'")
        return self.kis_search(question, top_k=top_k)


# Quick Testing / CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the AIC Qdrant collections.")
    parser.add_argument("query", nargs="+", help="Query text. Pass several for TRAKE mode.")
    parser.add_argument("--mode", choices=["kis", "trake", "qa"], default="kis")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--video-id", default=None, help="Restrict KIS search to one video.")
    args = parser.parse_args()

    retriever = SearchRetriever()
    if args.mode == "trake":
        for path in retriever.trake_search(args.query, top_k=args.top_k):
            print("=" * 80)
            print(f"Rank {path['rank']} | Score {path['score']} | Video {path['video_id']}")
            for ev in path["events"]:
                print(f"  E{ev['event_index'] + 1} frame={ev['frame_index']} score={ev['score']:.4f} {ev['image_path']}")
    else:
        query = " ".join(args.query)
        results = (retriever.qa_search_context(query, top_k=args.top_k) if args.mode == "qa"
                   else retriever.kis_search(query, top_k=args.top_k, video_id=args.video_id))
        for r in results:
            print("=" * 80)
            print(f"Rank {r['rank']} | RRF {r['rrf_score']} | Cosine {r['score']} | {r['video_id']} / {r['segment_id']}")
            print(f"  frame={r['frame_index']} time={r['start_time_sec']}-{r['end_time_sec']}s")
            print(f"  image: {r['image_path']}")
            if r["text"]:
                print(f"  caption: {r['text'].strip()}")
