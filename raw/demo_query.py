pip install qdrant-client torch tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
QDRANT_URL = "https://4ae329d5-5ea2-466b-a1a4-ff1d8754a68a.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDYyZjRkYmItYjU4Yi00ODhhLTk4ZDEtY2FiNzAwNmNmOTJjIn0.RDCtpTqEcEH5BnV3FKiRQo3J1SjzOYj0KtM00yK_H44"


# =========================
# CONFIG
# =========================


COLLECTION_NAME = "noisy_eggs"

# Phải là đúng model đã dùng để tạo embeddings.pt
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"


# =========================
# CONNECT QDRANT
# =========================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(MODEL_NAME)


# =========================
# QUERY
# =========================


query = "Find the specific short clip featuring three people (two women and one man) sitting side-by-side, focused on playing a round, hollow metal instrument with indentations that produce sound when struck by hand. One person wearing a white shirt is seated between two others wearing black shirts. The background features a multi-compartment bookshelf filled with colorful books."
COLLECTION_NAME = "noisy_eggs"
query_vector = model.encode(
    query,
    convert_to_tensor=False,
)

print("Query vector shape:", query_vector.shape)




results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector.tolist(),
    limit=10,
    with_payload=True,
)



for rank, point in enumerate(results.points, start=1):

    print("=" * 80)
    print(f"Rank: {rank}")
    print(f"Score: {point.score:.4f}")
    print(f"ID: {point.id}")

    print("Video:", point.payload.get("video_path"))
    print("Caption:", point.payload.get("caption"))
    print("Embedding index:", point.payload.get("embedding_index"))

COLLECTION_NAME = "aic_keyframé"
query = "Find the specific short clip featuring three people (two women and one man) sitting side-by-side, focused on playing a round, hollow metal instrument with indentations that produce sound when struck by hand. One person wearing a white shirt is seated between two others wearing black shirts. The background features a multi-compartment bookshelf filled with colorful books."


query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)


print("Query embedding shape:", query_embedding.shape)


# =========================
# Search keyframes
# =========================

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=10,
    with_payload=True,
)


# =========================
# Print results
# =========================

for rank, point in enumerate(results.points, start=1):

    payload = point.payload

    print("=" * 80)

    print(f"Rank: {rank}")
    print(f"Score: {point.score:.4f}")

    print("Video:", payload["video_id"])
    print("Clip:", payload["clip_id"])

    print("Keyframe:", payload["keyframe_index"])
    print("Frame:", payload["frame_index"])

    print("Image:", payload["relative_image_path"])
