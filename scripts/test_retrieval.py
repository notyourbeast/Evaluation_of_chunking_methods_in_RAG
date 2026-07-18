import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ==============================
# Paths
# ==============================

INDEX_FILE = "models/fixed_256_faiss.index"

METADATA_FILE = (
    "data/processed/embeddings/fixed_256_metadata.json"
)


# ==============================
# Load FAISS
# ==============================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)

print("Vectors:", index.ntotal)


# ==============================
# Load metadata
# ==============================

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)


# ==============================
# Load embedding model
# ==============================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==============================
# Query
# ==============================

query = input("\nEnter your question: ")


# ==============================
# Create query embedding
# ==============================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)


# ==============================
# Search
# ==============================

k = 5

distances, indices = index.search(
    query_embedding,
    k
)


# ==============================
# Display results
# ==============================

print("\nTop Results\n")


for rank, idx in enumerate(indices[0], start=1):

    print("=" * 60)

    print("Rank:", rank)
    print("Distance:", distances[0][rank-1])

    print("Title:",
          metadata[idx]["title"])

    print("Chunk ID:",
          metadata[idx]["chunk_id"])

    print()

    print(
        metadata[idx]
    )

