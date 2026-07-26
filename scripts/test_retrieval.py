import json
import sys
from pathlib import Path

import faiss

from sentence_transformers import SentenceTransformer


# ==========================================================
# Add project root to Python path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    FIXED_INDEX,
    FIXED_METADATA,
    EMBEDDING_MODEL,
    TOP_K
)


# ==========================================================
# Load FAISS Index
# ==========================================================

print("Loading FAISS index...")

index = faiss.read_index(
    FIXED_INDEX
)


print(
    "Vectors:",
    index.ntotal
)


# ==========================================================
# Load Metadata
# ==========================================================

print("\nLoading metadata...")


with open(
    FIXED_METADATA,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(
    "Metadata records:",
    len(metadata)
)


# ==========================================================
# Load Embedding Model
# ==========================================================

print("\nLoading embedding model...")


model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded"
)


# ==========================================================
# Query
# ==========================================================

query = input(
    "\nEnter your question: "
)


# ==========================================================
# Create Query Embedding
# ==========================================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ==========================================================
# Search FAISS
# ==========================================================

scores, indices = index.search(
    query_embedding,
    TOP_K
)


# ==========================================================
# Display Results
# ==========================================================

print("\nTop Results\n")


for rank, idx in enumerate(indices[0], start=1):

    print("=" * 60)

    print(
        "Rank:",
        rank
    )

    print(
        "Score:",
        scores[0][rank - 1]
    )

    print(
        "Title:",
        metadata[idx]["title"]
    )

    print(
        "Chunk ID:",
        metadata[idx]["chunk_id"]
    )

    print()

    print(
        metadata[idx]
    )