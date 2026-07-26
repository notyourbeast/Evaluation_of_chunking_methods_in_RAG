import json
import os
import sys
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    FIXED_CHUNK_FILE,
    FIXED_EMBEDDINGS,
    FIXED_METADATA,
    EMBEDDING_MODEL,
    EMBEDDING_DIR
)


# ==========================================================
# Load embedding model
# ==========================================================

print("Loading embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded")


# ==========================================================
# Load chunks
# ==========================================================

print("\nLoading chunks...")

with open(
    FIXED_CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    "Chunks loaded:",
    len(chunks)
)


texts = [
    chunk["text"]
    for chunk in chunks
]


# ==========================================================
# Generate embeddings
# ==========================================================

print("\nGenerating embeddings...")


embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================================
# Save embeddings
# ==========================================================

os.makedirs(
    EMBEDDING_DIR,
    exist_ok=True
)


np.save(
    FIXED_EMBEDDINGS,
    embeddings
)


# ==========================================================
# Save metadata
# ==========================================================

metadata = []


for chunk in chunks:

    metadata.append(
        {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "chunk_method": chunk["chunk_method"],
            "chunk_size": chunk["chunk_size"],
            "chunk_overlap": chunk["chunk_overlap"],
            "start_token": chunk["start_token"],
            "end_token": chunk["end_token"],
            "token_count": chunk["token_count"]
        }
    )


with open(
    FIXED_METADATA,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n===================================")
print("Embedding generation complete")
print("===================================")

print(
    "Embeddings:",
    FIXED_EMBEDDINGS
)

print(
    "Metadata:",
    FIXED_METADATA
)