import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer


# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = "data/processed/fixed_size/fixed_256_overlap32.json"

OUTPUT_EMBEDDINGS = (
    "data/processed/embeddings/fixed_256_embeddings.npy"
)

OUTPUT_METADATA = (
    "data/processed/embeddings/fixed_256_metadata.json"
)


# ==========================================================
# Load embedding model
# ==========================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# Load chunks
# ==========================================================

print("Loading chunks...")

with open(INPUT_FILE, "r") as f:
    chunks = json.load(f)


texts = [
    chunk["text"]
    for chunk in chunks
]


print(f"Loaded {len(texts)} chunks.")


# ==========================================================
# Generate embeddings
# ==========================================================

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)


print("Embedding shape:", embeddings.shape)


# ==========================================================
# Save embeddings
# ==========================================================

os.makedirs(
    "data/processed/embeddings",
    exist_ok=True
)

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)


# ==========================================================
# Save metadata
# ==========================================================

metadata = []

for chunk in chunks:

    metadata.append({

        "article_id": chunk["article_id"],
        "title": chunk["title"],
        "source": chunk["source"],
        "chunk_id": chunk["chunk_id"],
        "chunk_method": chunk["chunk_method"],
        "chunk_size": chunk["chunk_size"],
        "chunk_overlap": chunk["chunk_overlap"],
        "start_token": chunk["start_token"],
        "end_token": chunk["end_token"],
        "token_count": chunk["token_count"]

    })


with open(
    OUTPUT_METADATA,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


print()
print("===================================")
print("Embedding generation complete.")
print("Embeddings saved to:")
print(OUTPUT_EMBEDDINGS)
print()
print("Metadata saved to:")
print(OUTPUT_METADATA)
print("===================================")