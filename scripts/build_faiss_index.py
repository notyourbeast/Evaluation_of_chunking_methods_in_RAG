import os
import sys
from pathlib import Path

import numpy as np
import faiss


# ==========================================================
# Add project root to Python path
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    FIXED_EMBEDDINGS,
    FIXED_INDEX
)

# ==========================================================
# Load embeddings
# ==========================================================

print("Loading embeddings...")

embeddings = np.load(
    FIXED_EMBEDDINGS
).astype("float32")


print(
    f"Embedding shape: {embeddings.shape}"
)


# ==========================================================
# Normalize embeddings
# ==========================================================

print("Normalizing embeddings...")

faiss.normalize_L2(
    embeddings
)


# ==========================================================
# Create FAISS Index
# ==========================================================

dimension = embeddings.shape[1]


print(
    "Creating IndexFlatIP..."
)


index = faiss.IndexFlatIP(
    dimension
)


# ==========================================================
# Add vectors
# ==========================================================

print(
    "Adding embeddings to index..."
)


index.add(
    embeddings
)


print(
    f"Total vectors: {index.ntotal}"
)


# ==========================================================
# Save Index
# ==========================================================

index_directory = os.path.dirname(
    FIXED_INDEX
)


os.makedirs(
    index_directory,
    exist_ok=True
)


faiss.write_index(
    index,
    FIXED_INDEX
)


print()

print("=" * 40)
print("FAISS index created successfully")
print("=" * 40)

print(
    f"Saved to: {FIXED_INDEX}"
)