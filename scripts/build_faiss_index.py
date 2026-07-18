import os
import numpy as np
import faiss


# ==========================================================
# Paths
# ==========================================================

INPUT_EMBEDDINGS = (
    "data/processed/embeddings/fixed_256_embeddings.npy"
)

OUTPUT_INDEX = (
    "models/fixed_256_faiss.index"
)


# ==========================================================
# Load embeddings
# ==========================================================

print("Loading embeddings...")

embeddings = np.load(INPUT_EMBEDDINGS)


print("Embedding shape:", embeddings.shape)


# ==========================================================
# Create FAISS index
# ==========================================================

print("Creating FAISS index...")


dimension = embeddings.shape[1]


index = faiss.IndexFlatL2(
    dimension
)


# ==========================================================
# Add vectors
# ==========================================================

print("Adding embeddings to index...")

index.add(
    embeddings
)


print(
    "Total vectors in index:",
    index.ntotal
)


# ==========================================================
# Save index
# ==========================================================

os.makedirs(
    "models",
    exist_ok=True
)


faiss.write_index(
    index,
    OUTPUT_INDEX
)


print()
print("==============================")
print("FAISS index created successfully")
print("Saved:", OUTPUT_INDEX)
print("==============================")
