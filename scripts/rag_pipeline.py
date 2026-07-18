import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ==========================================================
# Paths
# ==========================================================

INDEX_FILE = "models/fixed_256_faiss.index"

METADATA_FILE = (
    "data/processed/embeddings/fixed_256_metadata.json"
)

CHUNK_FILE = (
    "data/processed/fixed_size/fixed_256_overlap32.json"
)

MODEL_PATH = (
    "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
)

# ==========================================================
# Load FAISS
# ==========================================================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)

print("Vectors:", index.ntotal)

# ==========================================================
# Load metadata
# ==========================================================

print("Loading metadata...")

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

print("Metadata records:", len(metadata))

# ==========================================================
# Load chunk data
# ==========================================================

print("Loading chunk data...")

with open(CHUNK_FILE, "r") as f:
    chunks = json.load(f)

print("Chunks:", len(chunks))

# ==========================================================
# Load embedding model
# ==========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# User Query
# ==========================================================

query = input("\nEnter your question: ")


# ==========================================================
# Create Query Embedding
# ==========================================================

query_embedding = embedding_model.encode(
    [query],
    convert_to_numpy=True
)


# ==========================================================
# Retrieve Top-k Chunks
# ==========================================================

TOP_K = 5

distances, indices = index.search(
    query_embedding,
    TOP_K
)

# ==========================================================
# Build Context
# ==========================================================

retrieved_chunks = []

print("\n")
print("=" * 70)
print("Retrieved Chunks")
print("=" * 70)

for rank, idx in enumerate(indices[0], start=1):

    chunk = chunks[idx]

    retrieved_chunks.append(chunk["text"])

    print(f"\nRank {rank}")
    print(f"Distance : {distances[0][rank-1]:.4f}")
    print(f"Title    : {chunk['title']}")
    print(f"Chunk ID : {chunk['chunk_id']}")
    print("-" * 70)
    print(chunk["text"][:500])
    print("...")
    
    # ==========================================================
# Combine Retrieved Context
# ==========================================================

context = "\n\n".join(retrieved_chunks)

print("\n")
print("=" * 70)
print("Context Ready")
print("=" * 70)

print(f"Retrieved {len(retrieved_chunks)} chunks.")
print(f"Total context length: {len(context)} characters.")