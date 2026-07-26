import json
import sys
from pathlib import Path

import faiss
import ollama

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
    FIXED_CHUNK_FILE,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
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
    f"FAISS vectors: {index.ntotal}"
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
    f"Metadata records: {len(metadata)}"
)


# ==========================================================
# Validate Index Consistency
# ==========================================================

if index.ntotal != len(metadata):

    raise ValueError(
        f"FAISS index contains {index.ntotal} vectors "
        f"but metadata contains {len(metadata)} records."
    )


# ==========================================================
# Load Chunks
# ==========================================================

print("\nLoading chunks...")


with open(
    FIXED_CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"Chunk records: {len(chunks)}"
)


if len(chunks) != len(metadata):

    raise ValueError(
        f"Chunk file contains {len(chunks)} records "
        f"but metadata contains {len(metadata)}."
    )


# ==========================================================
# Load Embedding Model
# ==========================================================

print("\nLoading embedding model...")


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded"
)


# ==========================================================
# User Query
# ==========================================================

query = input(
    "\nEnter your question: "
).strip()


if not query:

    raise ValueError(
        "Question cannot be empty."
    )


# ==========================================================
# Generate Query Embedding
# ==========================================================

query_embedding = embedding_model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)


# ==========================================================
# Retrieve Top-K Chunks
# ==========================================================

scores, indices = index.search(
    query_embedding,
    TOP_K
)


# ==========================================================
# Display Retrieved Chunks
# ==========================================================

retrieved_chunks = []


print("\n")
print("=" * 70)
print("Retrieved Chunks")
print("=" * 70)


for rank, idx in enumerate(indices[0], start=1):

    chunk = chunks[idx]

    retrieved_chunks.append(
        chunk["text"]
    )


    print()

    print(
        f"Rank      : {rank}"
    )

    print(
        f"Score     : {scores[0][rank-1]:.4f}"
    )

    print(
        f"Title     : {chunk['title']}"
    )

    print(
        f"Document  : {chunk['doc_id']}"
    )

    print(
        f"Chunk ID  : {chunk['chunk_id']}"
    )

    print(
        f"Tokens    : {chunk['token_count']}"
    )

    print("-" * 70)

    preview = chunk["text"][:500].replace(
        "\n",
        " "
    )

    print(preview)

    if len(chunk["text"]) > 500:
        print("...")


# ==========================================================
# Build Context
# ==========================================================

context = "\n\n".join(
    retrieved_chunks
)


print("\n")
print("=" * 70)
print("Context Ready")
print("=" * 70)

print(
    f"Chunks retrieved : {len(retrieved_chunks)}"
)

print(
    f"Context length   : {len(context):,} characters"
)


# ==========================================================
# Prompt
# ==========================================================

prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.

Answer the question using ONLY the retrieved context below.

If the answer cannot be determined from the context,
respond exactly:

"I do not have enough information in the retrieved context."

Do not invent facts.
Do not use outside knowledge.

Context:

{context}

Question:

{query}

Answer:
"""


# ==========================================================
# Generate Response
# ==========================================================

print(
    "\nGenerating answer using Ollama...\n"
)


response = ollama.chat(
    model=OLLAMA_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


answer = response["message"]["content"]


# ==========================================================
# Output
# ==========================================================

print("\n")
print("=" * 70)
print("Generated Answer")
print("=" * 70)

print(answer)