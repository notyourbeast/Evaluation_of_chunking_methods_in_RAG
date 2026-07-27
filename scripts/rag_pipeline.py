import argparse
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
    CHUNK_OUTPUTS,
    INDEX_OUTPUTS,
    METADATA_OUTPUTS,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    TOP_K
)


# ==========================================================
# Arguments
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--strategy",
    required=True,
    choices=[
        "fixed",
        "sentence",
        "semantic",
        "topic"
    ]
)


args = parser.parse_args()

strategy = args.strategy



# ==========================================================
# Select experiment files
# ==========================================================

INDEX_FILE = INDEX_OUTPUTS[strategy]

METADATA_FILE = METADATA_OUTPUTS[strategy]

CHUNK_FILE = CHUNK_OUTPUTS[strategy]


print()
print("=" * 70)
print("RAG Experiment")
print("=" * 70)

print(
    "Strategy:",
    strategy
)

print(
    "Index:",
    INDEX_FILE
)

print(
    "Chunks:",
    CHUNK_FILE
)



# ==========================================================
# Load FAISS Index
# ==========================================================

print("\nLoading FAISS index...")


index = faiss.read_index(
    INDEX_FILE
)


print(
    f"FAISS vectors: {index.ntotal}"
)



# ==========================================================
# Load Metadata
# ==========================================================

print("\nLoading metadata...")


with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(
    f"Metadata records: {len(metadata)}"
)



# ==========================================================
# Validate Index
# ==========================================================

if index.ntotal != len(metadata):

    raise ValueError(
        f"FAISS vectors ({index.ntotal}) "
        f"do not match metadata ({len(metadata)})"
    )



# ==========================================================
# Load Chunks
# ==========================================================

print("\nLoading chunks...")


with open(
    CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


print(
    f"Chunk records: {len(chunks)}"
)



if len(chunks) != len(metadata):

    raise ValueError(
        f"Chunks ({len(chunks)}) "
        f"do not match metadata ({len(metadata)})"
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
# Query
# ==========================================================

query = input(
    "\nEnter your question: "
).strip()


if not query:

    raise ValueError(
        "Question cannot be empty"
    )



# ==========================================================
# Query Embedding
# ==========================================================

query_embedding = embedding_model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
)



# ==========================================================
# Retrieval
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



for rank, idx in enumerate(
    indices[0],
    start=1
):

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
        f"Chunk ID  : {chunk['chunk_id']}"
    )

    print(
        f"Method    : {chunk['chunk_method']}"
    )

    print(
        f"Tokens    : {chunk['token_count']}"
    )

    print("-" * 70)

    print(
        chunk["text"][:500]
    )

    print("...")



# ==========================================================
# Context
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

Do not use outside knowledge.
Do not invent facts.

Context:

{context}


Question:

{query}


Answer:
"""



# ==========================================================
# Generate Answer
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