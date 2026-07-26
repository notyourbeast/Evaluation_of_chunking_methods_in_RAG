import json
import spacy
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer


print("=" * 50)
print("Week 1 RAG System Validation")
print("=" * 50)


# ----------------------------------------------------------
# Check spaCy
# ----------------------------------------------------------

print("\nChecking spaCy...")

nlp = spacy.load(
    "en_core_web_sm"
)

doc = nlp(
    "RAG systems improve retrieval quality."
)

print("spaCy OK")


# ----------------------------------------------------------
# Check dataset
# ----------------------------------------------------------

print("\nChecking dataset...")

with open(
    "data/raw/wiki_articles.json"
) as f:

    data = json.load(f)


print(
    "Articles loaded:",
    len(data)
)


# ----------------------------------------------------------
# Check embedding model
# ----------------------------------------------------------

print("\nChecking embedding model...")


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


embedding = embedding_model.encode(
    [
        "Artificial intelligence"
    ]
)


print(
    "Embedding shape:",
    embedding.shape
)


# ----------------------------------------------------------
# Check FAISS
# ----------------------------------------------------------

print("\nChecking FAISS...")


index = faiss.read_index(
    "models/fixed_256_faiss.index"
)


print(
    "FAISS vectors:",
    index.ntotal
)


# ----------------------------------------------------------
# Check Ollama
# ----------------------------------------------------------

print("\nChecking Ollama...")


response = ollama.chat(
    model="rag-mistral",
    messages=[
        {
            "role": "user",
            "content":
            "Explain artificial intelligence in one sentence."
        }
    ]
)


print(
    "LLM response:"
)

print(
    response["message"]["content"]
)


print("\n================================")
print("ALL SYSTEMS WORKING")
print("================================")