# ==========================================================
# RAG Experiment Configuration
# ==========================================================


# ----------------------------
# Dataset
# ----------------------------

RAW_DATA = (
    "data/raw/wikipedia_articles.jsonl"
)


PROCESSED_DATA = (
    "data/processed/wikipedia_processed.jsonl"
)



# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE = 256

CHUNK_OVERLAP = 32


CHUNK_OUTPUTS = {

    "fixed":
        "data/processed/fixed_size/fixed_256_overlap32.json",

    "sentence":
        "data/processed/sentence_based/sentence_chunks.json",

    "semantic":
        "data/processed/semantic_based/semantic_chunks.json",

    "topic":
        "data/processed/topic_based/topic_chunks.json"

}



# ----------------------------
# Embedding Model
# ----------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


EMBEDDING_DIR = (
    "data/processed/embeddings"
)



EMBEDDING_OUTPUTS = {

    "fixed":
        "data/processed/embeddings/fixed_embeddings.npy",

    "sentence":
        "data/processed/embeddings/sentence_embeddings.npy",

    "semantic":
        "data/processed/embeddings/semantic_embeddings.npy",

    "topic":
        "data/processed/embeddings/topic_embeddings.npy"

}



METADATA_OUTPUTS = {

    "fixed":
        "data/processed/embeddings/fixed_metadata.json",

    "sentence":
        "data/processed/embeddings/sentence_metadata.json",

    "semantic":
        "data/processed/embeddings/semantic_metadata.json",

    "topic":
        "data/processed/embeddings/topic_metadata.json"

}



# ----------------------------
# FAISS
# ----------------------------

INDEX_OUTPUTS = {

    "fixed":
        "models/fixed_faiss.index",

    "sentence":
        "models/sentence_faiss.index",

    "semantic":
        "models/semantic_faiss.index",

    "topic":
        "models/topic_faiss.index"

}



# ----------------------------
# Retrieval
# ----------------------------

TOP_K = 5



# ----------------------------
# LLM
# ----------------------------

OLLAMA_MODEL = (
    "rag-mistral"
)