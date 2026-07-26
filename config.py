# ==========================================================
# RAG Experiment Configuration
# ==========================================================


# ----------------------------
# Dataset
# ----------------------------

RAW_DATA = "data/raw/wikipedia_articles.jsonl"

PROCESSED_DATA = (
    "data/processed/wikipedia_processed.jsonl"
)


# ----------------------------
# Chunking
# ----------------------------

CHUNK_SIZE = 256

CHUNK_OVERLAP = 32


FIXED_CHUNK_FILE = (
    "data/processed/fixed_size/fixed_256_overlap32.json"
)


# ----------------------------
# Embedding Model
# ----------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


EMBEDDING_DIR = (
    "data/processed/embeddings"
)


FIXED_EMBEDDINGS = (
    "data/processed/embeddings/fixed_256_embeddings.npy"
)


FIXED_METADATA = (
    "data/processed/embeddings/fixed_256_metadata.json"
)


# ----------------------------
# FAISS
# ----------------------------

FIXED_INDEX = (
    "models/fixed_256_faiss.index"
)


# ----------------------------
# Retrieval
# ----------------------------

TOP_K = 5


# ----------------------------
# LLM
# ----------------------------

OLLAMA_MODEL = "rag-mistral"
