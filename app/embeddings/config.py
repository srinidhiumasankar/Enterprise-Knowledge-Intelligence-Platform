# app/embeddings/config.py
# ------------------------
# Dedicated ChromaDB configuration module.

from app.config import settings

# Standardized configurations for the vector database and embeddings module
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
CHROMA_COLLECTION_NAME = settings.CHROMA_COLLECTION_NAME
TOP_K_RESULTS = settings.TOP_K_RESULTS
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL_NAME
DEFAULT_TOP_K = settings.DEFAULT_TOP_K
SIMILARITY_THRESHOLD = settings.SIMILARITY_THRESHOLD
