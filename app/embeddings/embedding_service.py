# app/embeddings/embedding_service.py
# ------------------------------------
# Infrastructure layer for vector embedding generation using LangChain Google GenAI embeddings.

import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.langchain import get_embeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service class for initializing the LangChain Google GenAI embedding model
    and generating text embeddings. Follows the singleton pattern for model loading.
    """

    _model: Optional[Any] = None

    def __init__(self) -> None:
        """
        Initialize the EmbeddingService instance.
        """
        self.initialize()

    def initialize(self) -> None:
        """
        Setup the LangChain Google GenAI embeddings using configuration settings.
        Lazy-loads the embedding wrapper on first call or checks cached instance.
        """
        if EmbeddingService._model is None:
            logger.info("Initializing LangChain embeddings service...")
            try:
                # Reuse cached embedding instance from Phase 8.1
                EmbeddingService._model = get_embeddings()
                logger.info("LangChain embeddings service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize LangChain Google GenAI embeddings: {e}", exc_info=True)
                raise RuntimeError(f"Embedding service initialization failed: {e}") from e
        
        self.model = EmbeddingService._model

    def load_model(self) -> None:
        """
        Explicitly load the embedding model if not already loaded.
        """
        self.initialize()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a dense vector embedding for a single text input.

        Args:
            text: The input text to embed.

        Returns:
            List[float]: The generated embedding vector.
        """
        if not text or not text.strip():
            logger.error("Empty or whitespace-only text passed to generate_embedding.")
            raise ValueError("Text input cannot be empty or whitespace-only.")

        self.initialize()
        logger.info("Generating embedding for 1 chunk...")
        try:
            embedding = self.model.embed_query(text)
            if not embedding:
                raise RuntimeError("Underlying embedding provider returned empty result.")
            logger.info("Embedding generation completed.")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of text segments in batch mode.

        Args:
            texts: A list of text inputs to embed.

        Returns:
            List[List[float]]: A list of generated embedding vectors.
        """
        if not texts:
            logger.warning("Empty text list passed to generate_batch_embeddings. Returning empty list.")
            return []

        # Validate inputs
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.error(f"Empty or whitespace-only text found at index {i} in generate_batch_embeddings.")
                raise ValueError(f"Text at index {i} cannot be empty or whitespace-only.")

        self.initialize()
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        try:
            embeddings = self.model.embed_documents(texts)
            if not embeddings or len(embeddings) != len(texts):
                raise RuntimeError("Underlying embedding provider returned mismatched or empty results.")
            logger.info("Embedding generation completed.")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate vector embedding for a search query.

        Args:
            query: The search query string.

        Returns:
            List[float]: The generated query embedding vector.
        """
        if not query or not query.strip():
            logger.error("Empty or whitespace-only query passed to generate_query_embedding.")
            raise ValueError("Query input cannot be empty or whitespace-only.")

        return self.generate_embedding(query)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the embedding service.

        Returns:
            Dict[str, Any]: Health status dictionary.
        """
        try:
            self.initialize()
            model_loaded = self.model is not None
            status_str = "healthy" if model_loaded else "uninitialized"
            return {
                "status": status_str,
                "model_name": settings.EMBEDDING_MODEL_NAME,
                "model_loaded": model_loaded,
            }
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
