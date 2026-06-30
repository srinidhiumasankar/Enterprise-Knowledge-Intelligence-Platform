# app/embeddings/embedding_service.py
# ------------------------------------
# Infrastructure layer for vector embedding generation.

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service class for initializing the sentence-transformers embedding model
    and generating text embeddings. Follows the singleton pattern for model loading.
    """

    _model: Optional[SentenceTransformer] = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Retrieve the loaded SentenceTransformer model instance.
        Lazy-loads the model weights on first call.
        """
        if cls._model is None:
            logger.info(f"Loading SentenceTransformer model: '{settings.EMBEDDING_MODEL_NAME}'")
            try:
                cls._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                logger.info("SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{settings.EMBEDDING_MODEL_NAME}': {e}")
                raise RuntimeError(f"Could not initialize embedding model: {e}")
        return cls._model

    def load_model(self) -> None:
        """
        Explicitly load the embedding model weights if not already loaded.
        """
        self.get_model()

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

        try:
            model = self.get_model()
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of text segments in batch mode.

        Args:
            texts: A list of text inputs to embed.

        Returns:
            List[List[float]]: A list of generated embedding vectors.
        """
        if not texts:
            return []

        # Validate inputs
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.error(f"Empty text found at index {i} in generate_batch_embeddings.")
                raise ValueError(f"Text at index {i} cannot be empty or whitespace-only.")

        try:
            model = self.get_model()
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")

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
            model_loaded = self._model is not None
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
