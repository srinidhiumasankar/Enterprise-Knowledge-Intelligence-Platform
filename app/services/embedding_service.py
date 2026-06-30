# app/services/embedding_service.py
# ---------------------------------
# Service layer for sentence embedding generation using Hugging Face SentenceTransformers.

import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding generation service using Hugging Face sentence-transformers.
    Follows a lazy-loading singleton pattern for the model to minimize memory overhead.
    """

    _model: Union[SentenceTransformer, None] = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Retrieve the SentenceTransformer model instance (lazy loads on first call).
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

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        Returns a list of floats (embedding vector) of EMBEDDING_DIMENSION size.
        Returns a zero-vector if the text is empty or whitespace-only.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to embed_text. Returning zero-vector.")
            return [0.0] * settings.EMBEDDING_DIMENSION

        try:
            model = self.get_model()
            embedding = model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    def embed_documents(self, list_of_chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        Encodes in bulk for optimal performance.
        Returns a list of float vectors, maintaining 1:1 mapping with the input list.
        """
        if not list_of_chunks:
            return []

        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for idx, chunk in enumerate(list_of_chunks):
            if not chunk or not chunk.strip():
                logger.warning(f"Empty chunk found at index {idx} in embed_documents. Using zero-vector.")
                embeddings.append(None)
            else:
                embeddings.append(None)
                texts_to_embed.append(chunk)
                indices_to_embed.append(idx)

        if texts_to_embed:
            try:
                model = self.get_model()
                encoded_embeddings = model.encode(texts_to_embed)
                list_embeddings = encoded_embeddings.tolist()
                
                for embed, original_idx in zip(list_embeddings, indices_to_embed):
                    embeddings[original_idx] = embed
            except Exception as e:
                logger.error(f"Error generating embeddings for document chunks: {e}")
                raise RuntimeError(f"Bulk embedding generation failed: {e}")

        final_embeddings = [
            emb if emb is not None else [0.0] * settings.EMBEDDING_DIMENSION
            for emb in embeddings
        ]

        return final_embeddings
