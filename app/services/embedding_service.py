# app/services/embedding_service.py
# ---------------------------------
# Service layer for sentence embedding generation using LangChain Google GenAI embeddings.

import logging
from typing import List
from app.config import settings
from app.services.langchain import get_embeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding generation service using LangChain's Google GenAI embeddings.
    Provides standard helper functions to embed text and documents.
    """

    def __init__(self) -> None:
        """
        Initialize the EmbeddingService.
        """
        logger.info("Initializing LangChain embedding service wrapper...")
        try:
            self.model = get_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize LangChain embeddings model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize embedding model: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        Returns a list of floats (embedding vector) of EMBEDDING_DIMENSION size.
        Returns a zero-vector if the text is empty or whitespace-only.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to embed_text. Returning zero-vector.")
            return [0.0] * settings.EMBEDDING_DIMENSION

        logger.info("Generating embedding for 1 chunk...")
        try:
            embedding = self.model.embed_query(text)
            logger.info("Embedding generation completed.")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {e}") from e

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
            logger.info(f"Generating batch embeddings for {len(texts_to_embed)} chunks...")
            try:
                list_embeddings = self.model.embed_documents(texts_to_embed)
                logger.info("Embedding generation completed.")
                
                for embed, original_idx in zip(list_embeddings, indices_to_embed):
                    embeddings[original_idx] = embed
            except Exception as e:
                logger.error(f"Error generating embeddings for document chunks: {e}", exc_info=True)
                raise RuntimeError(f"Bulk embedding generation failed: {e}") from e

        final_embeddings = [
            emb if emb is not None else [0.0] * settings.EMBEDDING_DIMENSION
            for emb in embeddings
        ]

        return final_embeddings
