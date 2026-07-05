# app/services/langchain/embeddings.py
# ------------------------------------
# LangChain integration wrapper for Gemini embedding models.

import logging
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_cached_embeddings(model_name: str) -> GoogleGenerativeAIEmbeddings:
    """
    Private helper function to cache GoogleGenerativeAIEmbeddings instances.
    Uses lru_cache to ensure we reuse the same instance for identical arguments.
    """
    logger.info(f"Initializing new GoogleGenerativeAIEmbeddings instance (model={model_name})")
    
    api_key = settings.GEMINI_API_KEY
    if not api_key or not api_key.strip():
        logger.error("Initialization failed: GEMINI_API_KEY is not set.")
        raise ValueError("GEMINI_API_KEY is not configured in settings/environment.")
        
    try:
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
            output_dimensionality=768,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate GoogleGenerativeAIEmbeddings: {e}", exc_info=True)
        raise RuntimeError(f"Could not load Gemini Embeddings model: {e}") from e


def get_embeddings(model_name: str = "models/gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    """
    Get a configured and cached GoogleGenerativeAIEmbeddings instance.

    Purpose:
        Provides a reusable, single-instance LangChain wrapper for Google's Gemini embeddings
        without recreating it repeatedly throughout the request lifecycles.

    Parameters:
        model_name (str): The name of the embedding model to use. Defaults to "models/gemini-embedding-001".

    Returns:
        GoogleGenerativeAIEmbeddings: The cached or newly initialized Gemini Embeddings wrapper.
    """
    embeddings = _get_cached_embeddings(model_name)
    from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
    return RetrievalAnalyticsService().wrap_embeddings(embeddings)
