# app/services/langchain/llm.py
# -----------------------------
# LangChain integration wrapper for Gemini LLM models.

import logging
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_cached_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    """
    Private helper function to cache ChatGoogleGenerativeAI instances.
    Uses lru_cache to ensure we reuse the same instance for identical arguments.
    """
    logger.info(f"Initializing new ChatGoogleGenerativeAI instance (model={model_name}, temperature={temperature})")
    
    api_key = settings.GEMINI_API_KEY
    if not api_key or not api_key.strip():
        logger.error("Initialization failed: GEMINI_API_KEY is not set.")
        raise ValueError("GEMINI_API_KEY is not configured in settings/environment.")
        
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate ChatGoogleGenerativeAI: {e}", exc_info=True)
        raise RuntimeError(f"Could not load Gemini Chat model: {e}") from e


def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """
    Get a configured and cached ChatGoogleGenerativeAI instance.

    Purpose:
        Provides a reusable, single-instance LangChain wrapper for Google's Gemini LLM
        without recreating it repeatedly throughout the request lifecycles.

    Parameters:
        model_name (str): The name of the Gemini model to use. Defaults to "gemini-2.5-flash".
        temperature (float): Controls response randomness (higher = more creative). Defaults to 0.7.

    Returns:
        ChatGoogleGenerativeAI: The cached or newly initialized Gemini Chat model wrapper.
    """
    return _get_cached_llm(model_name, temperature)
