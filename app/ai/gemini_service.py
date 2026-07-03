# app/ai/gemini_service.py
# ------------------------
# Infrastructure service layer for Gemini 2.5 Flash API.

import logging
import time
from typing import Any, Dict, Optional
from google import genai
from google.genai import types
from google.genai import errors
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Base exception for Gemini service errors."""
    pass


class GeminiConfigurationError(GeminiError):
    """Raised when there is a configuration issue (e.g. invalid or missing API key)."""
    pass


class GeminiQuotaExceededError(GeminiError):
    """Raised when rate limits or quotas are exceeded."""
    pass


class GeminiTimeoutError(GeminiError):
    """Raised when the request times out."""
    pass


class GeminiAPIError(GeminiError):
    """Raised when the API returns an error."""
    pass


class GeminiService:
    """
    Service class managing connection and request lifecycle with Gemini 2.5 Flash.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the GeminiService.
        Loads the API key from settings if not explicitly provided.
        """
        logger.info("Initializing GeminiService...")
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key or not self.api_key.strip():
            logger.error("Failed to initialize GeminiService: GEMINI_API_KEY is missing.")
            raise GeminiConfigurationError("GEMINI_API_KEY is not set. Please configure it in Settings.")

        try:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("GeminiService client initialized.")
        except Exception as e:
            logger.error(f"Error instantiating GenAI client: {e}")
            raise GeminiConfigurationError(f"GenAI Client instantiation failed: {e}")

    def generate_answer(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash",
        timeout: float = 30.0,
    ) -> str:
        """
        Send a prompt to the Gemini API and return the generated answer.

        Args:
            prompt: The formatted prompt to send.
            model_name: The Gemini model to use. Defaults to gemini-2.5-flash.
            timeout: Maximum time in seconds to wait for the API response.

        Returns:
            str: Generated text answer.
        """
        if not prompt or not prompt.strip():
            logger.error("Empty prompt passed to generate_answer.")
            raise ValueError("Prompt cannot be empty or whitespace-only.")

        logger.info(f"Sending request to Gemini API (model={model_name}, timeout={timeout}s, prompt_len={len(prompt)})")
        
        try:
            timeout_ms = int(timeout * 1000)
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=timeout_ms)
            )
        except Exception as e:
            logger.error(f"Failed to configure GenAI client for request: {e}")
            raise GeminiConfigurationError(f"Client setup failed: {e}")

        start_time = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            latency = time.perf_counter() - start_time
            logger.info(f"Gemini API request succeeded in {latency:.4f}s")
            
            answer = response.text
            if not answer:
                logger.warning("Gemini API returned an empty response text.")
                return ""
                
            return answer

        except errors.APIError as e:
            latency = time.perf_counter() - start_time
            code = getattr(e, "code", None)
            message = getattr(e, "message", str(e))
            logger.error(
                f"Gemini API failure after {latency:.4f}s (code={code}): {message}",
                exc_info=True
            )
            
            if code == 429:
                raise GeminiQuotaExceededError(f"Gemini API quota exceeded: {message}") from e
            elif code in (400, 403) and any(
                k in message.lower() for k in ("key", "api_key", "permission", "credential", "invalid", "unauthorized")
            ):
                raise GeminiConfigurationError(f"Gemini API key is invalid or unauthorized: {message}") from e
            else:
                raise GeminiAPIError(f"Gemini API error (code={code}): {message}") from e

        except Exception as e:
            latency = time.perf_counter() - start_time
            err_name = type(e).__name__
            logger.error(
                f"Unexpected exception during Gemini API call after {latency:.4f}s: {err_name} - {e}",
                exc_info=True
            )
            
            if "timeout" in err_name.lower() or "timeout" in str(e).lower():
                raise GeminiTimeoutError(f"Gemini API request timed out: {e}") from e
                
            raise GeminiError(f"Failed to generate answer due to an unexpected error: {e}") from e
