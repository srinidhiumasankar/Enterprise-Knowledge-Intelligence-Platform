# tests/verify_phase_7_1.py
# -------------------------
# Automated verification script for Phase 7.1: LLM Integration (Gemini 2.5 Flash API).

import logging
import os
import sys
import subprocess
from unittest.mock import MagicMock, patch

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_7_1")


def test_missing_api_key_startup_validation():
    """
    Verify that settings.py fails startup validation if GEMINI_API_KEY is missing.
    We run this in a subprocess with a cleared environment to verify.
    """
    logger.info("Test Case 1: Verifying startup validation for missing GEMINI_API_KEY...")
    
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = ""
    
    command = [
        sys.executable,
        "-c",
        "from app.config import settings"
    ]
    
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0, "Expected import to fail when GEMINI_API_KEY is missing, but it succeeded!"
    assert "Configuration Error" in result.stderr or "ValueError" in result.stderr or "ValidationError" in result.stderr, (
        f"Expected a configuration/validation error message in stderr, got:\n{result.stderr}"
    )
    logger.info("✓ Missing API key startup validation succeeded (application refused to start).")


def test_api_key_loading():
    """
    Verify settings.py loads the API key correctly when configured.
    """
    logger.info("Test Case 2: Verifying API key loading when present...")
    os.environ["GEMINI_API_KEY"] = "mock-api-key-value-12345"
    
    import sys
    if "app.config.settings" in sys.modules:
        del sys.modules["app.config.settings"]
    if "app.config" in sys.modules:
        del sys.modules["app.config"]
    
    from app.config import settings
    assert settings.GEMINI_API_KEY == "mock-api-key-value-12345", "GEMINI_API_KEY was not loaded correctly!"
    logger.info("✓ API key loaded successfully.")


def test_prompt_builder_basic():
    """
    Verify that PromptBuilder builds prompt correctly under normal conditions.
    """
    logger.info("Test Case 3: Verifying PromptBuilder formatting and constraints...")
    from app.ai.prompt_builder import PromptBuilder

    question = "What is the vector dimension used for sentence embeddings?"
    chunks = [
        {
            "text": "The vector dimension for BAAI/bge-base-en-v1.5 is 768.",
            "document_id": 12,
            "chunk_id": "chunk-100",
            "metadata": {
                "filename": "embedding_spec.pdf",
                "owner_id": 1
            }
        },
        {
            "text": "ChromaDB stores these 768-dimensional embeddings for retrieval.",
            "document_id": 12,
            "chunk_id": "chunk-101",
            "metadata": {
                "filename": "embedding_spec.pdf",
                "owner_id": 1
            }
        }
    ]

    prompt = PromptBuilder.build_prompt(question, chunks)
    
    assert question in prompt, "Question not found in the generated prompt!"
    assert "The vector dimension for BAAI/bge-base-en-v1.5 is 768." in prompt, "Chunk text not found in prompt!"
    assert "embedding_spec.pdf" in prompt, "Metadata filename not found in prompt!"
    assert "Document ID: 12" in prompt, "Document ID not found in prompt!"
    assert "Chunk ID: chunk-100" in prompt, "Chunk ID not found in prompt!"
    assert "I cannot determine the answer from the uploaded documents." in prompt, "Hallucination constraints missing!"
    assert "Source 1" in prompt, "Citation instructions missing!"
    logger.info("✓ PromptBuilder formatted prompts and constraints correctly.")


def test_prompt_builder_empty_context():
    """
    Verify PromptBuilder handles empty or None context list gracefully.
    """
    logger.info("Test Case 4: Verifying PromptBuilder with empty context...")
    from app.ai.prompt_builder import PromptBuilder

    question = "How many documents are uploaded?"
    
    prompt_empty = PromptBuilder.build_prompt(question, [])
    assert "No relevant context available." in prompt_empty, "Empty context was not handled correctly!"
    assert question in prompt_empty, "Question missing when context is empty!"

    prompt_none = PromptBuilder.build_prompt(question, None)
    assert "No relevant context available." in prompt_none, "None context was not handled correctly!"
    
    logger.info("✓ Empty context handled correctly by PromptBuilder.")


@patch("google.genai.Client")
def test_gemini_service_initialization(mock_client_class):
    """
    Verify GeminiService initializes correctly.
    """
    logger.info("Test Case 5: Verifying GeminiService initialization...")
    from app.ai.gemini_service import GeminiService
    
    service = GeminiService()
    assert service.api_key == "mock-api-key-value-12345", "API key not set correctly on GeminiService!"
    mock_client_class.assert_called_once_with(api_key="mock-api-key-value-12345")
    logger.info("✓ GeminiService initializes correctly.")


@patch("google.genai.Client")
def test_gemini_service_api_success(mock_client_class):
    """
    Verify successful API responses.
    """
    logger.info("Test Case 6: Verifying GeminiService API success response...")
    from app.ai.gemini_service import GeminiService

    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.text = "Based on BAAI/bge-base-en-v1.5, the dimension is 768. [Source 1]"
    mock_client_instance.models.generate_content.return_value = mock_response

    service = GeminiService()
    answer = service.generate_answer("What is the dimension?")
    
    assert answer == "Based on BAAI/bge-base-en-v1.5, the dimension is 768. [Source 1]"
    mock_client_instance.models.generate_content.assert_called_once()
    logger.info("✓ Successful generation handled correctly.")


@patch("google.genai.Client")
def test_gemini_service_failures(mock_client_class):
    """
    Verify GeminiService correctly catches, maps, and logs API failures.
    """
    logger.info("Test Case 7: Verifying GeminiService exception mapping for failures...")
    from app.ai.gemini_service import (
        GeminiService,
        GeminiQuotaExceededError,
        GeminiConfigurationError,
        GeminiAPIError,
        GeminiTimeoutError
    )
    from google.genai.errors import APIError

    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    service = GeminiService()

    # 1. Quota Exceeded (429)
    logger.info("  Sub-test 7.1: Testing HTTP 429 Quota Exceeded...")
    mock_client_instance.models.generate_content.side_effect = APIError(
        code=429,
        response_json={"message": "Quota exceeded for model gemini-2.5-flash"}
    )
    try:
        service.generate_answer("test")
        assert False, "Expected GeminiQuotaExceededError not raised!"
    except GeminiQuotaExceededError as e:
        logger.info(f"    Caught expected: {e}")

    # 2. Invalid Key (403)
    logger.info("  Sub-test 7.2: Testing HTTP 403/400 Invalid Key...")
    mock_client_instance.models.generate_content.side_effect = APIError(
        code=403,
        response_json={"message": "API key not valid. Please pass a valid API key."}
    )
    try:
        service.generate_answer("test")
        assert False, "Expected GeminiConfigurationError not raised!"
    except GeminiConfigurationError as e:
        logger.info(f"    Caught expected: {e}")

    # 3. Timeout error
    logger.info("  Sub-test 7.3: Testing Timeout exception...")
    class MockTimeoutException(Exception):
        pass
    mock_client_instance.models.generate_content.side_effect = MockTimeoutException("Request timeout occurred")
    try:
        service.generate_answer("test")
        assert False, "Expected GeminiTimeoutError not raised!"
    except GeminiTimeoutError as e:
        logger.info(f"    Caught expected: {e}")

    # 4. General API error (500)
    logger.info("  Sub-test 7.4: Testing General API Error (HTTP 500)...")
    mock_client_instance.models.generate_content.side_effect = APIError(
        code=500,
        response_json={"message": "Internal server error"}
    )
    try:
        service.generate_answer("test")
        assert False, "Expected GeminiAPIError not raised!"
    except GeminiAPIError as e:
        logger.info(f"    Caught expected: {e}")

    logger.info("✓ Gemini service correctly mapped all failures to custom exceptions.")


def run_all_tests():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 7.1 GEMINI INTEGRATION VERIFICATION")
    logger.info("==========================================================")

    test_missing_api_key_startup_validation()
    test_api_key_loading()
    test_prompt_builder_basic()
    test_prompt_builder_empty_context()
    test_gemini_service_initialization()
    test_gemini_service_api_success()
    test_gemini_service_failures()

    logger.info("\n==========================================================")
    logger.info("ALL PHASE 7.1 GEMINI INTEGRATION VERIFICATION TESTS PASSED!")
    logger.info("==========================================================")


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as ae:
        logger.error(f"Assertion failed: {ae}")
        sys.exit(1)
    except Exception as ex:
        logger.error(f"Test run encountered unexpected error: {ex}", exc_info=True)
        sys.exit(1)
