# tests/verify_phase_7_2.py
# -------------------------
# Automated verification script for Phase 7.2: RAG Answer Generation.

import logging
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

# Create mock instances for endpoint dependency overrides
mock_retrieval_service = AsyncMock()
mock_gemini_service = MagicMock()
from fastapi.testclient import TestClient

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set mock API key before importing settings to pass startup validation
os.environ["GEMINI_API_KEY"] = "mock-api-key-for-phase-7-2"

from app.main import app
from app.config import settings
from app.ai.prompt_builder import PromptBuilder
from app.ai.gemini_service import (
    GeminiService,
    GeminiQuotaExceededError,
    GeminiConfigurationError,
    GeminiTimeoutError,
    GeminiAPIError
)
from app.api.deps import get_retrieval_service, get_gemini_service

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_7_2")

# Override FastAPI dependencies
app.dependency_overrides[get_retrieval_service] = lambda: mock_retrieval_service
app.dependency_overrides[get_gemini_service] = lambda: mock_gemini_service


def test_api_key_loading():
    """
    Verify Gemini API key loads correctly from configuration.
    """
    logger.info("Test Case 1: Verifying API key loading...")
    assert settings.GEMINI_API_KEY == "mock-api-key-for-phase-7-2", "API key not loaded correctly!"
    logger.info("✓ API key loaded successfully.")


def test_prompt_builder():
    """
    Verify PromptBuilder builds prompt successfully.
    """
    logger.info("Test Case 2: Verifying PromptBuilder prompt construction...")
    question = "Who are the founders of Google?"
    chunks = [
        {
            "text": "Google was founded in September 1998 by Larry Page and Sergey Brin.",
            "document_id": 5,
            "chunk_id": "chunk-google-1",
            "metadata": {"filename": "google_history.txt", "owner_id": 10}
        }
    ]
    prompt = PromptBuilder.build_prompt(question, chunks)
    assert question in prompt
    assert "Larry Page and Sergey Brin" in prompt
    assert "google_history.txt" in prompt
    assert "I cannot determine the answer from the uploaded documents." in prompt
    logger.info("✓ PromptBuilder constructs prompt successfully.")


@patch("google.genai.Client")
def test_gemini_service_generation(mock_client_class):
    """
    Verify GeminiService generates an answer.
    """
    logger.info("Test Case 3: Verifying GeminiService answer generation...")
    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.text = "Google was founded by Larry Page and Sergey Brin. [Source 1]"
    mock_client_instance.models.generate_content.return_value = mock_response

    service = GeminiService()
    answer = service.generate_answer("Who are the founders of Google?")
    assert "Larry Page and Sergey Brin" in answer
    logger.info("✓ GeminiService generates answer successfully.")


def test_search_endpoint_with_results():
    """
    Verify search endpoint returns both answer and sources when results exist.
    """
    logger.info("Test Case 4: Verifying Search endpoint with results (RAG integration)...")
    
    # 1. Setup mock return values
    mock_retrieval_service.retrieve.return_value = [
        {
            "document_id": 101,
            "chunk_id": 12,
            "score": 0.92,
            "text": "Deep learning is a subset of machine learning.",
            "metadata": {
                "document_id": 101,
                "chunk_id": "c-12",
                "owner_id": 1,
                "filename": "deep_learning.txt"
            }
        },
        {
            "document_id": 101,
            "chunk_id": 12,  # Duplicate chunk_id
            "score": 0.88,
            "text": "Deep learning is a subset of machine learning.",
            "metadata": {
                "document_id": 101,
                "chunk_id": "c-12",
                "owner_id": 1,
                "filename": "deep_learning.txt"
            }
        },
        {
            "document_id": 102,
            "chunk_id": 13,
            "score": 0.85,
            "text": "Artificial neural networks are inspired by biological networks.",
            "metadata": {
                "document_id": 102,
                "chunk_id": "c-13",
                "owner_id": 1,
                "filename": "neural_networks.txt"
            }
        }
    ]
    mock_gemini_service.generate_answer.return_value = "Deep learning is a subset of machine learning. It uses artificial neural networks."

    # 2. Mock authentication and active user dependencies
    client = TestClient(app)
    
    # Register and login test user
    suffix = uuid.uuid4().hex[:6]
    email = f"user_{suffix}@ekip-rag.com"
    password = "Password123!"
    client.post("/api/auth/register", json={"email": email, "password": password})
    login_res = client.post("/api/auth/login", json={"email": email, "password": password})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Call endpoint
    response = client.post(
        "/api/search",
        json={"query": "What is deep learning?", "top_k": 3},
        headers=headers
    )
    
    assert response.status_code == 200, f"Search failed: {response.text}"
    resp_data = response.json()
    
    # Assert correct response fields
    assert resp_data["query"] == "What is deep learning?"
    assert resp_data["answer"] == "Deep learning is a subset of machine learning. It uses artificial neural networks."
    assert len(resp_data["results"]) == 3
    assert resp_data["message"] is None
    
    # Assert citations fields, deduplication, and score ordering
    assert "citations" in resp_data, "citations field is missing from response!"
    citations = resp_data["citations"]
    # Should have 2 citations due to deduplication of chunk_id 12
    assert len(citations) == 2, f"Expected 2 deduplicated citations, got {len(citations)}"
    
    # Verify values and ordering (highest score first)
    assert citations[0]["chunk_id"] == 12
    assert citations[0]["document_id"] == 101
    assert citations[0]["filename"] == "deep_learning.txt"
    assert citations[0]["score"] == 0.92

    assert citations[1]["chunk_id"] == 13
    assert citations[1]["document_id"] == 102
    assert citations[1]["filename"] == "neural_networks.txt"
    assert citations[1]["score"] == 0.85

    # Verify score ordering check
    assert citations[0]["score"] >= citations[1]["score"], "Citations are not sorted by score descending!"

    # 4. Verify empty query validation (HTTP 400 Bad Request)
    logger.info("  Sub-test: Verifying empty query validation returns HTTP 400...")
    res_empty = client.post("/api/search", json={"query": "", "top_k": 3}, headers=headers)
    assert res_empty.status_code == 400, f"Expected HTTP 400 for empty query, got {res_empty.status_code}"
    
    res_whitespace = client.post("/api/search", json={"query": "   ", "top_k": 3}, headers=headers)
    assert res_whitespace.status_code == 400, f"Expected HTTP 400 for whitespace-only query, got {res_whitespace.status_code}"

    # Verify gemini_service was called
    mock_gemini_service.generate_answer.assert_called_once()
    logger.info("✓ Search endpoint successfully returns clean answer and structured sorted citations, and validates empty inputs.")


def test_search_endpoint_empty_results():
    """
    Verify search endpoint skips Gemini call when empty results are retrieved.
    """
    logger.info("Test Case 5: Verifying Search endpoint with empty results (skips LLM)...")
    
    # 1. Setup mock to return empty
    mock_retrieval_service.retrieve.return_value = []
    mock_gemini_service.generate_answer.reset_mock()

    client = TestClient(app)
    
    # Register and login test user
    suffix = uuid.uuid4().hex[:6]
    email = f"user_{suffix}@ekip-rag.com"
    password = "Password123!"
    client.post("/api/auth/register", json={"email": email, "password": password})
    login_res = client.post("/api/auth/login", json={"email": email, "password": password})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Call endpoint
    response = client.post(
        "/api/search",
        json={"query": "What is quantum computing?", "top_k": 5},
        headers=headers
    )
    
    assert response.status_code == 200, f"Search failed: {response.text}"
    resp_data = response.json()
    
    # Assert correct response fields
    assert resp_data["query"] == "What is quantum computing?"
    assert resp_data["answer"] == "I cannot determine the answer from the uploaded documents."
    assert resp_data["results"] == []
    assert resp_data["citations"] == [], f"Expected citations to be empty, got {resp_data['citations']}"
    assert resp_data["message"] is None
    
    # Verify gemini_service was NOT called
    mock_gemini_service.generate_answer.assert_not_called()
    logger.info("✓ Search endpoint successfully skipped Gemini on empty retrieval.")


def test_search_endpoint_error_handling():
    """
    Verify search endpoint handles Gemini exceptions and returns HTTP error codes.
    """
    logger.info("Test Case 6: Verifying Search endpoint error mappings...")
    
    # Setup mock to return valid chunks
    mock_retrieval_service.retrieve.return_value = [
        {
            "document_id": 101,
            "chunk_id": 12,
            "score": 0.92,
            "text": "Deep learning is a subset of machine learning.",
            "metadata": {"filename": "deep_learning.txt"}
        }
    ]

    client = TestClient(app)
    suffix = uuid.uuid4().hex[:6]
    email = f"user_{suffix}@ekip-rag.com"
    password = "Password123!"
    client.post("/api/auth/register", json={"email": email, "password": password})
    login_res = client.post("/api/auth/login", json={"email": email, "password": password})
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Quota Exceeded → 429
    mock_gemini_service.generate_answer.side_effect = GeminiQuotaExceededError("Rate limit exceeded")
    res_429 = client.post("/api/search", json={"query": "test", "top_k": 3}, headers=headers)
    assert res_429.status_code == 429, f"Expected 429, got {res_429.status_code}: {res_429.text}"
    assert "quota" in res_429.json().get("detail", "").lower(), f"Expected 'quota' in detail, got: {res_429.json()}"
    logger.info("  - QuotaExceededError correctly mapped to HTTP 429.")

    # 2. Timeout → 504
    mock_gemini_service.generate_answer.side_effect = GeminiTimeoutError("Connection timed out")
    res_504 = client.post("/api/search", json={"query": "test", "top_k": 3}, headers=headers)
    assert res_504.status_code == 504, f"Expected 504, got {res_504.status_code}: {res_504.text}"
    assert "timed out" in res_504.json().get("detail", "").lower() or "timeout" in res_504.json().get("detail", "").lower(), f"Expected 'timed out' or 'timeout' in detail, got: {res_504.json()}"
    logger.info("  - TimeoutError correctly mapped to HTTP 504.")

    # 3. API Error → 502
    mock_gemini_service.generate_answer.side_effect = GeminiAPIError("Server error on Google side")
    res_502 = client.post("/api/search", json={"query": "test", "top_k": 3}, headers=headers)
    assert res_502.status_code == 502, f"Expected 502, got {res_502.status_code}: {res_502.text}"
    assert "api error" in res_502.json().get("detail", "").lower(), f"Expected 'api error' in detail, got: {res_502.json()}"
    logger.info("  - APIError correctly mapped to HTTP 502.")

    # 4. Configuration Error → 500
    mock_gemini_service.generate_answer.side_effect = GeminiConfigurationError("Bad API key")
    res_500 = client.post("/api/search", json={"query": "test", "top_k": 3}, headers=headers)
    assert res_500.status_code == 500, f"Expected 500, got {res_500.status_code}: {res_500.text}"
    assert "configuration" in res_500.json().get("detail", "").lower(), f"Expected 'configuration' in detail, got: {res_500.json()}"
    logger.info("  - ConfigurationError correctly mapped to HTTP 500.")

    logger.info("✓ Search endpoint exception handling verified successfully.")


def run_all_tests():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 7.2 RAG INTEGRATION VERIFICATION")
    logger.info("==========================================================")

    test_api_key_loading()
    test_prompt_builder()
    test_gemini_service_generation()
    test_search_endpoint_with_results()
    test_search_endpoint_empty_results()
    test_search_endpoint_error_handling()

    logger.info("\n==========================================================")
    logger.info("ALL PHASE 7.2 RAG INTEGRATION VERIFICATION TESTS PASSED!")
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
