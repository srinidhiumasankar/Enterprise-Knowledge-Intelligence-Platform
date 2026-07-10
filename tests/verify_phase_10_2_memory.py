# tests/verify_phase_10_2_memory.py
# ---------------------------------
# Standalone verification script for Phase 10.2 (Multi-Turn Conversation Memory).

import os
import sys
import uuid
import logging
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_10_2_memory")

def verify_phase_10_2_memory():
    logger.info("==================================================")
    logger.info("STARTING PHASE 10.2 MULTI-TURN MEMORY VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"mem_agent_{random_suffix}@ekip-test.com"
    test_password = "SecurePassword123!"

    logger.info(f"Step 1: Registering test user '{test_email}'...")
    reg_response = client.post(
        "/api/auth/register",
        json={"email": test_email, "password": test_password}
    )
    assert reg_response.status_code == 201, f"Registration failed: {reg_response.text}"
    user_id = reg_response.json().get('id')
    logger.info(f"User registered successfully. ID: {user_id}")

    # 2. Login to obtain token
    logger.info("Step 2: Authenticating to obtain JWT token...")
    login_response = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": test_password}
    )
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    access_token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    logger.info("JWT token obtained successfully.")

    # 3. Upload and index Document 1
    logger.info("Step 3: Uploading and chunking Document 1...")
    doc_content = "Machine Learning is a method of data analysis that automates analytical model building."
    files = {"file": ("ml_overview.txt", doc_content, "text/plain")}
    upload_res = client.post("/api/upload", files=files, headers=headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    proc_res = client.post(f"/api/upload/{doc_id}/process", headers=headers)
    assert proc_res.status_code == 200
    chunk_res = client.post(f"/api/upload/{doc_id}/chunk", headers=headers)
    assert chunk_res.status_code == 200
    logger.info("Document uploaded and chunked.")

    # 4. Conversation A - Question 1
    logger.info("Step 4: Creating Conversation A...")
    conv_a_res = client.post("/api/conversations", json={"title": "Conversation A"}, headers=headers)
    assert conv_a_res.status_code == 201
    conv_a_id = conv_a_res.json()["id"]

    query1 = "What is Machine Learning?"
    ans1 = "Machine Learning is an analytical model building automation method."

    # Post User Message 1
    client.post(f"/api/conversations/{conv_a_id}/messages", json={"role": "USER", "content": query1}, headers=headers)

    # Search with query 1
    with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value=ans1) as mock_gemini:
        search_res = client.post(
            "/api/retrieval/search",
            json={"query": query1, "top_k": 5, "conversation_id": conv_a_id},
            headers=headers
        )
        assert search_res.status_code == 200
        # Verify prompt built has NO history (since this is the first message in the conversation)
        called_prompt = mock_gemini.call_args[1]["prompt"]
        assert "=== Conversation History ===" not in called_prompt, "First message should not contain conversation history section."

    # Post Assistant Message 1
    client.post(
        f"/api/conversations/{conv_a_id}/messages",
        json={
            "role": "ASSISTANT",
            "content": ans1,
            "model_name": "gemini-2.5-flash",
            "metadata_json": {"simulated": False}
        },
        headers=headers
    )
    logger.info("✓ Conversation A Message 1 saved.")

    # 5. Conversation A - Question 2 (Multi-turn follow-up)
    logger.info("Step 5: Follow-up question in Conversation A ('Explain it simply')...")
    query2 = "Explain it simply."
    ans2 = "It means computers learn from patterns automatically."

    # Post User Message 2
    client.post(f"/api/conversations/{conv_a_id}/messages", json={"role": "USER", "content": query2}, headers=headers)

    # Search with query 2
    with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value=ans2) as mock_gemini:
        search_res = client.post(
            "/api/retrieval/search",
            json={"query": query2, "top_k": 5, "conversation_id": conv_a_id},
            headers=headers
        )
        assert search_res.status_code == 200
        # Verify prompt built contains Conversation History with previous question & answer
        called_prompt = mock_gemini.call_args[1]["prompt"]
        assert "=== Conversation History ===" in called_prompt, "Prompt should contain the Conversation History header."
        assert "What is Machine Learning?" in called_prompt, "Prompt history should contain first user query."
        assert "building automation" in called_prompt, "Prompt history should contain first assistant answer."

    # Post Assistant Message 2
    client.post(
        f"/api/conversations/{conv_a_id}/messages",
        json={
            "role": "ASSISTANT",
            "content": ans2,
            "model_name": "gemini-2.5-flash",
            "metadata_json": {"simulated": False}
        },
        headers=headers
    )
    logger.info("✓ Conversation A Multi-turn follow-up verified successfully.")

    # 6. Conversation B - Context Isolation Check
    logger.info("Step 6: Creating Conversation B to check isolation...")
    conv_b_res = client.post("/api/conversations", json={"title": "Conversation B"}, headers=headers)
    assert conv_b_res.status_code == 201
    conv_b_id = conv_b_res.json()["id"]

    query3 = "Explain it."
    ans3 = "I don't know what it refers to."

    # Post User Message to Conv B
    client.post(f"/api/conversations/{conv_b_id}/messages", json={"role": "USER", "content": query3}, headers=headers)

    # Search in Conversation B
    with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value=ans3) as mock_gemini:
        search_res = client.post(
            "/api/retrieval/search",
            json={"query": query3, "top_k": 5, "conversation_id": conv_b_id},
            headers=headers
        )
        assert search_res.status_code == 200
        called_prompt = mock_gemini.call_args[1]["prompt"]
        # History segment must NOT exist or must not have Machine Learning context
        assert "What is Machine Learning?" not in called_prompt, "Memory context leaked from Conversation A to Conversation B!"
        logger.info("✓ Conversation isolation verified successfully (no context leakage).")

    # Cleanup
    logger.info("Cleaning up verification resources...")
    client.delete(f"/api/upload/{doc_id}", headers=headers)
    client.delete(f"/api/conversations/{conv_a_id}/permanent", headers=headers)
    client.delete(f"/api/conversations/{conv_b_id}/permanent", headers=headers)
    logger.info("Cleanup complete.")
    logger.info("==========================================")
    logger.info("PASS - PHASE 10.2 CONVERSATION MEMORY VERIFIED SUCCESSFULLY")
    logger.info("==========================================")

if __name__ == "__main__":
    verify_phase_10_2_memory()
