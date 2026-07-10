# tests/verify_phase_10_2.py
# --------------------------
# Standalone verification script for Phase 10.2 (Production RAG Integration).

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
logger = logging.getLogger("verify_phase_10_2")

def verify_phase_10_2():
    logger.info("==================================================")
    logger.info("STARTING PHASE 10.2 PRODUCTION RAG VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"rag_prod_{random_suffix}@ekip-test.com"
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
    doc1_content = "Deep Learning uses artificial neural networks to model complex patterns in data."
    files1 = {"file": ("deep_learning.txt", doc1_content, "text/plain")}
    upload1_res = client.post("/api/upload", files=files1, headers=headers)
    assert upload1_res.status_code == 201, f"Upload 1 failed: {upload1_res.text}"
    doc1_id = upload1_res.json()["document_id"]

    # Process & Chunk Document 1
    proc1_res = client.post(f"/api/upload/{doc1_id}/process", headers=headers)
    assert proc1_res.status_code == 200, f"Process 1 failed: {proc1_res.text}"
    chunk1_res = client.post(f"/api/upload/{doc1_id}/chunk", headers=headers)
    assert chunk1_res.status_code == 200, f"Chunk 1 failed: {chunk1_res.text}"
    logger.info("Document 1 uploaded, processed, and chunked successfully.")

    # 4. Create conversation thread
    logger.info("Step 4: Creating conversation thread...")
    conv_res = client.post("/api/conversations", json={"title": "Prod RAG Test"}, headers=headers)
    assert conv_res.status_code == 201, f"Conversation creation failed: {conv_res.text}"
    conv_id = conv_res.json()["id"]

    # 5. Call search endpoint with mock Gemini return value to test flow
    logger.info("Step 5: Testing POST /api/retrieval/search flow...")
    query = "Explain Deep Learning"
    mocked_ans = "Mocked: Deep learning is based on artificial neural networks."
    
    with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value=mocked_ans):
        search_res = client.post(
            "/api/retrieval/search",
            json={"query": query, "top_k": 5},
            headers=headers
        )
        assert search_res.status_code == 200, f"Search API failed: {search_res.text}"
        search_data = search_res.json()
        assert search_data["answer"] == mocked_ans
        assert len(search_data["results"]) > 0
        assert len(search_data["citations"]) > 0

    # 6. Post User message
    logger.info("Step 6: Posting User message...")
    user_msg_res = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "USER", "content": query},
        headers=headers
    )
    assert user_msg_res.status_code == 200

    # 7. Post Assistant message with metadata citations & search results
    logger.info("Step 7: Saving Assistant response with citations...")
    assistant_msg_res = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "role": "ASSISTANT",
            "content": search_data["answer"],
            "model_name": "gemini-2.5-flash",
            "metadata_json": {
                "simulated": False,
                "citations": search_data["citations"],
                "search_results": search_data["results"]
            }
        },
        headers=headers
    )
    assert assistant_msg_res.status_code == 200
    saved_msg = assistant_msg_res.json()
    assert saved_msg["content"] == mocked_ans
    assert saved_msg["metadata_json"]["simulated"] is False
    assert len(saved_msg["metadata_json"]["citations"]) > 0

    # 8. Verify search history log
    logger.info("Step 8: Verifying search history logging...")
    history_res = client.get("/api/search-history", headers=headers)
    assert history_res.status_code == 200
    history_items = history_res.json()["items"]
    assert len(history_items) == 1, "Expected search query recorded in search history!"
    assert history_items[0]["query"] == query

    # Clean up
    logger.info("Cleaning up verification resources...")
    client.delete(f"/api/upload/{doc1_id}", headers=headers)
    client.delete(f"/api/conversations/{conv_id}/permanent", headers=headers)
    logger.info("Cleanup complete.")
    logger.info("==========================================")
    logger.info("PASS - PHASE 10.2 PRODUCTION RAG VERIFIED SUCCESSFULLY")
    logger.info("==========================================")

if __name__ == "__main__":
    verify_phase_10_2()
