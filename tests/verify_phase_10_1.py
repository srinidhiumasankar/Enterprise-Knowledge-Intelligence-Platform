# tests/verify_phase_10_1.py
# --------------------------
# Standalone verification script for Phase 10.1 (Offline RAG Integration).

import os
import sys
import uuid
import logging
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_10_1")

def verify_phase_10_1():
    logger.info("==================================================")
    logger.info("STARTING PHASE 10.1 OFFLINE RAG INTEGRATION VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"rag_agent_{random_suffix}@ekip-test.com"
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
    doc1_content = "This is a document about machine learning. Machine learning is a subfield of artificial intelligence."
    files1 = {"file": ("ml_notes.txt", doc1_content, "text/plain")}
    upload1_res = client.post("/api/upload", files=files1, headers=headers)
    assert upload1_res.status_code == 201, f"Upload 1 failed: {upload1_res.text}"
    doc1_id = upload1_res.json()["document_id"]

    # Process & Chunk Document 1
    proc1_res = client.post(f"/api/upload/{doc1_id}/process", headers=headers)
    assert proc1_res.status_code == 200, f"Process 1 failed: {proc1_res.text}"
    chunk1_res = client.post(f"/api/upload/{doc1_id}/chunk", headers=headers)
    assert chunk1_res.status_code == 200, f"Chunk 1 failed: {chunk1_res.text}"
    logger.info("Document 1 uploaded, processed, and chunked successfully.")

    # 4. Upload and index Document 2
    logger.info("Step 4: Uploading and chunking Document 2...")
    doc2_content = "This is a spec sheet on platform architecture. The architecture contains FastAPI, SQLite, and ChromaDB."
    files2 = {"file": ("architecture_specs.txt", doc2_content, "text/plain")}
    upload2_res = client.post("/api/upload", files=files2, headers=headers)
    assert upload2_res.status_code == 201, f"Upload 2 failed: {upload2_res.text}"
    doc2_id = upload2_res.json()["document_id"]

    # Process & Chunk Document 2
    proc2_res = client.post(f"/api/upload/{doc2_id}/process", headers=headers)
    assert proc2_res.status_code == 200, f"Process 2 failed: {proc2_res.text}"
    chunk2_res = client.post(f"/api/upload/{doc2_id}/chunk", headers=headers)
    assert chunk2_res.status_code == 200, f"Chunk 2 failed: {chunk2_res.text}"
    logger.info("Document 2 uploaded, processed, and chunked successfully.")

    # 5. Create conversation thread
    logger.info("Step 5: Creating conversation thread...")
    conv_res = client.post("/api/conversations", json={"title": "RAG Chat Test"}, headers=headers)
    assert conv_res.status_code == 201, f"Conversation creation failed: {conv_res.text}"
    conv_id = conv_res.json()["id"]

    # 6. Verify "List available documents" prompt
    logger.info("Step 6: Testing list available documents prompt...")
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "List available documents"},
        headers=headers
    )
    # Simulate client simulated post (which backend should intercept and bypass duplicate entry)
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    # Verify assistant output
    conv_details = client.get(f"/api/conversations/{conv_id}", headers=headers)
    messages = conv_details.json()["messages"]
    # We should have user message + assistant response
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assistant_msg = messages[-1]["content"]
    assert "ml_notes.txt" in assistant_msg, "ml_notes.txt not listed in available documents summary"
    assert "architecture_specs.txt" in assistant_msg, "architecture_specs.txt not listed"
    logger.info("✓ List available documents verified successfully.")

    # 7. Verify "Summarize first document" prompt
    logger.info("Step 7: Testing summarize first document prompt...")
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Summarize first document"},
        headers=headers
    )
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    conv_details = client.get(f"/api/conversations/{conv_id}", headers=headers)
    messages = conv_details.json()["messages"]
    assistant_msg = messages[-1]["content"]
    assert "ml_notes.txt" in assistant_msg, "Document name ml_notes.txt missing in summary"
    assert "machine learning" in assistant_msg.lower(), "Summary didn't retrieve content from first document"
    assert "architecture" not in assistant_msg.lower(), "Summary leaked content from second document"
    logger.info("✓ Summarize first document verified successfully.")

    # 8. Verify "Summarize second document" prompt
    logger.info("Step 8: Testing summarize second document prompt...")
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Summarize second document"},
        headers=headers
    )
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    conv_details = client.get(f"/api/conversations/{conv_id}", headers=headers)
    messages = conv_details.json()["messages"]
    assistant_msg = messages[-1]["content"]
    assert "architecture_specs.txt" in assistant_msg, "Document name architecture_specs.txt missing in summary"
    assert "architecture" in assistant_msg.lower(), "Summary didn't retrieve content from second document"
    assert "machine learning" not in assistant_msg.lower(), "Summary leaked content from first document"
    logger.info("✓ Summarize second document verified successfully.")

    # 9. Verify "Compare both documents" prompt
    logger.info("Step 9: Testing compare both documents prompt...")
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Compare both documents"},
        headers=headers
    )
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    conv_details = client.get(f"/api/conversations/{conv_id}", headers=headers)
    messages = conv_details.json()["messages"]
    assistant_msg = messages[-1]["content"]
    assert "ml_notes.txt" in assistant_msg and "architecture_specs.txt" in assistant_msg, "Comparison missing files"
    logger.info("✓ Compare both documents verified successfully.")

    # 10. Verify general RAG retrieval query ("Explain machine learning")
    logger.info("Step 10: Testing general RAG retrieval query...")
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "user", "content": "Explain machine learning"},
        headers=headers
    )
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    conv_details = client.get(f"/api/conversations/{conv_id}", headers=headers)
    messages = conv_details.json()["messages"]
    assistant_msg = messages[-1]["content"]
    assert "machine learning" in assistant_msg.lower(), "General RAG search query failed to locate ML content"
    logger.info("✓ General RAG retrieval query verified successfully.")

    # 11. Verify Search History updates after every query
    logger.info("Step 11: Verifying search history logging...")
    history_res = client.get("/api/search-history", headers=headers)
    assert history_res.status_code == 200, f"Failed to list search history: {history_res.text}"
    history_data = history_res.json()["items"]
    # We ran 5 user RAG queries, so there must be 5 search history entries logged!
    assert len(history_data) == 5, f"Expected 5 search history records, got {len(history_data)}"
    logger.info("✓ Search history recording verified successfully.")

    # 12. Verify Dashboard query count updates
    logger.info("Step 12: Verifying dashboard statistics update...")
    dashboard_res = client.get("/api/dashboard", headers=headers)
    assert dashboard_res.status_code == 200, f"Failed to get dashboard statistics: {dashboard_res.text}"
    db_data = dashboard_res.json()
    searches_today = db_data["search_metrics"]["searches_today"]
    assert searches_today == 5, f"Expected 5 searches logged today on dashboard, got {searches_today}"
    logger.info("✓ Dashboard search metrics updates verified successfully.")

    # 13. Verify Workspace Isolation
    logger.info("Step 13: Verifying workspace isolation (empty workspace fallback)...")
    # Create second workspace for user
    ws_create_res = client.post("/api/workspaces", json={"name": "Workspace Isolation Test", "description": "isolation check"}, headers=headers)
    assert ws_create_res.status_code == 201, f"Failed to create workspace: {ws_create_res.text}"
    isolated_ws_id = ws_create_res.json()["id"]
    
    # Switch to the new empty workspace
    switch_res = client.post(f"/api/workspaces/{isolated_ws_id}/switch", headers=headers)
    assert switch_res.status_code == 200, f"Failed to switch workspace context: {switch_res.text}"

    # Create new isolated conversation
    iso_conv_res = client.post("/api/conversations", json={"title": "Isolated RAG Chat"}, headers=headers)
    assert iso_conv_res.status_code == 201, f"Failed to create conversation: {iso_conv_res.text}"
    iso_conv_id = iso_conv_res.json()["id"]

    # Post query to empty workspace conversation
    client.post(
        f"/api/conversations/{iso_conv_id}/messages",
        json={"role": "user", "content": "Summarize uploaded documents"},
        headers=headers
    )
    client.post(
        f"/api/conversations/{iso_conv_id}/messages",
        json={"role": "assistant", "content": "Simulated output"},
        headers=headers
    )
    iso_details = client.get(f"/api/conversations/{iso_conv_id}", headers=headers)
    iso_assistant_msg = iso_details.json()["messages"][-1]["content"]
    assert "There are no uploaded documents." in iso_assistant_msg, f"Expected 'There are no uploaded documents.', got '{iso_assistant_msg}'"
    logger.info("✓ Workspace isolation and empty workspace fallback verified successfully.")

    # Clean up test user resources
    logger.info("Cleaning up verification resources...")
    for d_id in [doc1_id, doc2_id]:
        client.delete(f"/api/upload/{d_id}", headers=headers)
    client.delete(f"/api/conversations/{conv_id}/permanent", headers=headers)
    client.delete(f"/api/conversations/{iso_conv_id}/permanent", headers=headers)
    client.delete(f"/api/workspaces/{isolated_ws_id}", headers=headers)
    logger.info("Cleanup complete.")
    logger.info("==========================================")
    logger.info("PASS - PHASE 10.1 OFFLINE RAG INTEGRATION VERIFIED SUCCESSFULLY")
    logger.info("==========================================")

if __name__ == "__main__":
    verify_phase_10_1()
