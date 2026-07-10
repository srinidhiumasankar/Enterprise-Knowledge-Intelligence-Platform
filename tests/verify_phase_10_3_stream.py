# tests/verify_phase_10_3_stream.py
# ---------------------------------
# Standalone verification script for Phase 10.3 (Streaming RAG answers).

import os
import sys
import uuid
import json
import logging
from unittest.mock import patch
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_10_3_stream")

def verify_phase_10_3_stream():
    logger.info("==================================================")
    logger.info("STARTING PHASE 10.3 STREAMING RAG VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"stream_agent_{random_suffix}@ekip-test.com"
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

    # 3. Create conversation
    logger.info("Step 3: Creating conversation thread...")
    conv_res = client.post("/api/conversations", json={"title": "Streaming Test"}, headers=headers)
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    # 4. Stream response and parse event blocks
    logger.info("Step 4: Executing streaming request POST /api/conversations/{conv_id}/stream ...")
    query = "What is enterprise artificial intelligence?"
    
    mock_chunks = [
        "Enterprise ", "Artificial ", "Intelligence ", "enables ", "businesses ", "to ", "automate ", "decisions."
    ]

    with patch("app.ai.gemini_service.GeminiService.generate_stream", return_value=mock_chunks):
        response = client.post(
            f"/api/conversations/{conv_id}/stream",
            json={"query": query, "top_k": 5},
            headers=headers
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        parsed_events = []
        # Parse stream lines
        for line in response.iter_lines():
            if line:
                decoded_line = line if isinstance(line, str) else line.decode("utf-8")
                decoded_line = decoded_line.strip()
                if decoded_line:
                    event = json.loads(decoded_line)
                    parsed_events.append(event)

        logger.info(f"Parsed {len(parsed_events)} streaming JSON event blocks from backend response stream.")
        
        # Verify event structures
        assert parsed_events[0]["type"] == "metadata", "First event block must be metadata."
        assert "citations" in parsed_events[0]
        assert "results" in parsed_events[0]

        text_events = [e for e in parsed_events if e["type"] == "text"]
        assert len(text_events) == len(mock_chunks), f"Expected {len(mock_chunks)} text events, got {len(text_events)}."
        
        accumulated_text = "".join(e["content"] for e in text_events)
        expected_accumulated = "".join(mock_chunks)
        assert accumulated_text == expected_accumulated

        done_event = parsed_events[-1]
        assert done_event["type"] == "done", "Last event must be done."
        assert done_event["content"] == expected_accumulated
        assert "message_id" in done_event

    # 5. Check if assistant response message was saved in DB
    logger.info("Step 5: Verifying completed response message persistence in DB...")
    messages_res = client.get(f"/api/conversations/{conv_id}", headers=headers)
    assert messages_res.status_code == 200
    messages = messages_res.json()["messages"]
    
    # We should have the USER message (if posted, wait, in this test we did not post USER message before calling stream,
    # so we should have just the ASSISTANT message that was auto-saved by stream completion!)
    # Let's verify that the assistant message exists in the conversation messages
    assistant_msgs = [m for m in messages if m["role"].lower() == "assistant"]
    assert len(assistant_msgs) == 1, "There should be exactly one ASSISTANT message saved."
    assert assistant_msgs[0]["content"] == expected_accumulated
    assert assistant_msgs[0]["metadata_json"]["simulated"] is False

    # Cleanup
    logger.info("Cleaning up verification resources...")
    client.delete(f"/api/conversations/{conv_id}/permanent", headers=headers)
    logger.info("Cleanup complete.")
    logger.info("==========================================")
    logger.info("PASS - PHASE 10.3 STREAMING RAG VERIFIED SUCCESSFULLY")
    logger.info("==========================================")

if __name__ == "__main__":
    verify_phase_10_3_stream()
