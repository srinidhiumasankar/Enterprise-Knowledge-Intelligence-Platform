# tests/verify_phase_9_2.py
# --------------------------
# Standalone verification script for Phase 9.2 (Enterprise Conversation Service).
# Verifies all endpoints (create, retrieve, paginate, append, soft-delete, restore, permanent delete) via TestClient.

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
logger = logging.getLogger("verify_phase_9_2")


def verify_phase_9_2():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.2 CONVERSATION SERVICE VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"agent_{random_suffix}@ekip-test.com"
    test_password = "SecurePassword123!"

    logger.info(f"Step 1: Registering test user '{test_email}'...")
    reg_response = client.post(
        "/api/auth/register",
        json={"email": test_email, "password": test_password}
    )
    assert reg_response.status_code == 201, f"Registration failed: {reg_response.text}"
    user_data = reg_response.json()
    logger.info(f"User registered successfully. ID: {user_data.get('id')}")

    # 2. Login to get access token
    logger.info("Step 2: Authenticating to obtain JWT token...")
    login_response = client.post(
        "/api/auth/login",
        json={"email": test_email, "password": test_password}
    )
    assert login_response.status_code == 200, f"Login failed: {login_response.text}"
    token_data = login_response.json()
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    logger.info("JWT token obtained successfully.")

    # 3. Create conversation
    logger.info("Step 3: Creating a new conversation...")
    create_response = client.post(
        "/api/conversations",
        json={"title": "Verification Chat"},
        headers=headers
    )
    assert create_response.status_code == 201, f"Conversation creation failed: {create_response.text}"
    conv_data = create_response.json()
    conv_id = conv_data["id"]
    logger.info(f"Conversation created successfully. ID: {conv_id}, Title: {conv_data['title']}")

    # 4. Append message
    logger.info(f"Step 4: Appending message to conversation {conv_id}...")
    append_response = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "role": "user",
            "content": "Verify system stability.",
            "token_count": 5
        },
        headers=headers
    )
    assert append_response.status_code == 200, f"Failed to append message: {append_response.text}"
    msg_data = append_response.json()
    logger.info(f"Message appended successfully. ID: {msg_data['id']}, role: {msg_data['role']}")

    # 5. Retrieve conversation details
    logger.info(f"Step 5: Retrieving conversation {conv_id} details...")
    get_response = client.get(
        f"/api/conversations/{conv_id}",
        headers=headers
    )
    assert get_response.status_code == 200, f"Failed to retrieve conversation: {get_response.text}"
    fetched_data = get_response.json()
    assert len(fetched_data["messages"]) == 1, "Messages listing count mismatch!"
    logger.info(f"Conversation retrieved successfully. Active messages count: {len(fetched_data['messages'])}")

    # 6. Pagination check
    logger.info("Step 6: Creating additional conversations for pagination check...")
    # Create 3 more conversations
    for i in range(3):
        client.post(
            "/api/conversations",
            json={"title": f"Chat Page test {i}"},
            headers=headers
        )
    
    list_response = client.get(
        "/api/conversations?page=1&page_size=2",
        headers=headers
    )
    assert list_response.status_code == 200, f"Listing failed: {list_response.text}"
    list_data = list_response.json()
    assert len(list_data["items"]) == 2, f"Expected 2 conversations in page list, got {len(list_data['items'])}"
    assert list_data["total_records"] == 4, f"Expected 4 total records, got {list_data['total_records']}"
    assert list_data["total_pages"] == 2, f"Expected 2 total pages, got {list_data['total_pages']}"
    logger.info("Pagination listing checks verified successfully.")

    # 7. Soft delete check
    logger.info(f"Step 7: Soft-deleting conversation {conv_id}...")
    del_response = client.delete(
        f"/api/conversations/{conv_id}",
        headers=headers
    )
    assert del_response.status_code == 200, f"Failed to soft delete: {del_response.text}"
    logger.info("Conversation soft-deleted successfully.")

    # 8. Restore check
    logger.info(f"Step 8: Restoring conversation {conv_id}...")
    rest_response = client.post(
        f"/api/conversations/{conv_id}/restore",
        headers=headers
    )
    assert rest_response.status_code == 200, f"Failed to restore: {rest_response.text}"
    logger.info("Conversation restored successfully.")

    # 9. Permanent delete check
    logger.info(f"Step 9: Permanently deleting conversation {conv_id}...")
    hard_response = client.delete(
        f"/api/conversations/{conv_id}/permanent",
        headers=headers
    )
    assert hard_response.status_code == 200, f"Failed to permanent delete: {hard_response.text}"
    logger.info("Conversation permanently hard-deleted.")

    # Verify that fetching it now fails with 404
    check_response = client.get(
        f"/api/conversations/{conv_id}",
        headers=headers
    )
    assert check_response.status_code == 404, f"Expected 404 not found, got status {check_response.status_code}"
    logger.info("Verified that permanent delete drops record.")

    # 10. Clean up remaining conversations
    logger.info("Step 10: Cleaning up test conversations database...")
    clean_list_response = client.get(
        "/api/conversations?page=1&page_size=100",
        headers=headers
    )
    for c in clean_list_response.json()["items"]:
        client.delete(
            f"/api/conversations/{c['id']}/permanent",
            headers=headers
        )
    logger.info("Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.2 CONVERSATION SERVICE VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.2 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.2 CONVERSATION SERVICE VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.2 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_2()
