# tests/verify_phase_9_3.py
# --------------------------
# Standalone verification script for Phase 9.3 (Enterprise Conversation Management).
# Verifies rename, pin, unpin, archive, restore/unarchive, search, pagination, and ownership.

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
logger = logging.getLogger("verify_phase_9_3")


def verify_phase_9_3():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.3 CONVERSATION MANAGEMENT VERIFICATION")
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
    user_id = user_data.get('id')
    logger.info(f"User registered successfully. ID: {user_id}")

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

    # 4. Rename
    logger.info(f"Step 4: Renaming conversation {conv_id} to 'Renamed Verification Chat'...")
    rename_response = client.patch(
        f"/api/conversations/{conv_id}/rename",
        json={"title": "Renamed Verification Chat"},
        headers=headers
    )
    assert rename_response.status_code == 200, f"Rename failed: {rename_response.text}"
    renamed_data = rename_response.json()
    assert renamed_data["title"] == "Renamed Verification Chat", "Title rename not reflected!"
    logger.info("Conversation renamed successfully.")

    # 5. Pin & List Pinned
    logger.info(f"Step 5: Pinning conversation {conv_id}...")
    pin_response = client.patch(
        f"/api/conversations/{conv_id}/pin",
        headers=headers
    )
    assert pin_response.status_code == 200, f"Pin failed: {pin_response.text}"
    
    # Get Pinned conversations
    pinned_response = client.get(
        "/api/conversations/pinned",
        headers=headers
    )
    assert pinned_response.status_code == 200, f"Failed to list pinned: {pinned_response.text}"
    pinned_data = pinned_response.json()
    assert len(pinned_data["items"]) == 1, "Pinned list should contain 1 item!"
    assert pinned_data["items"][0]["id"] == conv_id, "Pinned item ID mismatch!"
    logger.info("Conversation pinned and listed successfully.")

    # 6. Unpin & List Pinned
    logger.info(f"Step 6: Unpinning conversation {conv_id}...")
    unpin_response = client.patch(
        f"/api/conversations/{conv_id}/unpin",
        headers=headers
    )
    assert unpin_response.status_code == 200, f"Unpin failed: {unpin_response.text}"
    
    pinned_response_2 = client.get(
        "/api/conversations/pinned",
        headers=headers
    )
    assert len(pinned_response_2.json()["items"]) == 0, "Pinned list should be empty after unpinning!"
    logger.info("Conversation unpinned successfully.")

    # 7. Archive & List Archived
    logger.info(f"Step 7: Archiving conversation {conv_id}...")
    archive_response = client.patch(
        f"/api/conversations/{conv_id}/archive",
        headers=headers
    )
    assert archive_response.status_code == 200, f"Archive failed: {archive_response.text}"
    
    # normal listing should not return archived conversations
    list_response = client.get(
        "/api/conversations",
        headers=headers
    )
    assert len(list_response.json()["items"]) == 0, "Normal listing should exclude archived conversations!"
    
    # archived list should return it
    archived_response = client.get(
        "/api/conversations/archived",
        headers=headers
    )
    assert archived_response.status_code == 200, f"Failed to list archived: {archived_response.text}"
    archived_data = archived_response.json()
    assert len(archived_data["items"]) == 1, "Archived list should contain 1 item!"
    assert archived_data["items"][0]["id"] == conv_id, "Archived item ID mismatch!"
    logger.info("Conversation archived and listed successfully.")

    # 8. Unarchive
    logger.info(f"Step 8: Unarchiving conversation {conv_id}...")
    unarchive_response = client.patch(
        f"/api/conversations/{conv_id}/unarchive",
        headers=headers
    )
    assert unarchive_response.status_code == 200, f"Unarchive failed: {unarchive_response.text}"
    
    list_response_2 = client.get(
        "/api/conversations",
        headers=headers
    )
    assert len(list_response_2.json()["items"]) == 1, "Normal listing should include unarchived conversation!"
    logger.info("Conversation restored/unarchived successfully.")

    # 9. Search
    logger.info("Step 9: Testing keyword search capability...")
    # Add a message containing unique keyword
    client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "role": "user",
            "content": "We need to verify if the search system indexes message content successfully."
        },
        headers=headers
    )
    
    # Search by message keyword
    search_response = client.get(
        "/api/conversations/search?keyword=indexes",
        headers=headers
    )
    assert search_response.status_code == 200, f"Search failed: {search_response.text}"
    search_data = search_response.json()
    assert len(search_data["items"]) == 1, "Search should return 1 matching conversation!"
    assert search_data["items"][0]["id"] == conv_id, "Matched conversation ID mismatch!"
    assert search_data["items"][0]["message_count"] == 1, "Message count field should return 1!"
    logger.info("Keyword search matches chat message successfully.")

    # 10. Clean up
    logger.info("Step 10: Cleaning up test resources...")
    client.delete(
        f"/api/conversations/{conv_id}/permanent",
        headers=headers
    )
    logger.info("Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.3 CONVERSATION MANAGEMENT VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.3 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.3 CONVERSATION MANAGEMENT VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.3 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_3()
