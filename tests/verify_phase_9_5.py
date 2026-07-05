# tests/verify_phase_9_5.py
# --------------------------
# Standalone verification script for Phase 9.5 (Document Collections).
# Verifies collections CRUD, document mapping, stats, pagination, and cleanup via TestClient.

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
logger = logging.getLogger("verify_phase_9_5")


def verify_phase_9_5():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.5 DOCUMENT COLLECTIONS VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)

    # 1. Register test user
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

    # 3. Create a default workspace and upload a document
    logger.info("Step 3: Uploading a test document...")
    file_content = "This is a document to test collection unlinking capabilities."
    upload_files = {"file": ("col_test_doc.txt", file_content, "text/plain")}
    
    upload_response = client.post(
        "/api/upload",
        files=upload_files,
        headers=headers
    )
    assert upload_response.status_code == 201, f"Upload failed: {upload_response.text}"
    doc_data = upload_response.json()
    doc_id = doc_data["document_id"]
    logger.info(f"Test document uploaded successfully. ID: {doc_id}")

    # 4. Create collection
    logger.info("Step 4: Creating a document collection...")
    create_response = client.post(
        "/api/collections",
        json={"name": "Marketing Docs", "description": "Docs for campaign Q3"},
        headers=headers
    )
    assert create_response.status_code == 201, f"Collection creation failed: {create_response.text}"
    col_data = create_response.json()
    col_id = col_data["id"]
    logger.info(f"Collection created successfully. ID: {col_id}, Name: {col_data['name']}")

    # 5. Update collection
    logger.info(f"Step 5: Updating collection {col_id} metadata...")
    update_response = client.patch(
        f"/api/collections/{col_id}",
        json={"name": "Marketing Q3", "description": "Updated Campaign Q3 Docs"},
        headers=headers
    )
    assert update_response.status_code == 200, f"Collection update failed: {update_response.text}"
    updated_data = update_response.json()
    assert updated_data["name"] == "Marketing Q3", "Collection name mismatch!"
    logger.info("Collection metadata updated successfully.")

    # 6. Add document to collection
    logger.info(f"Step 6: Linking document {doc_id} to collection {col_id}...")
    link_response = client.post(
        f"/api/collections/{col_id}/documents",
        json={"document_id": doc_id},
        headers=headers
    )
    assert link_response.status_code == 200, f"Failed to link document: {link_response.text}"
    logger.info("Document linked to collection successfully.")

    # 7. List documents in collection
    logger.info(f"Step 7: Listing documents inside collection {col_id}...")
    docs_response = client.get(
        f"/api/collections/{col_id}/documents",
        headers=headers
    )
    assert docs_response.status_code == 200, f"Failed to list documents: {docs_response.text}"
    docs_list = docs_response.json()
    assert len(docs_list) == 1, "Documents listing length mismatch!"
    assert docs_list[0]["id"] == doc_id, "Linked document ID mismatch!"
    logger.info("Collection documents list retrieved successfully.")

    # 8. Collection statistics
    logger.info(f"Step 8: Retrieving collection {col_id} statistics report...")
    stats_response = client.get(
        f"/api/collections/{col_id}/statistics",
        headers=headers
    )
    assert stats_response.status_code == 200, f"Failed to load statistics: {stats_response.text}"
    stats_data = stats_response.json()
    assert stats_data["document_count"] == 1, "Stats document_count mismatch!"
    assert stats_data["total_size"] == len(file_content), "Stats total_size mismatch!"
    logger.info("Collection statistics report verified successfully.")

    # 9. Pagination
    logger.info("Step 9: Verifying collection listing pagination...")
    # Create 3 more collections
    for i in range(3):
        client.post(
            "/api/collections",
            json={"name": f"Collection Test List {i}"},
            headers=headers
        )
    
    list_response = client.get(
        "/api/collections?page=1&page_size=2",
        headers=headers
    )
    assert list_response.status_code == 200, f"Failed to list collections: {list_response.text}"
    list_data = list_response.json()
    assert len(list_data["items"]) == 2, f"Expected 2 collections, got {len(list_data['items'])}"
    assert list_data["total_records"] == 4, f"Expected 4 total records, got {list_data['total_records']}"
    assert list_data["total_pages"] == 2, f"Expected 2 total pages, got {list_data['total_pages']}"
    logger.info("Pagination controls verified successfully.")

    # 10. Remove document from collection
    logger.info(f"Step 10: Unlinking document {doc_id} from collection {col_id}...")
    unlink_response = client.delete(
        f"/api/collections/{col_id}/documents/{doc_id}",
        headers=headers
    )
    assert unlink_response.status_code == 200, f"Failed to unlink document: {unlink_response.text}"
    
    # Confirm docs list is empty now
    docs_response_2 = client.get(
        f"/api/collections/{col_id}/documents",
        headers=headers
    )
    assert len(docs_response_2.json()) == 0, "Collection documents list should be empty after unlinking!"
    logger.info("Document successfully unlinked from collection.")

    # 11. Delete collection metadata
    logger.info(f"Step 11: Deleting collection {col_id}...")
    delete_response = client.delete(
        f"/api/collections/{col_id}",
        headers=headers
    )
    assert delete_response.status_code == 200, f"Collection delete failed: {delete_response.text}"
    
    # Confirm collection is not found
    check_response = client.get(
        f"/api/collections/{col_id}",
        headers=headers
    )
    assert check_response.status_code == 404, "Expected 404 not found for deleted collection!"
    logger.info("Collection metadata deleted successfully.")

    # 12. Cleanup remaining test records
    logger.info("Step 12: Cleaning up remaining test collections and documents...")
    # Delete uploaded document
    doc_del = client.delete(
        f"/api/upload/{doc_id}",
        headers=headers
    )
    # Check if delete is supported, if not, standard rollback or database reset handles it.
    
    # Permanent delete remaining test collections
    clean_list = client.get(
        "/api/collections?page=1&page_size=100",
        headers=headers
    )
    for col in clean_list.json()["items"]:
        client.delete(
            f"/api/collections/{col['id']}",
            headers=headers
        )
    logger.info("Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.5 DOCUMENT COLLECTIONS VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.5 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.5 DOCUMENT COLLECTIONS VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.5 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_5()
