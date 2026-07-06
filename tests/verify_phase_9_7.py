# tests/verify_phase_9_7.py
# --------------------------
# Standalone verification script for Phase 9.7 (Multiple Workspaces).
# Verifies CRUD, stats, defaults, isolation gates, and constraints via TestClient.

import os
import sys
import uuid
import logging
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import SessionLocal
from app.models import User, Workspace

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_7")


def verify_phase_9_7():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.7 MULTIPLE WORKSPACES VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Register test user A
        random_suffix = uuid.uuid4().hex[:6]
        email_a = f"alice_{random_suffix}@workspaces-test.com"
        pw = "SecurePassword123!"

        logger.info(f"Step 1: Registering user '{email_a}'...")
        reg_a = client.post("/api/auth/register", json={"email": email_a, "password": pw})
        assert reg_a.status_code == 201, f"Registration failed: {reg_a.text}"
        user_a_id = reg_a.json()["id"]

        # Login A
        login_a = client.post("/api/auth/login", json={"email": email_a, "password": pw})
        assert login_a.status_code == 200
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        # 2. Get default workspace
        logger.info("Step 2: Retrieving default workspace...")
        def_response = client.get("/api/workspaces/default", headers=headers_a)
        assert def_response.status_code == 200, f"Failed to get default workspace: {def_response.text}"
        ws_default = def_response.json()
        assert ws_default["is_default"] == True
        assert ws_default["name"] == "My Workspace"
        logger.info(f"✓ Default workspace verified. ID: {ws_default['id']}")

        # 3. Create second workspace
        logger.info("Step 3: Creating second workspace...")
        create_response = client.post(
            "/api/workspaces",
            json={"name": "Campaigns WS", "description": "Workspace for campaigns"},
            headers=headers_a
        )
        assert create_response.status_code == 201, f"Failed to create workspace: {create_response.text}"
        ws_second = create_response.json()
        assert ws_second["name"] == "Campaigns WS"
        assert ws_second["is_default"] == False
        logger.info(f"✓ Second workspace created successfully. ID: {ws_second['id']}")

        # 4. Prevent duplicate workspace name for the same user
        logger.info("Step 4: Verifying duplicate name prevention...")
        dup_response = client.post(
            "/api/workspaces",
            json={"name": "Campaigns WS"},
            headers=headers_a
        )
        assert dup_response.status_code == 400, "Should have failed with duplicate workspace name"
        logger.info("✓ Duplicate workspace name check successfully rejected.")

        # 5. Update workspace
        logger.info("Step 5: Updating workspace metadata...")
        update_response = client.patch(
            f"/api/workspaces/{ws_second['id']}",
            json={"name": "Marketing Campaigns", "description": "Updated Campaigns Desc"},
            headers=headers_a
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Marketing Campaigns"
        logger.info("✓ Workspace metadata updated successfully.")

        # 6. Change default workspace
        logger.info("Step 6: Swapping default workspaces...")
        set_default_response = client.post(
            f"/api/workspaces/{ws_second['id']}/default",
            headers=headers_a
        )
        assert set_default_response.status_code == 200
        
        # Verify the change
        def_response_2 = client.get("/api/workspaces/default", headers=headers_a)
        assert def_response_2.json()["id"] == ws_second["id"]
        logger.info("✓ Default workspace changed and validated successfully.")

        # 7. Get workspace statistics
        logger.info("Step 7: Retrieving workspace statistics report...")
        stats_response = client.get(
            f"/api/workspaces/{ws_second['id']}/statistics",
            headers=headers_a
        )
        assert stats_response.status_code == 200, f"Failed to load stats: {stats_response.text}"
        stats_data = stats_response.json()
        assert stats_data["document_count"] == 0
        logger.info("✓ Workspace statistics validated.")

        # 8. Check Workspace / User Isolation
        logger.info("Step 8: Verifying cross-user workspace access is blocked...")
        email_b = f"bob_{random_suffix}@workspaces-test.com"
        reg_b = client.post("/api/auth/register", json={"email": email_b, "password": pw})
        assert reg_b.status_code == 201
        
        login_b = client.post("/api/auth/login", json={"email": email_b, "password": pw})
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        # User B attempts to access User A's workspace
        isolated_response = client.get(f"/api/workspaces/{ws_second['id']}", headers=headers_b)
        assert isolated_response.status_code == 403, f"Expected 403 Forbidden, got {isolated_response.status_code}"
        logger.info("✓ Cross-user workspace validation successfully blocked unauthorized access.")

        # 9. Delete safety checks
        logger.info("Step 9: Testing deletion safety gates...")
        # Try to delete default workspace
        del_def_response = client.delete(f"/api/workspaces/{ws_second['id']}", headers=headers_a)
        assert del_def_response.status_code == 400, "Should prevent deleting the default workspace"
        logger.info("✓ Deletion safety gate blocked deleting default workspace.")

        # Set ws_default back as default
        client.post(f"/api/workspaces/{ws_default['id']}/default", headers=headers_a)

        # Now delete ws_second
        del_second_response = client.delete(f"/api/workspaces/{ws_second['id']}", headers=headers_a)
        assert del_second_response.status_code == 200, f"Failed to delete non-default workspace: {del_second_response.text}"
        logger.info("✓ Non-default workspace deleted successfully.")

        # Try to delete the last remaining workspace
        del_last_response = client.delete(f"/api/workspaces/{ws_default['id']}", headers=headers_a)
        assert del_last_response.status_code == 400, "Should prevent deleting the last workspace"
        logger.info("✓ Deletion safety gate blocked deleting the last workspace.")

        # 10. Clean up
        logger.info("Step 10: Cleaning up test records...")
        user_a = db.get(User, user_a_id)
        db.delete(user_a)
        user_b = db.get(User, reg_b.json()["id"])
        db.delete(user_b)
        db.commit()
        logger.info("Cleanup complete.")

    except Exception as e:
        db.rollback()
        logger.error(f"Verification encountered an exception: {e}", exc_info=True)
        raise e
    finally:
        db.close()

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.7 MULTIPLE WORKSPACES VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.7 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.7 MULTIPLE WORKSPACES VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.7 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_7()
