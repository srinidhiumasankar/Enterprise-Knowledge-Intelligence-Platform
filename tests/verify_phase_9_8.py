# tests/verify_phase_9_8.py
# --------------------------
# Standalone verification script for Phase 9.8 (Workspace Switching).
# Verifies switching APIs, context mapping, and client isolation rules.

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
logger = logging.getLogger("verify_phase_9_8")


def verify_phase_9_8():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.8 WORKSPACE SWITCHING VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Register test user A
        random_suffix = uuid.uuid4().hex[:6]
        email_a = f"alice_{random_suffix}@switching-test.com"
        pw = "SecurePassword123!"

        logger.info(f"Step 1: Registering user '{email_a}'...")
        reg_a = client.post("/api/auth/register", json={"email": email_a, "password": pw})
        assert reg_a.status_code == 201, f"Registration failed: {reg_a.text}"
        user_a_id = reg_a.json()["id"]

        # Login A
        login_a = client.post("/api/auth/login", json={"email": email_a, "password": pw})
        assert login_a.status_code == 200
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        # 2. Get current active workspace (defaults to "My Workspace")
        logger.info("Step 2: Retrieving current active workspace...")
        curr_res = client.get("/api/workspaces/current", headers=headers_a)
        assert curr_res.status_code == 200, f"Failed to get current: {curr_res.text}"
        ws_default = curr_res.json()
        assert ws_default["is_default"] == True
        assert ws_default["name"] == "My Workspace"
        logger.info(f"✓ Default workspace is initially active. ID: {ws_default['id']}")

        # 3. Create second workspace
        logger.info("Step 3: Creating second workspace...")
        create_res = client.post(
            "/api/workspaces",
            json={"name": "Workspace B", "description": "Second workspace environment"},
            headers=headers_a
        )
        assert create_res.status_code == 201
        ws_b = create_res.json()
        assert ws_b["name"] == "Workspace B"

        # 4. Switch active workspace to Workspace B
        logger.info(f"Step 4: Switching active workspace context to ID {ws_b['id']}...")
        switch_res = client.post(f"/api/workspaces/{ws_b['id']}/switch", headers=headers_a)
        assert switch_res.status_code == 200, f"Switch failed: {switch_res.text}"
        assert switch_res.json()["success"] is True
        logger.info("✓ Context switch response succeeded.")

        # 5. Verify the active workspace is now Workspace B
        logger.info("Step 5: Confirming active workspace context switch...")
        curr_res_2 = client.get("/api/workspaces/current", headers=headers_a)
        assert curr_res_2.status_code == 200
        ws_curr = curr_res_2.json()
        assert ws_curr["id"] == ws_b["id"]
        assert ws_curr["name"] == "Workspace B"
        logger.info("✓ Context verification successfully matches Workspace B.")

        # 6. Retrieve active workspace statistics
        logger.info("Step 6: Querying active workspace statistics...")
        stats_res = client.get("/api/workspaces/current/statistics", headers=headers_a)
        assert stats_res.status_code == 200, f"Stats retrieval failed: {stats_res.text}"
        stats_data = stats_res.json()
        assert stats_data["workspace_id"] == ws_b["id"]
        logger.info("✓ Statistics successfully matched current active workspace ID.")

        # 7. Create collection (should automatically use active workspace)
        logger.info("Step 7: Validating dynamic service routing for document collections...")
        col_res = client.post(
            "/api/collections",
            json={"name": "Active Workspace Collection", "description": "Collection scoped automatically"},
            headers=headers_a
        )
        assert col_res.status_code == 201, f"Failed to create collection: {col_res.text}"
        assert col_res.json()["workspace_id"] == ws_b["id"]
        logger.info("✓ Collection automatically routed into the active workspace context.")

        # 8. Create conversation (should automatically use active workspace)
        logger.info("Step 8: Validating dynamic service routing for conversations...")
        conv_res = client.post(
            "/api/conversations",
            json={"title": "Active Thread"},
            headers=headers_a
        )
        assert conv_res.status_code == 201, f"Failed to create conversation: {conv_res.text}"
        assert conv_res.json()["workspace_id"] == ws_b["id"]
        logger.info("✓ Conversation automatically routed into the active workspace context.")

        # 9. Verify Security validations (prevent Bob switching to Alice's workspace)
        logger.info("Step 9: Testing security boundary controls...")
        email_b = f"bob_{random_suffix}@switching-test.com"
        reg_b = client.post("/api/auth/register", json={"email": email_b, "password": pw})
        assert reg_b.status_code == 201
        
        login_b = client.post("/api/auth/login", json={"email": email_b, "password": pw})
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        # Bob switches to Alice's workspace B -> 403 Forbidden
        bob_switch = client.post(f"/api/workspaces/{ws_b['id']}/switch", headers=headers_b)
        assert bob_switch.status_code == 403, f"Expected 403 Forbidden, got {bob_switch.status_code}"
        logger.info("✓ Unauthorized workspace switches successfully blocked.")

        # Try switching to non-existent workspace ID -> 404 Not Found
        invalid_switch = client.post("/api/workspaces/999999/switch", headers=headers_a)
        assert invalid_switch.status_code == 404, f"Expected 404 Not Found, got {invalid_switch.status_code}"
        logger.info("✓ Switching to invalid workspace IDs successfully rejected.")

        # 10. Clean up
        logger.info("Step 10: Cleaning up test data...")
        user_a = db.get(User, user_a_id)
        db.delete(user_a)
        user_b = db.get(User, reg_b.json()["id"])
        db.delete(user_b)
        db.commit()
        logger.info("Cleanup complete.")

    except Exception as e:
        db.rollback()
        logger.error(f"Verification script crashed: {e}", exc_info=True)
        raise e
    finally:
        db.close()

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.8 WORKSPACE SWITCHING VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.8 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.8 WORKSPACE SWITCHING VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.8 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_8()
