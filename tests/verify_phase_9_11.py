# tests/verify_phase_9_11.py
# ---------------------------
# Standalone integration verification script for Phase 9.11 (Enterprise Dashboard).

import os
import sys
import uuid
import time
import logging
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import SessionLocal
from app.models import User, Workspace, Document, SearchHistory, Conversation, ChatMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_11")


def verify_phase_9_11():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.11 ENTERPRISE DASHBOARD VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Register test user Alice
        random_suffix = uuid.uuid4().hex[:6]
        email_a = f"alice_{random_suffix}@dashboard-test.com"
        pw = "SecurePassword123!"

        logger.info(f"Step 1: Registering user '{email_a}'...")
        reg_a = client.post("/api/auth/register", json={"email": email_a, "password": pw})
        assert reg_a.status_code == 201, f"Registration failed: {reg_a.text}"
        user_a_id = reg_a.json()["id"]

        # Login Alice
        login_a = client.post("/api/auth/login", json={"email": email_a, "password": pw})
        assert login_a.status_code == 200
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        # Retrieve default workspace ID
        ws_res = client.get("/api/workspaces/current", headers=headers_a)
        assert ws_res.status_code == 200
        ws_default_id = ws_res.json()["id"]

        # 2. Add some mock data to populate statistics
        logger.info("Step 2: Provisioning mock conversation and search records for Alice...")
        
        # Create a conversation
        conv = Conversation(workspace_id=ws_default_id, user_id=user_a_id, title="Test Conversation")
        db.add(conv)
        db.commit()

        # Add message
        msg = ChatMessage(conversation_id=conv.id, role="USER", content="Query payload")
        db.add(msg)

        # Add search history
        search = SearchHistory(user_id=user_a_id, workspace_id=ws_default_id, query="Enterprise dashboard structure", execution_time_ms=150, result_count=5)
        db.add(search)
        db.commit()

        # 3. Verify complete dashboard summary endpoint
        logger.info("Step 3: Accessing /api/dashboard summary endpoint...")
        dash_res = client.get("/api/dashboard", headers=headers_a)
        assert dash_res.status_code == 200, f"Dashboard summary failed: {dash_res.text}"
        dash_data = dash_res.json()
        assert dash_data["overview"]["workspace_name"] == "My Workspace"
        assert dash_data["conversation_metrics"]["total_conversations"] == 1
        assert dash_data["search_metrics"]["searches_today"] == 1
        assert len(dash_data["recent_activity"]) >= 2
        logger.info("✓ Dashboard unified summary endpoint validated successfully.")

        # 4. Verify overview endpoint
        logger.info("Step 4: Accessing /api/dashboard/overview endpoint...")
        overview_res = client.get("/api/dashboard/overview", headers=headers_a)
        assert overview_res.status_code == 200
        overview_data = overview_res.json()
        assert overview_data["workspace_name"] == "My Workspace"
        assert overview_data["owner_name"].startswith("alice")  # Default generated full name from email contains email prefix
        logger.info("✓ Dashboard overview endpoint validated successfully.")

        # 5. Verify core metrics endpoint
        logger.info("Step 5: Accessing /api/dashboard/metrics endpoint...")
        metrics_res = client.get("/api/dashboard/metrics", headers=headers_a)
        assert metrics_res.status_code == 200
        metrics_data = metrics_res.json()
        assert metrics_data["conversations"]["total_conversations"] == 1
        assert metrics_data["searches"]["searches_today"] == 1
        logger.info("✓ Dashboard core metrics endpoint validated successfully.")

        # 6. Verify activity endpoint
        logger.info("Step 6: Accessing /api/dashboard/activity endpoint...")
        activity_res = client.get("/api/dashboard/activity", headers=headers_a)
        assert activity_res.status_code == 200
        activity_data = activity_res.json()
        assert len(activity_data) >= 2
        logger.info("✓ Dashboard activity endpoint validated successfully.")

        # 7. Verify storage metrics endpoint
        logger.info("Step 7: Accessing /api/dashboard/storage endpoint...")
        storage_res = client.get("/api/dashboard/storage", headers=headers_a)
        assert storage_res.status_code == 200
        storage_data = storage_res.json()
        assert "total_storage_bytes" in storage_data
        assert "vector_db_size_bytes" in storage_data
        logger.info("✓ Dashboard storage metrics endpoint validated successfully.")

        # 8. Verify Workspace Isolation
        logger.info("Step 8: Verifying workspace boundary isolation checks...")
        email_b = f"bob_{random_suffix}@dashboard-test.com"
        reg_b = client.post("/api/auth/register", json={"email": email_b, "password": pw})
        assert reg_b.status_code == 201
        
        login_b = client.post("/api/auth/login", json={"email": email_b, "password": pw})
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        # Bob tries to query Alice's workspace details using direct dashboard service call (which will raise PermissionError/HTTPException)
        # Wait, since the endpoints get the workspace ID dynamically from the user's active workspace, we should verify that Bob's dashboard shows Bob's workspace, not Alice's.
        bob_dash_res = client.get("/api/dashboard", headers=headers_b)
        assert bob_dash_res.status_code == 200
        assert bob_dash_res.json()["overview"]["workspace_name"] == "My Workspace"
        # Bob should have 0 conversations and 0 searches
        assert bob_dash_res.json()["conversation_metrics"]["total_conversations"] == 0
        assert bob_dash_res.json()["search_metrics"]["searches_today"] == 0
        logger.info("✓ Workspace isolation validated. Bob's dashboard is completely isolated from Alice's.")

        # 9. Clean up test database records
        logger.info("Step 9: Cleaning up test database records...")
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
    logger.info("PASS - PHASE 9.11 ENTERPRISE DASHBOARD VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.11 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.11 ENTERPRISE DASHBOARD VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.11 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_11()
