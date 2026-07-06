# tests/verify_phase_9_10.py
# ---------------------------
# Standalone integration verification script for Phase 9.10 (Enterprise Search History).

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
from app.models import User, Workspace, SearchHistory

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_10")


def verify_phase_9_10():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.10 SEARCH HISTORY VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Register test user Alice
        random_suffix = uuid.uuid4().hex[:6]
        email_a = f"alice_{random_suffix}@search-history-test.com"
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

        # 2. Trigger Search History Recording via Retrieval Pipeline
        # We call the collection search endpoint /api/retrieval/search.
        logger.info("Step 2: Triggering search via retrieval API...")
        search_payload = {
            "query": "What are agentic design patterns?",
            "top_k": 3,
            "workspace_id": ws_default_id
        }
        # Call retrieval search
        search_res = client.post("/api/retrieval/search", json=search_payload, headers=headers_a)
        # Even if RAG system says "No chunks retrieved" (200 OK), the search recording should trigger!
        assert search_res.status_code == 200, f"Search request failed: {search_res.text}"
        
        # Wait 1.5 seconds for background daemon thread recording to complete DB insert
        logger.info("Waiting for background search history recording...")
        time.sleep(1.5)

        # 3. Retrieve Search History List
        logger.info("Step 3: Checking list search history endpoint...")
        history_list_res = client.get("/api/search-history", headers=headers_a)
        assert history_list_res.status_code == 200, f"List history failed: {history_list_res.text}"
        history_data = history_list_res.json()
        assert history_data["total_records"] >= 1, "Search history was not recorded in background"
        history_id = history_data["items"][0]["id"]
        logger.info(f"✓ Recorded search history entry found. Query: '{history_data['items'][0]['query']}'")

        # 4. Retrieve Recent Searches
        logger.info("Step 4: Checking recent searches endpoint...")
        recent_res = client.get("/api/search-history/recent", headers=headers_a)
        assert recent_res.status_code == 200
        recent_data = recent_res.json()
        assert len(recent_data) >= 1
        assert recent_data[0]["query"] == "What are agentic design patterns?"
        logger.info("✓ Recent searches successfully matched.")

        # 5. Retrieve Frequent Searches
        logger.info("Step 5: Checking frequent searches endpoint...")
        # Execute another identical search to increment count
        client.post("/api/retrieval/search", json=search_payload, headers=headers_a)
        time.sleep(1.2)

        frequent_res = client.get("/api/search-history/frequent", headers=headers_a)
        assert frequent_res.status_code == 200
        frequent_data = frequent_res.json()
        assert len(frequent_data) >= 1
        assert frequent_data[0]["query"] == "What are agentic design patterns?"
        logger.info(f"✓ Frequent searches matched. Rank count: {frequent_data[0]['count']}")

        # 6. Retrieve Statistics
        logger.info("Step 6: Checking statistics endpoint...")
        stats_res = client.get("/api/search-history/statistics", headers=headers_a)
        assert stats_res.status_code == 200
        stats_data = stats_res.json()
        assert stats_data["total_searches"] >= 2
        assert stats_data["most_frequent_query"] == "What are agentic design patterns?"
        logger.info("✓ Search history analytics statistics validated successfully.")

        # 7. Verify Workspace Isolation
        logger.info("Step 7: Verifying workspace isolation...")
        # Create Workspace B
        ws_b_res = client.post("/api/workspaces", json={"name": "Workspace B"}, headers=headers_a)
        assert ws_b_res.status_code == 201
        ws_b_id = ws_b_res.json()["id"]

        # Switch to Workspace B
        switch_res = client.post(f"/api/workspaces/{ws_b_id}/switch", headers=headers_a)
        assert switch_res.status_code == 200

        # Query recent history inside Workspace B context (should be empty!)
        recent_b_res = client.get("/api/search-history/recent", headers=headers_a)
        assert recent_b_res.status_code == 200
        assert len(recent_b_res.json()) == 0, "History leaked across workspaces!"
        logger.info("✓ Workspace isolation verified successfully. Workspace B history is empty.")

        # Switch back to Workspace A
        client.post(f"/api/workspaces/{ws_default_id}/switch", headers=headers_a)

        # 8. Verify Security Boundaries
        logger.info("Step 8: Verifying security isolation boundaries...")
        email_b = f"bob_{random_suffix}@search-history-test.com"
        reg_b = client.post("/api/auth/register", json={"email": email_b, "password": pw})
        assert reg_b.status_code == 201
        
        login_b = client.post("/api/auth/login", json={"email": email_b, "password": pw})
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        # Bob tries to delete Alice's history entry -> 403 Forbidden
        bob_del = client.delete(f"/api/search-history/{history_id}", headers=headers_b)
        assert bob_del.status_code == 403, f"Expected 403 Forbidden, got {bob_del.status_code}"
        logger.info("✓ Cross-user delete attempt successfully blocked.")

        # 9. Delete single history record
        logger.info("Step 9: Testing delete single history entry...")
        del_res = client.delete(f"/api/search-history/{history_id}", headers=headers_a)
        assert del_res.status_code == 200
        logger.info("✓ Single search history entry deleted successfully.")

        # 10. Clear all history
        logger.info("Step 10: Testing clear all search history...")
        clear_res = client.delete("/api/search-history", headers=headers_a)
        assert clear_res.status_code == 200
        
        recent_clear = client.get("/api/search-history/recent", headers=headers_a)
        assert len(recent_clear.json()) == 0
        logger.info("✓ All search history cleared successfully.")

        # 11. Cleanup test records
        logger.info("Step 11: Cleaning up test database records...")
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
    logger.info("PASS - PHASE 9.10 SEARCH HISTORY VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.10 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.10 SEARCH HISTORY VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.10 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_10()
