# tests/verify_phase_9_12.py
# -------------------------
# Integration verification script for Phase 9.12 - Enterprise User Interface.

import os
import sys
import logging
import random
from fastapi.testclient import TestClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import SessionLocal
from app.models.user import User
from app.models.workspace import Workspace
from app.models.conversation import Conversation
from app.models.collection import Collection
from app.models.search_history import SearchHistory


def verify_phase_9_12():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.12 ENTERPRISE UI VERIFICATION")
    logger.info("==================================================")

    client = TestClient(app)
    db = SessionLocal()

    # Step 1: Register and Log in a user
    random_suffix = f"{random.randint(100000, 999999)}"
    email = f"ui_test_{random_suffix}@ekip-ui.com"
    password = "Ui_secure_password_123"

    logger.info(f"Step 1: Registering verification user '{email}'...")
    reg_res = client.post("/api/auth/register", json={"email": email, "password": password})
    assert reg_res.status_code == 201, f"Registration failed: {reg_res.text}"

    logger.info("Step 2: Logging in verification user...")
    login_res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Step 3: Fetch pages and verify Sidebar and Navigation compiles
    logger.info("Step 3: Checking HTML routes rendering and navigation sidebars...")
    pages = {
        "/dashboard": "Workspace Dashboard",
        "/conversations": "Conversations",
        "/collections": "Document Collections",
        "/documents": "Document Management",
        "/workspaces": "Workspaces",
        "/search-history": "Search History & Analytics"
    }

    for path, expected_text in pages.items():
        res = client.get(path)
        assert res.status_code == 200, f"Page {path} failed: {res.status_code}"
        assert expected_text in res.text, f"Expected text '{expected_text}' not found in {path}"
        
        # Verify navigation contains expected link attributes
        assert 'href="/dashboard"' in res.text, f"Dashboard navigation link missing in {path}"
        assert 'href="/conversations"' in res.text, f"Conversations navigation link missing in {path}"
        assert 'href="/collections"' in res.text, f"Collections navigation link missing in {path}"
        assert 'href="/documents"' in res.text, f"Documents navigation link missing in {path}"
        assert 'href="/workspaces"' in res.text, f"Workspaces navigation link missing in {path}"
        assert 'href="/search-history"' in res.text, f"Search history navigation link missing in {path}"
        logger.info(f"  [OK] Checked {path} - compiles navigation and header sidebar components successfully.")

    # Step 4: Verify API availability/integration from user context
    logger.info("Step 4: Checking AJAX endpoint responses...")
    
    # 4.1 Workspaces API
    ws_res = client.get("/api/workspaces", headers=headers)
    assert ws_res.status_code == 200, f"Workspaces API failed: {ws_res.text}"
    logger.info("  [OK] Workspaces AJAX integration verified.")

    # 4.2 Conversations API
    conv_res = client.get("/api/conversations", headers=headers)
    assert conv_res.status_code == 200, f"Conversations AJAX integration failed: {conv_res.text}"
    logger.info("  [OK] Conversations AJAX integration verified.")

    # 4.3 Collections API
    coll_res = client.get("/api/collections", headers=headers)
    assert coll_res.status_code == 200, f"Collections AJAX integration failed: {coll_res.text}"
    logger.info("  [OK] Collections AJAX integration verified.")

    # 4.4 Search History API
    sh_res = client.get("/api/search-history", headers=headers)
    assert sh_res.status_code == 200, f"Search History AJAX integration failed: {sh_res.text}"
    logger.info("  [OK] Search History AJAX integration verified.")

    # Step 5: Clean up database records
    logger.info("Step 5: Cleaning up test database records...")
    user = db.query(User).filter(User.email == email).first()
    if user:
        # Delete dependent data
        db.query(SearchHistory).filter(SearchHistory.user_id == user.id).delete()
        db.query(Conversation).filter(Conversation.user_id == user.id).delete()
        db.query(Collection).filter(Collection.workspace_id.in_(
            db.query(Workspace.id).filter(Workspace.owner_id == user.id)
        )).delete()
        db.query(Workspace).filter(Workspace.owner_id == user.id).delete()
        db.query(User).filter(User.id == user.id).delete()
        db.commit()
    db.close()
    logger.info("Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.12 ENTERPRISE UI VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.12 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.12 ENTERPRISE UI VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.12 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        verify_phase_9_12()
    except Exception as e:
        logger.error(f"Verification script crashed: {e}", exc_info=True)
        sys.exit(1)
