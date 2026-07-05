# tests/verify_phase_9_6.py
# --------------------------
# Standalone verification script for Phase 9.6 (Collection-Aware Retrieval).
# Verifies metadata generation, single/multiple collection retrieval, user/workspace isolation, and API endpoints.

import os
import sys
import uuid
import logging
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import SessionLocal
from app.models import User, Workspace, Collection, Document
from app.embeddings.chroma_service import ChromaService

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_6")


def verify_phase_9_6():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.6 COLLECTION-AWARE RETRIEVAL VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    client = TestClient(app)

    try:
        # 1. Setup DB entities
        logger.info("Step 1: Creating database records...")
        unique_suffix = uuid.uuid4().hex[:6]
        user_email = f"user_{unique_suffix}@retrieval-test.com"
        user_pw = "SecurePassword123!"
        
        # Register user
        reg = client.post("/api/auth/register", json={"email": user_email, "password": user_pw})
        assert reg.status_code == 201
        user_id = reg.json()["id"]

        # Authenticate
        login = client.post("/api/auth/login", json={"email": user_email, "password": user_pw})
        assert login.status_code == 200
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        # Create workspace manually
        ws = Workspace(owner_id=user_id, name="Verify WS")
        db.add(ws)
        db.commit()
        ws_id = ws.id

        # Insert documents
        doc_a = Document(filename="doc_a.pdf", stored_filename="sa.pdf", file_extension="pdf", mime_type="pdf", file_size=10, file_path="/pa", owner_id=user_id, workspace_id=ws_id)
        doc_b = Document(filename="doc_b.pdf", stored_filename="sb.pdf", file_extension="pdf", mime_type="pdf", file_size=10, file_path="/pb", owner_id=user_id, workspace_id=ws_id)
        db.add_all([doc_a, doc_b])
        db.commit()

        # Create collections
        col_a = Collection(workspace_id=ws_id, owner_id=user_id, name="Collection A")
        col_b = Collection(workspace_id=ws_id, owner_id=user_id, name="Collection B")
        db.add_all([col_a, col_b])
        db.commit()

        # Link document a to collection a, document b to collection b
        col_a.documents.append(doc_a)
        col_b.documents.append(doc_b)
        db.commit()

        logger.info(f"User ID: {user_id}, Workspace ID: {ws_id}")
        logger.info(f"Collection A (ID={col_a.id}) linked to Doc A (ID={doc_a.id})")
        logger.info(f"Collection B (ID={col_b.id}) linked to Doc B (ID={doc_b.id})")

        # 2. Verify Single Collection Metadata Generation & Retrieval
        logger.info("Step 2: Testing single collection search scope...")
        received_filters = []

        def mock_similarity_search(query_embedding, n_results, where):
            received_filters.append(where)
            return {
                "ids": [["chunk_uuid_123"]],
                "distances": [[0.1]],
                "documents": [["Mocked collection RAG content"]],
                "metadatas": [[{"document_id": doc_a.id, "filename": "doc_a.pdf"}]]
            }

        with patch.object(ChromaService, "similarity_search", side_effect=mock_similarity_search):
            with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value="Mocked response"):
                search_payload = {
                    "query": "Explain quantum computing",
                    "collection_ids": [col_a.id],
                    "workspace_id": ws_id
                }
                response = client.post("/api/retrieval/search", json=search_payload, headers=headers)
                assert response.status_code == 200, f"Search failed: {response.text}"
                
                # Check filter
                assert len(received_filters) > 0, "No filters received by ChromaDB!"
                last_filter = received_filters[-1]
                logger.info(f"ChromaDB filter received: {last_filter}")
                assert "document_id" in str(last_filter), "document_id filter not present"
                assert str(doc_a.id) in str(last_filter), "document_id value mismatch"

        # 3. Verify Multiple Collections Retrieval
        logger.info("Step 3: Testing multiple collections search scope...")
        received_filters.clear()

        with patch.object(ChromaService, "similarity_search", side_effect=mock_similarity_search):
            with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value="Mocked response"):
                search_payload = {
                    "query": "Explain neural networks",
                    "collection_ids": [col_a.id, col_b.id],
                    "workspace_id": ws_id
                }
                response = client.post("/api/retrieval/search", json=search_payload, headers=headers)
                assert response.status_code == 200, f"Search failed: {response.text}"
                
                # Check filter
                last_filter = received_filters[-1]
                logger.info(f"ChromaDB filter received (multiple): {last_filter}")
                assert "$in" in str(last_filter), "$in operator missing from filter"
                assert str(doc_a.id) in str(last_filter)
                assert str(doc_b.id) in str(last_filter)

        # 4. Verify User & Workspace Isolation
        logger.info("Step 4: Verifying user isolation gates...")
        # Create user B
        user_b_email = f"user_b_{unique_suffix}@retrieval-test.com"
        reg_b = client.post("/api/auth/register", json={"email": user_b_email, "password": user_pw})
        assert reg_b.status_code == 201
        
        # User B tries to query User A's collection (col_a.id) -> should raise 403 Forbidden
        login_b = client.post("/api/auth/login", json={"email": user_b_email, "password": user_pw})
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}
        
        hack_payload = {
            "query": "Hacking documents",
            "collection_ids": [col_a.id],
            "workspace_id": ws_id
        }
        response_hack = client.post("/api/retrieval/search", json=hack_payload, headers=headers_b)
        assert response_hack.status_code == 403, f"Expected 403 Forbidden, got {response_hack.status_code}"
        logger.info("✓ User isolation validation successfully blocked unauthorized access.")

        # 5. Clean up
        logger.info("Step 5: Cleaning up test records...")
        user = db.get(User, user_id)
        db.delete(user)
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
    logger.info("PASS - PHASE 9.6 COLLECTION RETRIEVAL VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.6 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.6 COLLECTION RETRIEVAL VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.6 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_6()
