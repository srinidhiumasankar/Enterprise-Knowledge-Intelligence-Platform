# tests/verify_phase_5_3.py
# -------------------------
# Automated verification script for Phase 5.3: Embedding Generation and Semantic Retrieval.

import logging
import os
import sys
import uuid
from typing import List
from fastapi.testclient import TestClient

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_5_3")


def run_verification():
    logger.info("======================================================================")
    logger.info("STARTING PHASE 5.3 EMBEDDING AND SEMANTIC RETRIEVAL VERIFICATION")
    logger.info("======================================================================")

    # Use the live FastAPI app via TestClient
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # TEST 1: Embedding Service Singleton & Lazy Loading
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 1: Embedding Service Singleton ---")
    model_1 = EmbeddingService.get_model()
    model_2 = EmbeddingService.get_model()
    assert model_1 is model_2, "Embedding model singleton failed! Different model instances found."
    logger.info("✓ EmbeddingService singleton verified successfully.")

    # -------------------------------------------------------------------------
    # TEST 2: Chroma Client Singleton Pattern
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 2: Chroma Client Singleton ---")
    vs_1 = VectorService()
    vs_2 = VectorService()
    assert vs_1.client is vs_2.client, "Chroma Client singleton failed! Different client instances found."
    logger.info("✓ VectorService client singleton verified successfully.")

    # -------------------------------------------------------------------------
    # SETUP: Register two separate test users (User A & User B)
    # -------------------------------------------------------------------------
    logger.info("\n--- SETUP: Creating Test Users ---")
    suffix_a = uuid.uuid4().hex[:6]
    suffix_b = uuid.uuid4().hex[:6]
    email_a = f"usera_{suffix_a}@ekip.com"
    email_b = f"userb_{suffix_b}@ekip.com"
    password = "SecurePassword123!"

    # User A Registration & Authentication
    reg_a = client.post("/api/auth/register", json={"email": email_a, "password": password})
    assert reg_a.status_code == 201, f"User A registration failed: {reg_a.text}"
    token_a = client.post("/api/auth/login", json={"email": email_a, "password": password}).json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}
    logger.info(f"User A registered and authenticated (ID: {reg_a.json()['id']})")

    # User B Registration & Authentication
    reg_b = client.post("/api/auth/register", json={"email": email_b, "password": password})
    assert reg_b.status_code == 201, f"User B registration failed: {reg_b.text}"
    token_b = client.post("/api/auth/login", json={"email": email_b, "password": password}).json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    logger.info(f"User B registered and authenticated (ID: {reg_b.json()['id']})")

    # -------------------------------------------------------------------------
    # SETUP: Upload and process documents for User A and User B
    # -------------------------------------------------------------------------
    logger.info("\n--- SETUP: Uploading and chunking documents ---")
    
    # Doc A (belonging to User A) contains information about "LangChain"
    content_a = (
        "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).\n\n"
        "It provides modular components like prompt templates, models, and memory buffers to string together chains.\n\n"
        "LangChain enables developers to build powerful RAG pipelines and agents easily."
    )
    upload_a = client.post(
        "/api/upload",
        files={"file": ("langchain_guide.txt", content_a, "text/plain")},
        headers=headers_a
    )
    assert upload_a.status_code == 201
    doc_id_a = upload_a.json()["document_id"]
    
    # Process & Chunk Doc A
    client.post(f"/api/upload/{doc_id_a}/process", headers=headers_a)
    chunk_res_a = client.post(f"/api/upload/{doc_id_a}/chunk", headers=headers_a)
    assert chunk_res_a.status_code == 200
    logger.info(f"User A Document chunked successfully (Doc ID: {doc_id_a}, Chunks: {chunk_res_a.json()['total_chunks']})")

    # Doc B (belonging to User B) contains information about "FastAPI"
    content_b = (
        "FastAPI is a modern, fast, high-performance web framework for building APIs with Python 3.8+.\n\n"
        "It is built on top of Starlette for web parts and Pydantic for data parts.\n\n"
        "FastAPI is standard-based, using OpenAPI and JSON Schema."
    )
    upload_b = client.post(
        "/api/upload",
        files={"file": ("fastapi_guide.txt", content_b, "text/plain")},
        headers=headers_b
    )
    assert upload_b.status_code == 201
    doc_id_b = upload_b.json()["document_id"]
    
    # Process & Chunk Doc B
    client.post(f"/api/upload/{doc_id_b}/process", headers=headers_b)
    chunk_res_b = client.post(f"/api/upload/{doc_id_b}/chunk", headers=headers_b)
    assert chunk_res_b.status_code == 200
    logger.info(f"User B Document chunked successfully (Doc ID: {doc_id_b}, Chunks: {chunk_res_b.json()['total_chunks']})")

    # -------------------------------------------------------------------------
    # TEST 3: Duplicate Prevention
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 3: Duplicate Prevention ---")
    # Trigger chunking of Doc A again
    rechunk_res = client.post(f"/api/upload/{doc_id_a}/chunk", headers=headers_a)
    assert rechunk_res.status_code == 200
    
    # Check vectors count for User A's document directly in ChromaDB
    fetched_a = vs_1.fetch_vectors_by_document_id(doc_id_a)
    expected_chunks = chunk_res_a.json()["total_chunks"]
    assert len(fetched_a["ids"]) == expected_chunks, f"Duplicates detected! Expected {expected_chunks} vectors, found {len(fetched_a['ids'])}"
    logger.info("✓ Duplicate prevention verified successfully. Repeating chunking skipped embedding regeneration and preserved existing records.")

    # -------------------------------------------------------------------------
    # TEST 4: Semantic Search & Ranking
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 4: Semantic Search & Ranking ---")
    search_res_a = client.post(
        "/api/search",
        json={"query": "Explain LangChain and modular components", "top_k": 3},
        headers=headers_a
    )
    assert search_res_a.status_code == 200, f"Search failed: {search_res_a.text}"
    search_data_a = search_res_a.json()
    
    assert search_data_a["query"] == "Explain LangChain and modular components"
    results = search_data_a["results"]
    assert len(results) > 0, "No results returned for relevant query!"
    
    # Verify scores are sorted in descending order and bounded in [0, 1]
    last_score = 1.0
    for r in results:
        assert 0.0 <= r["score"] <= 1.0, f"Score {r['score']} is out of bounds!"
        assert r["score"] <= last_score, "Results are not sorted by similarity score descending!"
        assert r["document_id"] == doc_id_a, f"Retrieved document ID {r['document_id']} does not match expected Doc A!"
        assert r["chunk_id"] is not None
        assert "text" in r and len(r["text"]) > 0
        last_score = r["score"]
        logger.info(f"Result - Doc: {r['document_id']}, Chunk ID: {r['chunk_id']}, Score: {r['score']}, Snippet: {r['text'][:50]}...")
        
    logger.info("✓ Semantic search, ranking, formatting, and scores verified successfully.")

    # -------------------------------------------------------------------------
    # TEST 5: User Isolation (Security check)
    # -------------------------------------------------------------------------
    logger.info("\n--- TEST 5: User Isolation Security Check ---")
    
    # User A queries for "FastAPI" (which only User B has)
    search_res_a_isolated = client.post(
        "/api/search",
        json={"query": "FastAPI high-performance web framework", "top_k": 5},
        headers=headers_a
    )
    assert search_res_a_isolated.status_code == 200
    results_a_isolated = search_res_a_isolated.json()["results"]
    
    # Ensure User A NEVER sees User B's document chunks
    for r in results_a_isolated:
        assert r["document_id"] != doc_id_b, f"SECURITY BREACH! User A retrieved User B's document {doc_id_b}."
    logger.info("✓ Verified: User A cannot retrieve User B's embeddings.")

    # User B queries for "LangChain" (which only User A has)
    search_res_b_isolated = client.post(
        "/api/search",
        json={"query": "LangChain modular framework", "top_k": 5},
        headers=headers_b
    )
    assert search_res_b_isolated.status_code == 200
    results_b_isolated = search_res_b_isolated.json()["results"]
    
    # Ensure User B NEVER sees User A's document chunks
    for r in results_b_isolated:
        assert r["document_id"] != doc_id_a, f"SECURITY BREACH! User B retrieved User A's document {doc_id_a}."
    logger.info("✓ Verified: User B cannot retrieve User A's embeddings.")

    # -------------------------------------------------------------------------
    # CLEANUP: Delete documents for both users
    # -------------------------------------------------------------------------
    logger.info("\n--- CLEANUP: Deleting test documents ---")
    del_a = client.delete(f"/api/upload/{doc_id_a}", headers=headers_a)
    assert del_a.status_code == 200
    del_b = client.delete(f"/api/upload/{doc_id_b}", headers=headers_b)
    assert del_b.status_code == 200
    logger.info("✓ Test cleanup completed successfully.")

    logger.info("\n======================================================================")
    logger.info("ALL PHASE 5.3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    logger.info("======================================================================")


if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        logger.error(f"Verification script failed: {e}", exc_info=True)
        sys.exit(1)
