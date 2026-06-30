# tests/verify_api_flow.py
# -----------------------
# End-to-end integration test of the upload, processing, chunking, and vector database flow.

import logging
import os
import sys
import uuid
from fastapi.testclient import TestClient

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.services.vector_service import VectorService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_api_flow")


def run_api_tests():
    logger.info("Starting API End-to-End Flow Verification...")

    # We will use the live FastAPI app via TestClient
    client = TestClient(app)

    # 1. Register a test user
    random_suffix = uuid.uuid4().hex[:6]
    test_email = f"user_{random_suffix}@ekip-test.com"
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

    # 3. Upload a document
    logger.info("Step 3: Uploading dummy text document...")
    file_content = (
        "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+.\n\n"
        "SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that gives application developers "
        "the full power and flexibility of SQL.\n\n"
        "ChromaDB is a database for building AI applications with embeddings."
    )
    upload_files = {"file": ("test_rag_doc.txt", file_content, "text/plain")}
    
    upload_response = client.post(
        "/api/upload",
        files=upload_files,
        headers=headers
    )
    assert upload_response.status_code == 201, f"Upload failed: {upload_response.text}"
    doc_data = upload_response.json()
    doc_id = doc_data["document_id"]
    logger.info(f"Uploaded successfully. Document ID: {doc_id}")

    # 4. Process the document (extract text)
    logger.info(f"Step 4: Processing document {doc_id}...")
    process_response = client.post(
        f"/api/upload/{doc_id}/process",
        headers=headers
    )
    assert process_response.status_code == 200, f"Processing failed: {process_response.text}"
    proc_data = process_response.json()
    logger.info(f"Processed successfully. Status: {proc_data['status']}, Characters: {proc_data['characters']}")

    # 5. Chunk the document (generates chunks, generates embeddings, persists vectors in ChromaDB)
    logger.info(f"Step 5: Chunking document {doc_id}...")
    chunk_response = client.post(
        f"/api/upload/{doc_id}/chunk",
        headers=headers
    )
    assert chunk_response.status_code == 200, f"Chunking failed: {chunk_response.text}"
    chunk_data = chunk_response.json()
    logger.info(f"Chunked successfully. Total Chunks: {chunk_data['total_chunks']}, Average size: {chunk_data['average_chunk_size']}")

    # 6. Verify ChromaDB contains the vectors
    logger.info("Step 6: Verifying vector store contents directly...")
    vector_service = VectorService()
    fetched = vector_service.fetch_vectors_by_document_id(doc_id)
    assert len(fetched["ids"]) == chunk_data["total_chunks"], f"Expected {chunk_data['total_chunks']} vectors in ChromaDB, found {len(fetched['ids'])}"
    
    # Check that metadata and text match
    for idx, cid in enumerate(fetched["ids"]):
        assert fetched["metadatas"][idx]["document_id"] == doc_id
        assert fetched["metadatas"][idx]["chunk_index"] is not None
        assert fetched["documents"][idx] is not None
        assert len(fetched["embeddings"][idx]) == settings.EMBEDDING_DIMENSION
        logger.info(f"Vector verified. Chunk Index: {fetched['metadatas'][idx]['chunk_index']}, ID: {cid}")

    # 7. Check Collection Statistics
    logger.info("Step 7: Verifying collection statistics...")
    stats = vector_service.get_collection_statistics()
    logger.info(f"Collection stats: {stats}")
    assert stats["total_vectors"] >= chunk_data["total_chunks"]

    # 8. Re-chunk the document and verify no duplicates are created
    logger.info("Step 8: Re-chunking same document to test duplicate prevention...")
    rechunk_response = client.post(
        f"/api/upload/{doc_id}/chunk",
        headers=headers
    )
    assert rechunk_response.status_code == 200
    
    # Fetch vectors again
    fetched_after_rechunk = vector_service.fetch_vectors_by_document_id(doc_id)
    assert len(fetched_after_rechunk["ids"]) == chunk_data["total_chunks"], (
        f"Duplicate vectors found! Expected {chunk_data['total_chunks']}, got {len(fetched_after_rechunk['ids'])}"
    )
    logger.info("Duplicate prevention verified: old vectors were successfully deleted before re-insertion.")

    # 9. Delete document (should delete SQLite chunks, disk file, and ChromaDB vectors)
    logger.info(f"Step 9: Deleting document {doc_id} and checking vector cleanup...")
    delete_response = client.delete(
        f"/api/upload/{doc_id}",
        headers=headers
    )
    assert delete_response.status_code == 200, f"Deletion failed: {delete_response.text}"
    logger.info("Document deleted via API.")

    # Verify vectors are gone from ChromaDB
    fetched_after_delete = vector_service.fetch_vectors_by_document_id(doc_id)
    assert len(fetched_after_delete["ids"]) == 0, f"Expected 0 vectors after document deletion, found {len(fetched_after_delete['ids'])}"
    logger.info("Vector cleanup verified successfully! All vectors purged from ChromaDB.")

    logger.info("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    try:
        run_api_tests()
    except Exception as e:
        logger.error(f"E2E Integration Verification Failed: {e}", exc_info=True)
        sys.exit(1)
