# tests/verify_phase_8_3.py
# -------------------------
# Verification script for Phase 8.3 (LangChain Retrieval Integration).
# Verifies initialization, connections, inserting, query retrieval,
# metadata filtering, top-k retrieval, correctness of retrieved docs, and cleanup.

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.langchain import get_embeddings, get_retriever
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_3")


def verify_phase_8_3():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.3 LANGCHAIN RETRIEVAL INTEGRATION VERIFICATION")
    logger.info("==========================================================")

    # 1. Initialize LangChain Retriever
    logger.info("Step 1: Initializing LangChain retriever...")
    try:
        retriever = get_retriever(top_k=2)
        logger.info(f"✓ LangChain retriever initialized successfully: {retriever}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangChain retriever: {e}", exc_info=True)
        sys.exit(1)

    # 2. Verify connection to ChromaDB & insert test documents
    logger.info("Step 2: Connecting to ChromaDB and preparing test data...")
    test_doc_id_1 = 830001
    test_doc_id_2 = 830002
    test_owner_id_1 = 999
    test_owner_id_2 = 888

    chunks_doc_1 = [
        "First document paragraph: LangChain simplifies LLM application building blocks.",
        "Second document paragraph: A retriever defines how documents are queried and retrieved."
    ]
    chunks_doc_2 = [
        "Third document paragraph: ChromaDB acts as our primary high-performance vector store.",
        "Fourth document paragraph: Gemini 2.5 Flash is our default generation LLM."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id_1})
        chroma_service.collection.delete(where={"document_id": test_doc_id_2})

        # Generate embeddings using Phase 8.2 compatible embedding wrapper
        vectors_doc_1 = embeddings.embed_documents(chunks_doc_1)
        vectors_doc_2 = embeddings.embed_documents(chunks_doc_2)

        # Insert Doc 1
        chroma_service.add_documents(
            ids=[f"chunk-1-doc1-{i}" for i in range(len(chunks_doc_1))],
            embeddings=vectors_doc_1,
            metadatas=[
                {"owner_id": test_owner_id_1, "document_id": test_doc_id_1, "filename": "doc1.txt", "chunk_index": i}
                for i in range(len(chunks_doc_1))
            ],
            documents=chunks_doc_1
        )

        # Insert Doc 2
        chroma_service.add_documents(
            ids=[f"chunk-2-doc2-{i}" for i in range(len(chunks_doc_2))],
            embeddings=vectors_doc_2,
            metadatas=[
                {"owner_id": test_owner_id_2, "document_id": test_doc_id_2, "filename": "doc2.txt", "chunk_index": i}
                for i in range(len(chunks_doc_2))
            ],
            documents=chunks_doc_2
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to setup ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    # 3. Retrieve using Query
    logger.info("Step 3: Verifying standard retrieval via LangChain retriever...")
    try:
        retriever_all = get_retriever(top_k=2)
        results = retriever_all.invoke("ChromaDB vector store")
        logger.info(f"✓ Retrieved documents: {[doc.page_content for doc in results]}")
        assert len(results) > 0, "No documents were retrieved!"
    except Exception as e:
        logger.error(f"❌ Failed standard query retrieval: {e}", exc_info=True)
        sys.exit(1)

    # 4. Metadata Filtering (owner isolation)
    logger.info("Step 4: Verifying metadata filtering by owner_id...")
    try:
        # Retrieve filtered by owner 999 (should only return doc 1 text)
        retriever_owner = get_retriever(owner_id=test_owner_id_1, top_k=3)
        results_owner = retriever_owner.invoke("paragraph")
        
        logger.info(f"✓ Retrieved documents for owner {test_owner_id_1}: {[doc.page_content for doc in results_owner]}")
        
        # Verify correctness: none of the returned documents should belong to doc 2
        for doc in results_owner:
            assert doc.metadata["owner_id"] == test_owner_id_1, f"Expected owner_id {test_owner_id_1}, got {doc.metadata['owner_id']}"
            assert "Third" not in doc.page_content and "Fourth" not in doc.page_content, "Retrieved document outside owner filter scope!"
        logger.info("✓ Metadata owner filtering verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed owner filtering test: {e}", exc_info=True)
        sys.exit(1)

    # 5. Metadata Filtering (document ID filtering)
    logger.info("Step 5: Verifying metadata filtering by document_id...")
    try:
        # Retrieve filtered by document 830002 (should only return doc 2 text)
        retriever_doc = get_retriever(document_id=test_doc_id_2, top_k=3)
        results_doc = retriever_doc.invoke("paragraph")
        
        logger.info(f"✓ Retrieved documents for doc {test_doc_id_2}: {[doc.page_content for doc in results_doc]}")
        
        # Verify correctness
        for doc in results_doc:
            assert doc.metadata["document_id"] == test_doc_id_2, f"Expected document_id {test_doc_id_2}, got {doc.metadata['document_id']}"
            assert "First" not in doc.page_content and "Second" not in doc.page_content, "Retrieved document outside document_id filter scope!"
        logger.info("✓ Metadata document ID filtering verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed document ID filtering test: {e}", exc_info=True)
        sys.exit(1)

    # 6. Top-K Retrieval
    logger.info("Step 6: Verifying top-k retrieval bounds...")
    try:
        # Top-k = 1
        retriever_k1 = get_retriever(top_k=1)
        results_k1 = retriever_k1.invoke("paragraph")
        logger.info(f"✓ Retrieved documents for top_k=1: {len(results_k1)}")
        assert len(results_k1) == 1, f"Expected top_k of 1 document, retrieved {len(results_k1)}"
        logger.info("✓ Top-k retrieval bounds verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed top-k retrieval test: {e}", exc_info=True)
        sys.exit(1)

    # 7. Cleanup after test
    logger.info("Step 7: Cleaning up test data from ChromaDB...")
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id_1})
        chroma_service.collection.delete(where={"document_id": test_doc_id_2})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.3 LANGCHAIN RETRIEVER VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS - PHASE 8.3 LANGCHAIN RETRIEVER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_3()
