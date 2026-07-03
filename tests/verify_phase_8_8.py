# tests/verify_phase_8_8.py
# -------------------------
# Verification script for Phase 8.8 (Multi Query Retriever).

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.langchain import get_embeddings, get_llm
from app.services.langchain.multi_query import get_multi_query_retriever
from app.embeddings.chroma_service import ChromaService

# Configure stdout logging formatted as expected
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_8")


def verify_phase_8_8():
    print("========================================================")
    print("Starting Phase 8.8 Verification")
    print("========================================================")

    # Setup test data
    test_doc_id = 880001
    test_owner_id = 999
    chunks = [
        "Explain Retrieval Augmented Generation and its core components.",
        "RAG Architecture uses a retriever and generator model.",
        "How RAG Works: document chunks are combined as context.",
        "Retrieval Augmented Generation Overview: an AI search technique.",
        "Document Retrieval for LLMs matches queries with database texts."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})

        # Insert Doc
        vectors = embeddings.embed_documents(chunks)
        chroma_service.add_documents(
            ids=[f"chunk-88-{test_doc_id}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id, "filename": "rag_details.txt", "chunk_index": i}
                for i in range(len(chunks))
            ],
            documents=chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    original_query = "What is Retrieval Augmented Generation?"
    print(f"\nOriginal Query\n\"{original_query}\"")

    # 1. Initialize Multi Query Retriever
    try:
        retriever = get_multi_query_retriever(
            owner_id=test_owner_id,
            top_k=4,
            llm=get_llm()
        )
        logger.info("✓ Multi Query Retriever initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Invoke and verify retrieval
    try:
        fused_docs = retriever.invoke(original_query)
        
        # Verify result requirements
        assert len(fused_docs) <= 4, f"Top-k limit exceeded. Got {len(fused_docs)} documents"
        assert len(fused_docs) > 0, "No documents returned"
        
        # Deduplication check
        chunk_ids = [d.metadata["chunk_id"] for d in fused_docs]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"
        
        # Metadata preservation check
        for doc in fused_docs:
            assert doc.metadata["owner_id"] == test_owner_id, "owner_id metadata lost"
            assert doc.metadata["document_id"] == test_doc_id, "document_id metadata lost"
            assert "filename" in doc.metadata, "filename metadata lost"
            assert "multi_query_rrf_score" in doc.metadata, "multi_query_rrf_score missing"
            
        logger.info("✓ Retrieval successfully returned ranked unique documents with preserved metadata.")
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}", exc_info=True)
        sys.exit(1)

    # Clean up test data
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    print("\nVerification completed successfully")
    print("========================================================")
    print("PASS - PHASE 8.8 MULTI QUERY RETRIEVER VERIFIED SUCCESSFULLY")
    print("========================================================")


if __name__ == "__main__":
    verify_phase_8_8()
