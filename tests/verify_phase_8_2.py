# tests/verify_phase_8_2.py
# -------------------------
# Verification script for Phase 8.2 (LangChain Embedding Pipeline).
# Verifies that:
# 1. LangChain embeddings initialize successfully.
# 2. Text queries can be embedded.
# 3. Documents/chunks can be embedded in batch.
# 4. Dimensions of the generated embeddings match settings.EMBEDDING_DIMENSION (768).
# 5. Generated embeddings can be inserted into ChromaDB.
# 6. Similarity search retrieval works using the generated query embedding.

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.langchain import get_embeddings
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_2")


def verify_phase_8_2():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.2 LANGCHAIN EMBEDDING PIPELINE VERIFICATION")
    logger.info("==========================================================")

    # 1. Initialize LangChain embeddings wrapper
    logger.info("Step 1: Initializing LangChain embeddings wrapper...")
    try:
        embeddings = get_embeddings()
        logger.info(f"✓ LangChain embeddings initialized: {embeddings}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangChain embeddings: {e}", exc_info=True)
        sys.exit(1)

    # 2. Generate embedding for a single text query
    logger.info("Step 2: Generating embedding for a single query...")
    query_text = "What is the future of agentic AI?"
    try:
        query_vector = embeddings.embed_query(query_text)
        logger.info(f"✓ Query embedding generated successfully.")
        logger.info(f"✓ Embedding type: {type(query_vector)}, dimension: {len(query_vector)}")
        assert isinstance(query_vector, list), "Embedding must be a list of floats"
        assert len(query_vector) == settings.EMBEDDING_DIMENSION, f"Expected dimension {settings.EMBEDDING_DIMENSION}, got {len(query_vector)}"
    except Exception as e:
        logger.error(f"❌ Failed to generate query embedding: {e}", exc_info=True)
        sys.exit(1)

    # 3. Embed multiple chunks (batch mode)
    logger.info("Step 3: Generating batch embeddings for multiple chunks...")
    chunks = [
        "Agentic AI systems can autonomously plan and execute complex tasks.",
        "LangChain provides abstractions for integrating LLMs with memory and retrieval systems.",
        "Vector databases like ChromaDB enable fast semantic search over document embeddings."
    ]
    try:
        batch_vectors = embeddings.embed_documents(chunks)
        logger.info("✓ Batch embeddings generated successfully.")
        assert isinstance(batch_vectors, list), "Batch embeddings must be a list of lists"
        assert len(batch_vectors) == len(chunks), f"Expected {len(chunks)} vectors, got {len(batch_vectors)}"
        for idx, vec in enumerate(batch_vectors):
            assert isinstance(vec, list), f"Vector at index {idx} must be a list of floats"
            assert len(vec) == settings.EMBEDDING_DIMENSION, f"Vector at index {idx} expected dimension {settings.EMBEDDING_DIMENSION}, got {len(vec)}"
        logger.info(f"✓ Verified {len(batch_vectors)} batch embeddings of dimension {settings.EMBEDDING_DIMENSION}.")
    except Exception as e:
        logger.error(f"❌ Failed to generate batch embeddings: {e}", exc_info=True)
        sys.exit(1)

    # 4. Initialize ChromaService
    logger.info("Step 4: Initializing ChromaService...")
    try:
        chroma_service = ChromaService()
        logger.info("✓ ChromaService initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaService: {e}", exc_info=True)
        sys.exit(1)

    # 5. Verify Chroma insertion using generated embeddings
    logger.info("Step 5: Inserting test documents and LangChain-generated embeddings into ChromaDB...")
    test_doc_id = 999999  # Unique ID for testing
    chunk_ids = [f"test-chunk-{test_doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"owner_id": 1, "document_id": test_doc_id, "filename": "test_langchain.txt", "chunk_index": i}
        for i in range(len(chunks))
    ]
    try:
        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        
        # Add documents with embeddings to ChromaDB
        chroma_service.add_documents(
            ids=chunk_ids,
            embeddings=batch_vectors,
            metadatas=metadatas,
            documents=chunks
        )
        logger.info("✓ Insertion in ChromaDB succeeded.")
    except Exception as e:
        logger.error(f"❌ Failed to insert documents into ChromaDB: {e}", exc_info=True)
        sys.exit(1)

    # 6. Verify retrieval still works using the generated query embedding
    logger.info("Step 6: Executing similarity search in ChromaDB using query embedding...")
    try:
        search_results = chroma_service.similarity_search(
            query_embedding=query_vector,
            n_results=2,
            where={"document_id": test_doc_id}
        )
        logger.info("✓ Similarity search executed successfully.")
        logger.info(f"Retrieve results: {search_results}")
        
        # Validate that results returned are from our test document
        assert "ids" in search_results and len(search_results["ids"]) > 0, "No results returned from query!"
        assert search_results["metadatas"][0][0]["document_id"] == test_doc_id, "Retrieved result does not match test document ID"
        logger.info("✓ Retrieval verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed during similarity search and verification: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up test data
        try:
            chroma_service.collection.delete(where={"document_id": test_doc_id})
            logger.info("✓ Test data cleaned up from ChromaDB.")
        except Exception as cleanup_err:
            logger.warning(f"Failed to clean up test data: {cleanup_err}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.2 LANGCHAIN EMBEDDING PIPELINE VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS")


if __name__ == "__main__":
    verify_phase_8_2()
