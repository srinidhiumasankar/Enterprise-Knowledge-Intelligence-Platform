# tests/verify_phase_8_4.py
# -------------------------
# Verification script for Phase 8.4 (LangChain RAG Chain Integration).

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.langchain import get_embeddings, create_rag_chain
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_4")


def verify_phase_8_4():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.4 LANGCHAIN RAG CHAIN INTEGRATION VERIFICATION")
    logger.info("==========================================================")

    # Setup test variables
    test_doc_id = 840001
    test_owner_id = 999
    chunks = [
        "Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin.",
        "It was incorporated as Google Inc. and went public through an initial public offering (IPO) in 2004."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})

        # Generate and insert embeddings
        vectors = embeddings.embed_documents(chunks)
        chroma_service.add_documents(
            ids=[f"chunk-84-{test_doc_id}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id, "filename": "google_history.txt", "chunk_index": i}
                for i in range(len(chunks))
            ],
            documents=chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    # 1. Initialize RAG Chain
    logger.info("Step 1: Initializing LangChain RAG chain...")
    try:
        rag_chain = create_rag_chain(owner_id=test_owner_id, top_k=2)
        logger.info("✓ RAG chain initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG chain: {e}", exc_info=True)
        sys.exit(1)

    # 2. Test Grounded Answer
    logger.info("Step 2: Testing grounded answer retrieval and LLM generation...")
    question = "Who founded Google and when?"
    try:
        answer = rag_chain.run(question)
        logger.info(f"Query: '{question}'")
        logger.info(f"Answer: '{answer}'")
        
        # Verify correctness
        assert "Larry Page" in answer or "Sergey Brin" in answer, "Answer is missing founders"
        assert "1998" in answer, "Answer is missing year founded"
        logger.info("✓ Grounded answer verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed grounded answer test: {e}", exc_info=True)
        sys.exit(1)

    # 3. Test Empty Retrieval Handling
    logger.info("Step 3: Testing empty retrieval handling (insufficient information)...")
    insufficient_question = "What is the capital of France?"
    try:
        # France capital is not in the Google history text
        answer_insufficient = rag_chain.run(insufficient_question)
        logger.info(f"Query: '{insufficient_question}'")
        logger.info(f"Answer: '{answer_insufficient}'")
        
        assert answer_insufficient == "Insufficient information found in uploaded documents.", (
            f"Expected fallback string, got: '{answer_insufficient}'"
        )
        logger.info("✓ Empty/insufficient retrieval handling verified successfully.")
    except Exception as e:
        logger.error(f"❌ Failed empty retrieval handling test: {e}", exc_info=True)
        sys.exit(1)

    # 4. Cleanup
    logger.info("Step 4: Cleaning up test data from ChromaDB...")
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.4 LANGCHAIN RAG CHAIN VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS - PHASE 8.4 LANGCHAIN RAG CHAIN VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_4()
