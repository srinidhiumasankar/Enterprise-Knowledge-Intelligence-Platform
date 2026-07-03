# tests/verify_phase_8_7.py
# -------------------------
# Verification script for Phase 8.7 (Hybrid Search Integration).

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.langchain import get_embeddings, get_hybrid_retriever
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_7")


def verify_phase_8_7():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.7 HYBRID SEARCH VERIFICATION")
    logger.info("==========================================================")

    # Setup test data
    test_doc_id = 870001
    test_owner_id = 999
    chunks = [
        "Semantic matching focuses on abstract vector embeddings: similarity search.",
        "Keyword matching matches exact terms: BM25 TF-IDF tokenization matching text.",
        "Duplicate chunk is present for duplicate removal validation test case."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})

        # Insert Doc
        vectors = embeddings.embed_documents(chunks)
        chroma_service.add_documents(
            ids=[f"chunk-87-{test_doc_id}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id, "filename": "hybrid_notes.txt", "chunk_index": i}
                for i in range(len(chunks))
            ],
            documents=chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    # 1. Initialize Hybrid Retriever & Config Weights
    logger.info("Step 1: Initializing Hybrid Retriever and verifying weights configuration...")
    try:
        semantic_weight = 0.7
        keyword_weight = 0.3
        
        # Verify settings configuration variables exist
        assert settings.HYBRID_SEMANTIC_WEIGHT == 0.7, f"Expected semantic weight 0.7, got {settings.HYBRID_SEMANTIC_WEIGHT}"
        assert settings.HYBRID_KEYWORD_WEIGHT == 0.3, f"Expected keyword weight 0.3, got {settings.HYBRID_KEYWORD_WEIGHT}"
        
        retriever = get_hybrid_retriever(
            owner_id=test_owner_id,
            top_k=2,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            chroma_service=chroma_service,
            embedding_service=embeddings
        )
        logger.info("✓ Hybrid Retriever initialized with custom weights successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Test Semantic Search Path
    logger.info("Step 2: Testing semantic search retrieval path...")
    try:
        results_semantic = retriever.chroma_service.similarity_search(
            query_embedding=embeddings.embed_query("abstract embeddings similarity"),
            n_results=2,
            where={"document_id": test_doc_id}
        )
        docs_s = results_semantic.get("documents", [[]])[0] if results_semantic.get("documents") else []
        logger.info(f"Semantic search raw results: {docs_s}")
        assert len(docs_s) > 0, "No semantic documents retrieved"
        logger.info("✓ Semantic retrieval path verified successfully.")
    except Exception as e:
        logger.error(f"❌ Semantic path test failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Test Keyword Search Path (BM25)
    logger.info("Step 3: Testing keyword search retrieval path...")
    try:
        all_chunks = retriever.chroma_service.collection.get(where={"document_id": test_doc_id})
        docs_all = all_chunks.get("documents", [])
        
        from app.services.langchain.hybrid_retriever import BM25, tokenize
        corpus = [tokenize(doc_text) for doc_text in docs_all]
        bm25 = BM25(corpus)
        query_tokens = tokenize("BM25 TF-IDF tokenization")
        scores = bm25.get_scores(query_tokens)
        
        logger.info(f"BM25 local corpus scores: {scores}")
        assert any(score > 0 for score in scores), "No keyword matches found in local corpus"
        logger.info("✓ Keyword retrieval path verified successfully.")
    except Exception as e:
        logger.error(f"❌ Keyword path test failed: {e}", exc_info=True)
        sys.exit(1)

    # 4. Test Hybrid Merging, Deduplication, and Top-K ranking
    logger.info("Step 4: Testing hybrid merging, deduplication, and RRF top-k ranking...")
    try:
        fused_docs = retriever.invoke("embeddings BM25 TF-IDF")
        logger.info(f"Fused hybrid documents retrieved (top-k=2): {[d.page_content for d in fused_docs]}")
        
        assert len(fused_docs) <= 2, f"Top-k limit exceeded. Retrieved {len(fused_docs)} documents"
        assert len(fused_docs) > 0, "No merged documents returned"
        
        chunk_ids = [d.metadata["chunk_id"] for d in fused_docs]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in ranked list!"
        
        for doc in fused_docs:
            assert doc.metadata["owner_id"] == test_owner_id, "Owner ID metadata lost"
            assert doc.metadata["document_id"] == test_doc_id, "Document ID metadata lost"
            assert "retrieval_source" in doc.metadata, "Retrieval source tracking metadata lost"
            assert doc.metadata["retrieval_source"] in ["semantic", "keyword", "hybrid"], (
                f"Invalid retrieval source tagged: {doc.metadata['retrieval_source']}"
            )
            assert "rrf_score" in doc.metadata, "RRF rank score missing"
            logger.info(
                f"Doc: '{doc.page_content[:30]}...' -> Source: {doc.metadata['retrieval_source']}, "
                f"RRF Score: {doc.metadata['rrf_score']:.4f}"
            )
        
        logger.info("✓ Hybrid rank fusion, deduplication, and metadata preservation verified successfully.")
    except Exception as e:
        logger.error(f"❌ Hybrid merging and fusion verification failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Cleanup
    logger.info("Step 5: Cleaning up test data from ChromaDB...")
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.7 HYBRID SEARCH VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS - PHASE 8.7 HYBRID SEARCH VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_7()
