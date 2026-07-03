# tests/verify_phase_8_9.py
# -------------------------
# Verification script for Phase 8.9 (Contextual Compression Retriever).

import os
import sys
import logging
from langchain_core.documents import Document

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.langchain import get_embeddings, get_llm
from app.services.langchain.compression import CompressionRetriever, LLMBulkDocumentCompressor
from app.services.compression_service import get_compression_service
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_9")


def verify_phase_8_9():
    logger.info("==========================================")
    logger.info("STARTING PHASE 8.9 CONTEXTUAL COMPRESSION VERIFICATION")
    logger.info("==========================================")

    # Setup test data
    test_doc_id = 890001
    test_owner_id = 999
    
    # 5 document chunks: 2 relevant, 3 completely irrelevant
    chunks = [
        "The leave policy states that permanent workers receive 25 annual holiday allowance days.",
        "To apply for holidays, submit a request via the HR portal at least 2 weeks in advance.",
        "The sky is blue today and the sun is shining brightly.",
        "Baking cookies requires flour, sugar, butter, and chocolate chips.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})

        # Insert Doc
        vectors = embeddings.embed_documents(chunks)
        chroma_service.add_documents(
            ids=[f"chunk-89-{test_doc_id}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            metadatas=[
                {
                    "owner_id": test_owner_id,
                    "document_id": test_doc_id,
                    "filename": "holiday_rules.pdf",
                    "chunk_index": i,
                    "page_number": i + 1,
                    "citation_key": f"holiday_rules_p{i+1}"
                }
                for i in range(len(chunks))
            ],
            documents=chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    query = "How many holiday leave allowance days do workers get?"
    logger.info(f"Query: '{query}'")

    # 1. Initialize Contextual Compression Retriever
    try:
        service = get_compression_service()
        retriever_wrapper = service.get_compression_retriever(
            owner_id=test_owner_id,
            top_k=5
        )
        logger.info("✓ Contextual Compression Retriever initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Retrieve & compress documents
    try:
        logger.info("Running Contextual Compression Retriever...")
        compressed_docs = retriever_wrapper.retrieve(query)
        logger.info(f"Original documents count: {len(chunks)}")
        logger.info(f"Compressed documents count: {len(compressed_docs)}")

        # Verify irrelevant content is removed/reduced
        assert len(compressed_docs) < len(chunks), "Compression did not filter out irrelevant documents"
        logger.info("✓ Compressed output contains fewer irrelevant sections.")

        # Verify metadata preservation (page number, document_id, filename, chunk_index, citations)
        for doc in compressed_docs:
            meta = doc.metadata
            assert meta["owner_id"] == test_owner_id, "owner_id metadata lost"
            assert meta["document_id"] == test_doc_id, "document_id metadata lost"
            assert meta["filename"] == "holiday_rules.pdf", "filename metadata lost"
            assert "chunk_index" in meta, "chunk_index metadata lost"
            assert "page_number" in meta, "page_number metadata lost"
            assert "citation_key" in meta, "citation metadata lost"
            
            # Check citation formatting works
            citation = f"Source: {meta['filename']} (Page {meta['page_number']}, Chunk {meta['chunk_index']})"
            logger.info(f"Verified Citation: '{citation}'")
            logger.info(f"Compressed content: '{doc.page_content}'")

        logger.info("✓ All document metadata fields (filename, page numbers, document IDs, chunk indices, citations) preserved successfully.")
    except Exception as e:
        logger.error(f"❌ Retrieval or metadata validation failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Verify compression failure fallback
    try:
        logger.info("Testing compression failure fallback...")
        # We pass a faulty compressor that raises an exception
        class FaultyCompressor(LLMBulkDocumentCompressor):
            def compress_documents(self, documents, query, callbacks=None):
                raise RuntimeError("Simulated compression engine failure")

        faulty_compressor = FaultyCompressor(llm=get_llm())
        retriever_wrapper.compression_retriever.base_compressor = faulty_compressor

        fallback_docs = retriever_wrapper.retrieve(query)
        assert len(fallback_docs) > 0, "Fallback returned empty document list"
        logger.info(f"Fallback successfully recovered {len(fallback_docs)} documents.")
        logger.info("✓ Compression failure fallback verified successfully.")
    except Exception as e:
        logger.error(f"❌ Compression failure fallback test failed: {e}", exc_info=True)
        sys.exit(1)

    # 4. Clean up test data
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Test data cleaned up successfully from ChromaDB.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.9 CONTEXTUAL COMPRESSION VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.9 CONTEXTUAL COMPRESSION VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_9()
