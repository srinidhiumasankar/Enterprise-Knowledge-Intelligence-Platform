# tests/verify_phase_8_10.py
# --------------------------
# Verification script for Phase 8.10 (Parent Document Retriever).

import os
import sys
import logging
from langchain_core.documents import Document

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.langchain import get_embeddings, get_llm
from app.services.langchain.parent_retriever import ParentRetriever
from app.services.parent_retrieval_service import get_parent_retrieval_service
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_10")


def verify_phase_8_10():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.10 PARENT DOCUMENT RETRIEVER VERIFICATION")
    logger.info("==================================================")

    test_doc_id = 810001
    test_owner_id = 999

    # 1. Sample Parent Document text
    # A single large parent document (will be split into multiple child chunks)
    parent_text = (
        "This is Section 1 of the Employee leave policy. Permanent workers receive 25 annual holiday allowance leave days. "
        "All requests must be submitted through the portal at least two weeks before the planned leave start date. "
        "Any requests submitted later may be denied. "
        "This is Section 2 of the policy. Sick leaves are managed separately. Employees receive 15 sick leave days per year, "
        "and doctor notes are required for absences extending past three consecutive business days."
    )
    
    parent_doc = Document(
        page_content=parent_text,
        metadata={
            "owner_id": test_owner_id,
            "document_id": test_doc_id,
            "filename": "leave_handbook.pdf",
            "page_number": 3,
            "citation_key": "leave_handbook_p3"
        }
    )

    chroma_service = ChromaService()
    try:
        # Cleanup pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

    # 2. Initialize Parent Retriever service
    try:
        service = get_parent_retrieval_service()
        # Initialize parent retriever with parent size 1024, child size 200
        retriever_wrapper = service.get_parent_retriever(
            owner_id=test_owner_id,
            document_id=test_doc_id,
            top_k=2,
            child_chunk_size=200,
            child_overlap=20,
            parent_chunk_size=1024,
            parent_overlap=100
        )
        logger.info("✓ Parent Retriever initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Add parent documents (splits into child chunks automatically and indexes them)
    try:
        logger.info("Indexing parent document and creating child chunks...")
        retriever_wrapper.add_documents([parent_doc])
        logger.info("✓ Parent document split and child chunks indexed successfully.")
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}", exc_info=True)
        sys.exit(1)

    # 4. Run retrieval and verify parent is returned
    query = "How many sick leave days do employees get?"
    logger.info(f"Retrieving for query: '{query}'")
    try:
        retrieved_docs = retriever_wrapper.retrieve(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        
        assert len(retrieved_docs) > 0, "No documents retrieved!"
        
        # Verify parent document chunk is returned (length should be larger than child chunk size of 200)
        # It should contain the full parent text (length around 400+)
        first_doc = retrieved_docs[0]
        logger.info(f"Retrieved document page content length: {len(first_doc.page_content)}")
        assert len(first_doc.page_content) > 300, f"Expected parent document, got child chunk length {len(first_doc.page_content)}"
        assert "Sick leaves are managed separately" in first_doc.page_content, "Retrieved parent document does not match query"
        logger.info("✓ Correct parent document returned successfully.")

        # Verify metadata, filename, page_number, document_id, parent_document_id, citations preserved
        meta = first_doc.metadata
        logger.info(f"Preserved Metadata: {meta}")
        assert meta["owner_id"] == test_owner_id, "owner_id metadata lost"
        assert meta["document_id"] == test_doc_id, "document_id metadata lost"
        assert meta["filename"] == "leave_handbook.pdf", "filename metadata lost"
        assert meta["page_number"] == 3, "page_number metadata lost"
        assert "parent_document_id" in meta, "parent_document_id missing"
        assert "citation_key" in meta, "citation metadata lost"
        
        # Verify child chunk text details are preserved
        assert "child_chunk_text" in meta, "child_chunk_text details lost"
        logger.info(f"Preserved child chunk snippet: '{meta['child_chunk_text'][:50]}...'")
        logger.info("✓ Filename, page number, document_id, parent_document_id, citations, and metadata verified successfully.")
    except Exception as e:
        logger.error(f"❌ Retrieval or metadata validation failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Verify fallback if parent mapping missing
    try:
        logger.info("Testing fallback mechanism when parent document mapping is missing...")
        # Corrupt/clear docstore mapping to test fallback
        retriever_wrapper.docstore.mdelete(list(retriever_wrapper.docstore.yield_keys()))
        
        fallback_docs = retriever_wrapper.retrieve(query)
        assert len(fallback_docs) > 0, "Fallback returned empty document list"
        
        # It should fall back to child chunks from hybrid search (length <= 250)
        logger.info(f"Fallback doc length: {len(fallback_docs[0].page_content)}")
        assert len(fallback_docs[0].page_content) <= 250, "Expected child chunk fallback, but got parent chunk length"
        logger.info("✓ Missing mapping fallback verified successfully.")
    except Exception as e:
        logger.error(f"❌ Fallback verification failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Cleanup
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Chroma test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up Chroma test data: {e}")

    logger.info("==================================================")
    logger.info("PASS - PHASE 8.10 PARENT DOCUMENT RETRIEVER VERIFIED SUCCESSFULLY")
    logger.info("==================================================")
    print("PASS - PHASE 8.10 PARENT DOCUMENT RETRIEVER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_10()
