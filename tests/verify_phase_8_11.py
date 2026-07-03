# tests/verify_phase_8_11.py
# --------------------------
# Verification script for Phase 8.11 (Self Query Retriever).

import os
import sys
import logging
from langchain_core.documents import Document

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.settings import settings
from app.services.langchain import get_embeddings, get_llm
from app.services.langchain.self_query import get_self_query_retriever
from app.services.self_query_service import get_self_query_service
from app.services.langchain.parent_retriever import ParentRetriever
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_11")


def verify_phase_8_11():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.11 SELF QUERY RETRIEVER VERIFICATION")
    logger.info("==================================================")

    owner_id = 999
    
    # 1. Insert documents with distinct metadata attributes
    doc_1 = Document(
        page_content="Finance revenue details for year 2024.",
        metadata={
            "owner_id": owner_id,
            "document_id": 811001,
            "filename": "finance_report_24.pdf",
            "department": "Finance",
            "year": 2024,
            "category": "report",
            "document_type": "PDF",
            "author": "Bob",
            "page_number": 1,
            "citation_key": "fin24"
        }
    )
    doc_2 = Document(
        page_content="Finance performance details for year 2022.",
        metadata={
            "owner_id": owner_id,
            "document_id": 811002,
            "filename": "finance_report_22.pdf",
            "department": "Finance",
            "year": 2022,
            "category": "report",
            "document_type": "PDF",
            "author": "Bob",
            "page_number": 1,
            "citation_key": "fin22"
        }
    )
    doc_3 = Document(
        page_content="HR employee orientation handbook details.",
        metadata={
            "owner_id": owner_id,
            "document_id": 811003,
            "filename": "hr_handbook.pdf",
            "department": "HR",
            "category": "handbook",
            "document_type": "PDF",
            "author": "Alice",
            "page_number": 2,
            "citation_key": "hr_handbook"
        }
    )
    doc_4 = Document(
        page_content="Marketing strategy details by Alice.",
        metadata={
            "owner_id": owner_id,
            "document_id": 811004,
            "filename": "marketing_guide.txt",
            "department": "Marketing",
            "category": "guide",
            "document_type": "Text",
            "author": "Alice",
            "page_number": 5,
            "citation_key": "mkt_guide"
        }
    )

    chroma_service = ChromaService()
    # Cleanup pre-existing test documents
    for doc_id in [811001, 811002, 811003, 811004]:
        try:
            chroma_service.collection.delete(where={"document_id": doc_id})
        except Exception:
            pass

    # Index documents via a base ParentRetriever to setup chunks and docstore mappings
    try:
        indexer = ParentRetriever(
            owner_id=owner_id,
            top_k=5,
            child_chunk_size=200,
            child_overlap=20,
            parent_chunk_size=1024,
            parent_overlap=100
        )
        indexer.add_documents([doc_1, doc_2, doc_3, doc_4])
        logger.info("✓ Sample documents indexed successfully.")
    except Exception as e:
        logger.error(f"❌ Document indexing failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Initialize Self Query Retriever
    try:
        service = get_self_query_service()
        retriever = service.get_self_query_retriever(
            owner_id=owner_id,
            top_k=2
        )
        logger.info("✓ Self Query Retriever initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Test Query A: "Finance reports after 2022"
    # Expected: returns Doc 1 (year 2024), NOT Doc 2 (year 2022) or Doc 3/Doc 4.
    q_a = "Finance reports after 2022"
    logger.info(f"\n--- Testing Query A: '{q_a}' ---")
    try:
        results_a = retriever.invoke(q_a)
        logger.info(f"Retrieved {len(results_a)} documents.")
        for d in results_a:
            logger.info(f"Doc: '{d.page_content}' | department: {d.metadata.get('department')} | year: {d.metadata.get('year')}")
            
        assert len(results_a) > 0, "No documents returned"
        # Check only Finance documents returned
        assert all(d.metadata.get("department") == "Finance" for d in results_a), "Returned non-Finance documents"
        # Check only year > 2022 returned
        assert all(d.metadata.get("year") > 2022 for d in results_a), "Returned documents from 2022 or earlier"
        
        logger.info("✓ Query A filters and results verified successfully.")
    except Exception as e:
        logger.error(f"❌ Query A test failed: {e}", exc_info=True)
        sys.exit(1)

    # Test Query B: "HR handbook"
    # Expected: returns Doc 3 (HR department)
    q_b = "HR handbook"
    logger.info(f"\n--- Testing Query B: '{q_b}' ---")
    try:
        results_b = retriever.invoke(q_b)
        logger.info(f"Retrieved {len(results_b)} documents.")
        for d in results_b:
            logger.info(f"Doc: '{d.page_content}' | department: {d.metadata.get('department')}")
            
        assert len(results_b) > 0, "No documents returned"
        assert any(d.metadata.get("department") == "HR" for d in results_b), "HR document not returned"
        
        logger.info("✓ Query B filters and results verified successfully.")
    except Exception as e:
        logger.error(f"❌ Query B test failed: {e}", exc_info=True)
        sys.exit(1)

    # Test Query C: "Documents by Alice"
    # Expected: returns Doc 3 and Doc 4 (author Alice)
    q_c = "Documents by Alice"
    logger.info(f"\n--- Testing Query C: '{q_c}' ---")
    try:
        results_c = retriever.invoke(q_c)
        logger.info(f"Retrieved {len(results_c)} documents.")
        for d in results_c:
            logger.info(f"Doc: '{d.page_content}' | author: {d.metadata.get('author')}")
            
        assert len(results_c) > 0, "No documents returned"
        assert all(d.metadata.get("author") == "Alice" for d in results_c), "Returned documents from non-Alice authors"
        
        logger.info("✓ Query C filters and results verified successfully.")
    except Exception as e:
        logger.error(f"❌ Query C test failed: {e}", exc_info=True)
        sys.exit(1)

    # Test Query D: Invalid metadata query (graceful fallback)
    # Expected: should not crash, falls back to normal multi-query retrieval
    q_d = "Unparseable query with invalid fields: non_existent_attr = 'val'"
    logger.info(f"\n--- Testing Query D (Invalid Metadata fallback): '{q_d}' ---")
    try:
        results_d = retriever.invoke(q_d)
        logger.info(f"Retrieved {len(results_d)} documents under fallback.")
        assert len(results_d) > 0, "Fallback returned empty document list"
        logger.info("✓ Graceful fallback verified successfully.")
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}", exc_info=True)
        sys.exit(1)

    # Verify metadata, citations, owner_id, document_id, filename preserved
    logger.info("\n--- Verifying metadata preservation ---")
    try:
        for d in results_b:
            meta = d.metadata
            assert meta["owner_id"] == owner_id, "owner_id lost"
            assert meta["filename"] == "hr_handbook.pdf", "filename lost"
            assert "citation_key" in meta, "citation_key lost"
            assert "document_id" in meta, "document_id lost"
            
        logger.info("✓ Metadata preservation verified successfully.")
    except Exception as e:
        logger.error(f"❌ Metadata verification failed: {e}", exc_info=True)
        sys.exit(1)

    # Cleanup test data
    logger.info("\nCleaning up test data from ChromaDB...")
    for doc_id in [811001, 811002, 811003, 811004]:
        try:
            chroma_service.collection.delete(where={"document_id": doc_id})
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.11 SELF QUERY RETRIEVER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.11 SELF QUERY RETRIEVER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_11()
