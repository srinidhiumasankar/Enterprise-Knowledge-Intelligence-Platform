# tests/verify_phase_8_12.py
# --------------------------
# Verification script for Phase 8.12 (Ensemble Retriever).
# Uses mocks to verify all requirements without consuming Gemini API quota.

import os
import sys
import logging
from typing import Any, List
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.langchain.ensemble import EnsembleRetriever
from app.services.langchain.ensemble_service import EnsembleRetrieverService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.pipeline_cache import RequestCache

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_12")


class MockRetriever(BaseRetriever):
    """
    Subclass of BaseRetriever to satisfy Pydantic types, delegating to a mock.
    """
    invoke_mock: Any = Field(default=None)
    where_override: Any = Field(default=None)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.invoke_mock:
            return self.invoke_mock(query)
        return []


class SimpleAssert:
    def assertEqual(self, a, b, msg=""):
        if a != b:
            raise AssertionError(f"{msg}: {a} != {b}")
    def assertTrue(self, cond, msg=""):
        if not cond:
            raise AssertionError(msg)
    def assertIn(self, item, container, msg=""):
        if item not in container:
            raise AssertionError(f"{msg}: {item} not in {container}")

self = SimpleAssert()


def verify_phase_8_12():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.12 ENSEMBLE RETRIEVER VERIFICATION")
    logger.info("==================================================")

    # 1. Initialize Mock-Delegating Retrievers
    mock_hybrid_call = MagicMock()
    mock_mq_call = MagicMock()
    mock_parent_call = MagicMock()

    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    mock_mq = MockRetriever(invoke_mock=mock_mq_call)
    mock_parent = MockRetriever(invoke_mock=mock_parent_call)

    # Documents representing matches from different retrieval strategies
    doc_1 = Document(
        page_content="Finance revenue details for year 2024.",
        metadata={
            "chunk_id": "child-811001-0",
            "owner_id": 999,
            "document_id": 811001,
            "filename": "finance_report_24.pdf",
            "page_number": 1,
            "citation_key": "fin24",
            "score": 0.95,
            "confidence": 0.9,
        }
    )
    # Duplicate doc returned by Multi Query path
    doc_1_dup = Document(
        page_content="Finance revenue details for year 2024.",
        metadata={
            "chunk_id": "child-811001-0",
            "owner_id": 999,
            "document_id": 811001,
            "filename": "finance_report_24.pdf",
            "page_number": 1,
            "citation_key": "fin24",
            "score": 0.95,
            "confidence": 0.9,
        }
    )
    doc_2 = Document(
        page_content="HR employee handbook orientation.",
        metadata={
            "chunk_id": "child-811003-1",
            "owner_id": 999,
            "document_id": 811003,
            "filename": "hr_handbook.pdf",
            "page_number": 2,
            "citation_key": "hr_handbook",
            "score": 0.8,
            "confidence": 0.85,
        }
    )
    doc_3_parent = Document(
        page_content="Marketing strategy parent document details.",
        metadata={
            "owner_id": 999,
            "document_id": 811004,
            "filename": "marketing_guide.txt",
            "page_number": 5,
            "citation_key": "mkt_guide",
            "parent_document_id": "parent-811004",
            "score": 0.75,
            "confidence": 0.7,
        }
    )

    # Hybrid returns child doc_1 and doc_2
    mock_hybrid_call.return_value = [doc_1, doc_2]
    # Multi Query returns duplicate doc_1_dup
    mock_mq_call.return_value = [doc_1_dup]
    # Parent returns parent document doc_3_parent
    mock_parent_call.return_value = [doc_3_parent]

    # 2. Verify Ensemble initializes correctly
    logger.info("Initializing EnsembleRetriever with default weights...")
    ensemble = EnsembleRetriever(
        hybrid_retriever=mock_hybrid,
        multi_query_retriever=mock_mq,
        parent_retriever=mock_parent,
        weights={"hybrid": 0.45, "multi_query": 0.35, "parent": 0.20},
        top_k=5
    )
    logger.info("✓ EnsembleRetriever initialized correctly.")

    # 3. Verify execution & Output merging & Deduplication
    logger.info("\n--- Verifying Retrieval Execution, Merging and Deduplication ---")
    results = ensemble.invoke("Finance reports")
    
    # Assert retrievers called
    self.assertEqual(mock_hybrid_call.call_count, 1, "Hybrid retriever not called")
    self.assertEqual(mock_mq_call.call_count, 1, "Multi query retriever not called")
    self.assertEqual(mock_parent_call.call_count, 1, "Parent retriever not called")

    # Dedup check: doc_1 and doc_1_dup must be merged. Total unique should be 3
    self.assertEqual(len(results), 3, f"Duplicate removal failed, expected 3 docs but got {len(results)}")
    
    # 4. Verify RRF sorting order
    # doc_1 ranks first in Hybrid (rank 0, weight 0.45) and first in MQ (rank 0, weight 0.35)
    # doc_2 ranks second in Hybrid (rank 1, weight 0.45)
    # doc_3 ranks first in Parent (rank 0, weight 0.20)
    # RRF scores should place doc_1 first by far.
    self.assertEqual(results[0].page_content, "Finance revenue details for year 2024.", "RRF sorting did not place doc_1 first")
    self.assertTrue(results[0].metadata.get("rrf_score") > results[1].metadata.get("rrf_score"), "RRF scoring ranking mismatch")

    # 5. Verify Metadata & Citations preservation
    logger.info("\n--- Verifying Metadata and Citation Preservation ---")
    for doc in results:
        meta = doc.metadata
        self.assertTrue("citation_key" in meta, "citation_key lost")
        self.assertTrue("filename" in meta, "filename lost")
        self.assertTrue("page_number" in meta, "page_number lost")
        self.assertTrue("owner_id" in meta, "owner_id lost")
        self.assertTrue("retrieval_source" in meta, "retrieval_source lost")
        self.assertTrue("document_id" in meta, "document_id lost")
        
        # Verify parent document ID preservation specifically
        if "parent" in doc.metadata.get("retrieval_sources", []) or doc.metadata.get("retrieval_source") == "parent":
            self.assertEqual(meta["parent_document_id"], "parent-811004", "parent_document_id lost")
            
    logger.info("✓ Metadata, citations, page numbers, owner_id, sources, and document_ids preserved.")

    # 6. Verify Configurable weights
    logger.info("\n--- Verifying Configurable Weights ---")
    mock_hybrid_call.reset_mock()
    mock_mq_call.reset_mock()
    mock_parent_call.reset_mock()

    custom_ensemble = EnsembleRetriever(
        hybrid_retriever=mock_hybrid,
        multi_query_retriever=mock_mq,
        parent_retriever=mock_parent,
        weights={"hybrid": 0.10, "multi_query": 0.10, "parent": 0.80},
        top_k=5
    )
    custom_results = custom_ensemble.invoke("Finance reports")
    
    # Under parent weight of 0.80, doc_3_parent should rank higher than doc_2 (hybrid weight 0.10)
    logger.info(f"Custom weights output ranks: 1={custom_results[0].metadata['retrieval_source']}, 2={custom_results[1].metadata['retrieval_source']}")
    self.assertEqual(custom_results[0].page_content, "Marketing strategy parent document details.")
    self.assertEqual(custom_results[1].metadata["retrieval_source"], "hybrid")
    logger.info("✓ Configurable weights successfully changed rank ordering.")

    # 7. Verify Graceful handle failures (1 failure)
    logger.info("\n--- Verifying Graceful Handling of Single Retriever Failure ---")
    mock_hybrid_call.reset_mock()
    mock_mq_call.reset_mock()
    mock_parent_call.reset_mock()

    # Make MQ retriever fail
    mock_mq_call.side_effect = Exception("Multi query index timeout error")
    
    # Should not crash and successfully return from Hybrid and Parent
    failed_1_results = ensemble.invoke("Finance reports")
    self.assertEqual(len(failed_1_results), 3, "Failure of MQ retriever should not affect Hybrid and Parent")
    self.assertEqual(mock_hybrid_call.call_count, 1)
    self.assertEqual(mock_parent_call.call_count, 1)
    logger.info("✓ Graceful single failure recovery verified successfully.")

    # 8. Verify Graceful handle failures (2 failures)
    logger.info("\n--- Verifying Graceful Handling of Double Retriever Failure ---")
    mock_hybrid_call.reset_mock()
    mock_mq_call.reset_mock()
    mock_parent_call.reset_mock()

    # Make Hybrid and MQ fail
    mock_hybrid_call.side_effect = Exception("Hybrid index database locked")
    mock_mq_call.side_effect = Exception("Multi query index timeout error")
    
    failed_2_results = ensemble.invoke("Finance reports")
    # Should only return Parent document doc_3_parent
    self.assertEqual(len(failed_2_results), 1, "Expected only parent document to be returned")
    self.assertEqual(failed_2_results[0].page_content, "Marketing strategy parent document details.")
    logger.info("✓ Graceful double failure recovery verified successfully.")

    # 9. Verify EnsembleRetrieverService wrapper construction
    logger.info("\n--- Verifying EnsembleRetrieverService initialization ---")
    with patch("app.services.langchain.ensemble_service.get_hybrid_retriever") as patch_get_hybrid, \
         patch("app.services.langchain.ensemble_service.ParentRetriever") as patch_parent_wrapper, \
         patch("app.services.langchain.ensemble_service.get_multi_query_retriever") as patch_get_mq:
        
        # Configure parent wrapper mock to return a valid retriever instance
        mock_parent_wrapper_inst = MagicMock()
        mock_parent_wrapper_inst.parent_document_retriever = mock_parent
        patch_parent_wrapper.return_value = mock_parent_wrapper_inst

        # Configure hybrid and multi query mocks
        patch_get_hybrid.return_value = mock_hybrid
        patch_get_mq.return_value = mock_mq

        mock_serv_llm = MagicMock()
        service = EnsembleRetrieverService(llm=mock_serv_llm)
        self.assertEqual(service.llm, mock_serv_llm)

        service.get_ensemble_retriever(owner_id=999, top_k=3)
        
        patch_get_hybrid.assert_called_once_with(owner_id=999, document_id=None, top_k=3)
        patch_parent_wrapper.assert_called_once()
        patch_get_mq.assert_called_once()
        
        logger.info("✓ EnsembleRetrieverService components configured correctly.")

    # 10. Verify Pipeline Integration
    logger.info("\n--- Verifying RetrievalPipeline Integration ---")
    
    with patch("app.services.langchain.self_query.load_query_constructor_chain") as mock_load_query_constructor_chain, \
         patch("app.services.langchain.hybrid_retriever.get_hybrid_retriever") as patch_pipeline_hybrid, \
         patch("app.services.langchain.parent_retriever.ParentRetriever") as patch_pipeline_parent, \
         patch("app.services.langchain.compression.CompressionRetriever") as patch_pipeline_comp, \
         patch("app.services.langchain.multi_query.get_multi_query_retriever") as patch_pipeline_mq:

        mock_chain = MagicMock()
        mock_load_query_constructor_chain.return_value = mock_chain
        mock_chain.invoke.return_value = {
            "query": "Finance reports after 2022",
            "text": {
                "query": "Finance reports",
                "filter": {"comparator": "gt", "attribute": "year", "value": 2022}
            }
        }

        # Mock the retrievers to return valid BaseRetriever instances
        patch_pipeline_hybrid.return_value = mock_hybrid
        patch_parent_wrapper_inst = MagicMock()
        patch_parent_wrapper_inst.parent_document_retriever = mock_parent
        patch_pipeline_parent.return_value = patch_parent_wrapper_inst
        
        patch_comp_inst = MagicMock()
        patch_pipeline_comp.return_value = patch_comp_inst
        
        patch_pipeline_mq.return_value = mock_mq

        # Initialize pipeline
        mock_llm = MagicMock()
        pipeline = RetrievalPipeline(llm=mock_llm)
        
        # Reset invokes
        mock_hybrid_call.reset_mock()
        mock_mq_call.reset_mock()
        mock_parent_call.reset_mock()
        mock_hybrid_call.side_effect = None
        mock_mq_call.side_effect = None
        mock_parent_call.side_effect = None

        # Run retrieval
        pipeline.retrieve("Finance reports after 2022", {"owner_id": 999, "top_k": 2})

        # Assert all retrievers inside the ensemble were invoked through the pipeline
        self.assertEqual(mock_hybrid_call.call_count, 1, "Pipeline did not execute Hybrid Retriever inside ensemble")
        self.assertEqual(mock_mq_call.call_count, 1, "Pipeline did not execute Multi Query Retriever inside ensemble")
        self.assertEqual(mock_parent_call.call_count, 1, "Pipeline did not execute Parent Retriever inside ensemble")
        
        logger.info("✓ EnsembleRetriever successfully active as highest-level retriever in RetrievalPipeline.")

    # 11. Cleanup mocks
    logger.info("\nCleaning up test artifacts...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.12 ENSEMBLE RETRIEVER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.12 ENSEMBLE RETRIEVER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_12()
