# tests/verify_phase_8_13.py
# --------------------------
# Verification script for Phase 8.13 (Adaptive Retriever).
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

from app.config import settings
from app.services.langchain.adaptive import AdaptiveRetriever, classify_query
from app.services.langchain.adaptive_service import AdaptiveRetrieverService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.pipeline_cache import RequestCache

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_13")


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


def verify_phase_8_13():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.13 ADAPTIVE RETRIEVER VERIFICATION")
    logger.info("==================================================")

    # 1. Verify Query Classification Heuristics
    logger.info("\n--- Verifying Rule-based Classifier ---")
    
    # Category 1: Simple factual question
    q1_type, q1_reason = classify_query("What is the capital of France?")
    self.assertEqual(q1_type, "simple_factual")
    
    # Category 2: Metadata filtering
    q2_type, q2_reason = classify_query("documents from year 2023 after June")
    self.assertEqual(q2_type, "metadata_filtering")
    q2b_type, q2b_reason = classify_query("owner_id = 999 department finance")
    self.assertEqual(q2b_type, "metadata_filtering")

    # Category 3: Long explanatory query
    q3_type, q3_reason = classify_query("explain how the embedding pipeline chunks documents and handles duplicate metadata")
    self.assertEqual(q3_type, "long_explanatory")

    # Category 4: Ambiguous query
    q4_type, q4_reason = classify_query("something like the topic of finance concepts")
    self.assertEqual(q4_type, "ambiguous")

    # Category 5: Comparison / multi-topic query
    q5_type, q5_reason = classify_query("hybrid retriever vs parent retriever differences")
    self.assertEqual(q5_type, "comparison_multi_topic")

    logger.info("✓ All query classification heuristics verified successfully.")

    # 2. Mock Retriever Sub-components
    mock_hybrid_call = MagicMock()
    mock_sq_call = MagicMock()
    mock_parent_call = MagicMock()
    mock_mq_call = MagicMock()
    mock_ensemble_call = MagicMock()

    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    mock_sq = MockRetriever(invoke_mock=mock_sq_call)
    mock_parent = MockRetriever(invoke_mock=mock_parent_call)
    mock_mq = MockRetriever(invoke_mock=mock_mq_call)
    mock_ensemble = MockRetriever(invoke_mock=mock_ensemble_call)

    # Define return values
    doc_hybrid = Document(page_content="Hybrid match", metadata={"retrieval_source": "hybrid"})
    doc_sq = Document(page_content="Self Query match", metadata={"retrieval_source": "self_query"})
    doc_parent = Document(page_content="Parent match", metadata={"retrieval_source": "parent"})
    doc_mq = Document(page_content="Multi Query match", metadata={"retrieval_source": "multi_query"})
    doc_ensemble = Document(page_content="Ensemble match", metadata={"retrieval_source": "ensemble"})

    mock_hybrid_call.return_value = [doc_hybrid]
    mock_sq_call.return_value = [doc_sq]
    mock_parent_call.return_value = [doc_parent]
    mock_mq_call.return_value = [doc_mq]
    mock_ensemble_call.return_value = [doc_ensemble]

    # Initialize AdaptiveRetriever
    adaptive_retriever = AdaptiveRetriever(
        hybrid_retriever=mock_hybrid,
        self_query_retriever=mock_sq,
        parent_retriever=mock_parent,
        multi_query_retriever=mock_mq,
        ensemble_retriever=mock_ensemble,
        enable_adaptive=True
    )

    # 3. Verify Query Routing
    logger.info("\n--- Verifying Routing Paths ---")
    
    # Route to Hybrid
    res_hybrid = adaptive_retriever.invoke("Capital of France")
    self.assertEqual(res_hybrid[0].metadata["retrieval_source"], "hybrid")
    self.assertEqual(mock_hybrid_call.call_count, 1)

    # Route to Self Query
    res_sq = adaptive_retriever.invoke("department finance since 2024")
    self.assertEqual(res_sq[0].metadata["retrieval_source"], "self_query")
    self.assertEqual(mock_sq_call.call_count, 1)

    # Route to Parent
    res_parent = adaptive_retriever.invoke("explain how does the pipeline load documents")
    self.assertEqual(res_parent[0].metadata["retrieval_source"], "parent")
    self.assertEqual(mock_parent_call.call_count, 1)

    # Route to Multi Query
    res_mq = adaptive_retriever.invoke("concept of retrieval caching")
    self.assertEqual(res_mq[0].metadata["retrieval_source"], "multi_query")
    self.assertEqual(mock_mq_call.call_count, 1)

    # Route to Ensemble
    res_ensemble = adaptive_retriever.invoke("hybrid retriever vs parent retriever comparison")
    self.assertEqual(res_ensemble[0].metadata["retrieval_source"], "ensemble")
    self.assertEqual(mock_ensemble_call.call_count, 1)

    logger.info("✓ Routing paths successfully executed correct retrievers.")

    # 4. Verify Configuration (enabling/disabling)
    logger.info("\n--- Verifying Adaptive Configuration ---")
    adaptive_retriever.enable_adaptive = False
    
    # Direct factual query would normally route to Hybrid, but with adaptive disabled
    # it must route to Self Query (the backward compatible default)
    mock_sq_call.reset_mock()
    mock_hybrid_call.reset_mock()
    
    disabled_res = adaptive_retriever.invoke("What is the capital of France?")
    self.assertEqual(disabled_res[0].metadata["retrieval_source"], "self_query")
    self.assertEqual(mock_sq_call.call_count, 1)
    self.assertEqual(mock_hybrid_call.call_count, 0)
    
    logger.info("✓ Disabling adaptive routing defaults to Self Query correctly.")
    adaptive_retriever.enable_adaptive = True

    # 5. Verify Fallback Behavior
    logger.info("\n--- Verifying Fallback Behavior ---")
    mock_hybrid_call.reset_mock()
    mock_sq_call.reset_mock()
    mock_ensemble_call.reset_mock()

    # Simulate failure on Multi Query
    mock_mq_call.reset_mock()
    mock_mq_call.side_effect = Exception("Multi query index is down")
    
    # Adaptive routes to Multi Query, it fails -> Fallbacks to Ensemble
    fallback_res_1 = adaptive_retriever.invoke("concept of retrieval caching")
    self.assertEqual(fallback_res_1[0].metadata["retrieval_source"], "ensemble")
    self.assertEqual(mock_mq_call.call_count, 1)
    self.assertEqual(mock_ensemble_call.call_count, 1)

    # Simulate double failure (MQ and Ensemble fail)
    mock_mq_call.reset_mock()
    mock_ensemble_call.reset_mock()
    mock_mq_call.side_effect = Exception("Multi query down")
    mock_ensemble_call.side_effect = Exception("Ensemble locked")
    
    # Adaptive routes to Multi Query, it fails -> Fallbacks to Ensemble, fails -> Fallbacks to Hybrid
    fallback_res_2 = adaptive_retriever.invoke("concept of retrieval caching")
    self.assertEqual(fallback_res_2[0].metadata["retrieval_source"], "hybrid")
    self.assertEqual(mock_mq_call.call_count, 1)
    self.assertEqual(mock_ensemble_call.call_count, 1)
    self.assertEqual(mock_hybrid_call.call_count, 1)

    # Simulate total failure (MQ, Ensemble, and Hybrid fail)
    mock_mq_call.reset_mock()
    mock_ensemble_call.reset_mock()
    mock_hybrid_call.reset_mock()
    mock_mq_call.side_effect = Exception("MQ down")
    mock_ensemble_call.side_effect = Exception("Ensemble down")
    mock_hybrid_call.side_effect = Exception("Hybrid down")

    # Should log errors and return empty list without raising exception
    fallback_res_empty = adaptive_retriever.invoke("concept of retrieval caching")
    self.assertEqual(fallback_res_empty, [])
    logger.info("✓ Fallback chain (Adaptive -> Ensemble -> Hybrid -> empty list) verified successfully.")

    # 6. Verify Service Wrapper construction
    logger.info("\n--- Verifying AdaptiveRetrieverService initialization ---")
    with patch("app.services.langchain.adaptive_service.get_hybrid_retriever") as patch_get_hybrid, \
         patch("app.services.langchain.adaptive_service.ParentRetriever") as patch_parent_wrapper, \
         patch("app.services.langchain.adaptive_service.get_self_query_retriever") as patch_get_sq, \
         patch("app.services.langchain.adaptive_service.get_multi_query_retriever") as patch_get_mq, \
         patch("app.services.langchain.adaptive_service.EnsembleRetriever") as patch_ensemble:
        
        # Configure return values to pass validation
        patch_get_hybrid.return_value = mock_hybrid
        mock_p_wrapper = MagicMock(parent_document_retriever=mock_parent)
        patch_parent_wrapper.return_value = mock_p_wrapper
        patch_get_sq.return_value = mock_sq
        patch_get_mq.return_value = mock_mq
        patch_ensemble.return_value = mock_ensemble

        service = AdaptiveRetrieverService(llm=MagicMock())
        ret = service.get_adaptive_retriever(owner_id=999, top_k=3)
        self.assertEqual(ret.enable_adaptive, True)
        self.assertEqual(ret.hybrid_retriever, mock_hybrid)
        
        logger.info("✓ AdaptiveRetrieverService components configured correctly.")

    # 7. Verify Pipeline Integration
    logger.info("\n--- Verifying RetrievalPipeline Integration ---")
    with patch("app.services.langchain.adaptive_service.get_hybrid_retriever") as p_hybrid, \
         patch("app.services.langchain.adaptive_service.ParentRetriever") as p_parent, \
         patch("app.services.langchain.adaptive_service.get_self_query_retriever") as p_sq, \
         patch("app.services.langchain.adaptive_service.get_multi_query_retriever") as p_mq, \
         patch("app.services.langchain.adaptive_service.EnsembleRetriever") as p_ens:
        
        p_hybrid.return_value = mock_hybrid
        p_parent.return_value = MagicMock(parent_document_retriever=mock_parent)
        p_sq.return_value = mock_sq
        p_mq.return_value = mock_mq
        p_ens.return_value = mock_ensemble

        # Reset invokes
        mock_hybrid_call.reset_mock()
        mock_hybrid_call.side_effect = None

        pipeline = RetrievalPipeline(llm=MagicMock())
        pipeline.retrieve("What is the capital of France?", {"owner_id": 999, "top_k": 2})

        # Simple query should route directly to Hybrid inside RetrievalPipeline
        self.assertEqual(mock_hybrid_call.call_count, 1)
        logger.info("✓ AdaptiveRetriever successfully integrated into RetrievalPipeline.")

    # 8. Cleanup
    logger.info("\nCleaning up test artifacts...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.13 ADAPTIVE RETRIEVER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.13 ADAPTIVE RETRIEVER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_13()
