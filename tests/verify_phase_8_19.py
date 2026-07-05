# tests/verify_phase_8_19.py
# --------------------------
# Verification script for Phase 8.19 (Retrieval Analytics & Performance Monitoring).
# Uses mocks to verify all requirements without consuming Gemini API quota.

import os
import sys
import logging
import time
from typing import Any, List
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.langchain.retrieval_analytics import RetrievalAnalytics, MonitoredRetriever
from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.adaptive import AdaptiveRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_19")


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


def verify_phase_8_19():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.19 RETRIEVAL ANALYTICS VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Analytics Initialization ---")
    service = RetrievalAnalyticsService()
    analytics = service.get_analytics()
    analytics.reset()
    
    self.assertEqual(len(analytics.history), 0)
    logger.info("✓ RetrievalAnalyticsService and RetrievalAnalytics initialized correctly.")

    # 2. Query tracking
    logger.info("\n--- Verifying Query Tracking ---")
    rid = analytics.start_request(
        owner_id=123,
        session_id="session_abc",
        original_query="Test original query"
    )
    self.assertTrue(len(rid) > 0)
    
    analytics.record_query("Test original query", "Test rewritten query", "Test final query")
    metrics = analytics._get_or_create_metrics()
    self.assertEqual(metrics.original_query, "Test original query")
    self.assertEqual(metrics.rewritten_query, "Test rewritten query")
    self.assertEqual(metrics.final_query, "Test final query")
    logger.info("✓ Query details registered correctly.")

    # 3. Retriever tracking
    logger.info("\n--- Verifying Retriever Tracking ---")
    # Record retriever run
    analytics.record_retriever(
        retriever_name="hybrid",
        executed=True,
        skipped=False,
        latency=12.5,
        returned_document_count=4,
        success=True,
        failure=False
    )
    self.assertEqual(metrics.retrievers["hybrid"].executed, True)
    self.assertEqual(metrics.retrievers["hybrid"].latency, 12.5)
    self.assertEqual(metrics.retrievers["hybrid"].returned_document_count, 4)
    self.assertEqual(metrics.retrievers["hybrid"].success, True)
    logger.info("✓ Retriever operations recorded correctly.")

    # 4. Cache tracking
    logger.info("\n--- Verifying Cache Tracking ---")
    analytics.record_cache(
        enabled=True,
        hit=True,
        miss=False,
        lookup_latency=1.2
    )
    self.assertEqual(metrics.cache_enabled, True)
    self.assertEqual(metrics.cache_hit, True)
    self.assertEqual(metrics.cache_lookup_latency, 1.2)
    logger.info("✓ Cache events and lookup latency recorded correctly.")

    # 5. Ranking tracking
    logger.info("\n--- Verifying Ranking Metrics ---")
    analytics.record_documents(before=10, after=5, duplicates=5)
    analytics.record_metadata(metadata_applied=True, threshold_applied=True, final_count=5)
    
    self.assertEqual(metrics.docs_before_ranking, 10)
    self.assertEqual(metrics.docs_after_ranking, 5)
    self.assertEqual(metrics.duplicate_removal_count, 5)
    self.assertEqual(metrics.metadata_ranking_applied, True)
    self.assertEqual(metrics.confidence_threshold_applied, True)
    self.assertEqual(metrics.final_ranked_document_count, 5)
    logger.info("✓ Document deduplication and metadata ranking events recorded correctly.")

    # 6. Latency tracking
    logger.info("\n--- Verifying Latency Recording ---")
    analytics.record_latency("metadata_ranker_latency", 4.2)
    analytics.record_latency("result_scorer_latency", 2.1)
    analytics.record_latency("llm_latency", 150.0)
    analytics.record_latency("embedding_latency", 45.0)

    self.assertEqual(metrics.metadata_ranker_latency, 4.2)
    self.assertEqual(metrics.result_scorer_latency, 2.1)
    self.assertEqual(metrics.total_llm_latency, 150.0)
    self.assertEqual(metrics.embedding_latency, 45.0)
    logger.info("✓ Telemetry latencies registered correctly.")

    # 7. Summary generation
    logger.info("\n--- Verifying Summary Generation ---")
    summary = analytics.build_summary()
    self.assertIn("Request ID:", summary)
    self.assertIn("Original Query:", summary)
    self.assertIn("Total Latency:", summary)
    logger.info("✓ Presentation summaries formatted correctly.")

    # 8. Request completion
    logger.info("\n--- Verifying Request Completion ---")
    analytics.finish_request(
        rewritten_query="Final rewritten",
        final_query="Final query",
        docs_before=10,
        docs_after=5
    )
    self.assertEqual(len(analytics.history), 1)
    # Check singleton thread-local cleared
    self.assertTrue(not hasattr(analytics._local, "current_metrics") or analytics._local.current_metrics is None)
    logger.info("✓ Request finished and thread-local scoped state cleaned successfully.")

    # 9. Reset Functionality
    logger.info("\n--- Verifying Reset Functionality ---")
    analytics.reset()
    self.assertEqual(len(analytics.history), 0)
    logger.info("✓ Analytics memory resets successfully.")

    # 10. RetrievalPipeline Integration
    logger.info("\n--- Verifying RetrievalPipeline Integration ---")
    mock_hybrid_call = MagicMock()
    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    
    doc_x = Document(page_content="Text chunk A", metadata={"document_id": "doc_x", "similarity_score": 0.80})
    doc_y = Document(page_content="Text chunk B", metadata={"document_id": "doc_y", "similarity_score": 0.40})
    mock_hybrid_call.return_value = [doc_x, doc_y]

    with patch("app.services.langchain.adaptive_service.AdaptiveRetrieverService.get_adaptive_retriever") as mock_get_adaptive:
        adaptive_retriever = AdaptiveRetriever(
            hybrid_retriever=mock_hybrid,
            self_query_retriever=mock_hybrid,
            parent_retriever=mock_hybrid,
            multi_query_retriever=mock_hybrid,
            ensemble_retriever=mock_hybrid,
            enable_adaptive=True
        )
        # Wrap it like in adaptive service
        monitored_adaptive = service.wrap_retriever(adaptive_retriever, "adaptive")
        mock_get_adaptive.return_value = monitored_adaptive

        pipeline = RetrievalPipeline()
        
        # Run retrieval pipeline
        pipeline_results = pipeline.retrieve("Test analytics", {"owner_id": 999, "session_id": "session_999"})
        
        # Verify telemetry log entries
        self.assertEqual(len(analytics.history), 1)
        logged_req = analytics.history[0]
        self.assertEqual(logged_req.owner_id, 999)
        self.assertEqual(logged_req.session_id, "session_999")
        self.assertEqual(logged_req.original_query, "Test analytics")
        self.assertTrue(logged_req.total_latency > 0.0)
        self.assertTrue(logged_req.retrievers["adaptive"].executed)
        self.assertEqual(logged_req.retrievers["adaptive"].returned_document_count, 2)
        
    logger.info("✓ RetrievalPipeline telemetry monitoring integration verified successfully.")

    # 11. Cleanup
    logger.info("\nCleaning up test resources...")
    analytics.reset()
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.19 RETRIEVAL ANALYTICS VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 8.19 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.19 RETRIEVAL ANALYTICS VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 8.19 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_19()
