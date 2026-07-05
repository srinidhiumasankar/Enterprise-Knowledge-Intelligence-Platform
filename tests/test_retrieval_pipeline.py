# tests/test_retrieval_pipeline.py
# --------------------------------
# Verification tests for RetrievalPipeline orchestration, caching, and diagnostics.

import os
import sys
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.structured_query import Comparison, Comparator, StructuredQuery

from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.pipeline_cache import RequestCache, CachingLLM, get_current_request_cache


from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

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


class TestRetrievalPipeline(unittest.TestCase):

    def test_request_cache_lifecycle(self):
        # 1. Test context manager activates and resets ContextVar
        self.assertIsNone(get_current_request_cache())
        with RequestCache() as cache:
            self.assertEqual(get_current_request_cache(), cache)
            self.assertEqual(cache.llm_call_count, 0)
            self.assertEqual(cache.cache_hits, 0)
            self.assertEqual(cache.cache_misses, 0)

            # Test increment and logging
            cache.increment_llm_calls()
            self.assertEqual(cache.llm_call_count, 1)

            cache.log_hit("Test hit")
            self.assertEqual(cache.cache_hits, 1)

            cache.log_miss("Test miss")
            self.assertEqual(cache.cache_misses, 1)

        self.assertIsNone(get_current_request_cache())

    def test_caching_llm_proxy(self):
        # Test CachingLLM forwards calls and caches results
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Mocked LLM Response"

        proxy = CachingLLM(mock_llm)

        # Calling without cache context shouldn't cache anything and should invoke mock_llm directly
        res1 = proxy.invoke("Hello")
        self.assertEqual(res1, "Mocked LLM Response")
        self.assertEqual(mock_llm.invoke.call_count, 1)

        mock_llm.invoke.reset_mock()

        # Calling within cache context
        with RequestCache() as cache:
            # First invoke (miss)
            res2 = proxy.invoke("Hello")
            self.assertEqual(res2, "Mocked LLM Response")
            self.assertEqual(mock_llm.invoke.call_count, 1)
            self.assertEqual(cache.llm_call_count, 1)
            self.assertEqual(cache.cache_hits, 0)
            self.assertEqual(cache.cache_misses, 1)

            # Second invoke with same args (hit)
            res3 = proxy.invoke("Hello")
            self.assertEqual(res3, "Mocked LLM Response")
            # Underlying invoke count should NOT increase
            self.assertEqual(mock_llm.invoke.call_count, 1)
            self.assertEqual(cache.llm_call_count, 1)
            self.assertEqual(cache.cache_hits, 1)
            self.assertEqual(cache.cache_misses, 1)

            # Third invoke with different args (miss)
            mock_llm.invoke.return_value = "Different Response"
            res4 = proxy.invoke("Goodbye")
            self.assertEqual(res4, "Different Response")
            self.assertEqual(mock_llm.invoke.call_count, 2)
            self.assertEqual(cache.llm_call_count, 2)
            self.assertEqual(cache.cache_hits, 1)
            self.assertEqual(cache.cache_misses, 2)

    @patch("app.services.langchain.self_query.load_query_constructor_chain")
    @patch("app.services.langchain.adaptive_service.get_hybrid_retriever")
    @patch("app.services.langchain.adaptive_service.ParentRetriever")
    @patch("app.services.langchain.adaptive_service.get_multi_query_retriever")
    @patch("app.services.langchain.hybrid_retriever.get_hybrid_retriever")
    @patch("app.services.langchain.parent_retriever.ParentRetriever")
    @patch("app.services.langchain.multi_query.get_multi_query_retriever")
    def test_pipeline_orchestration_and_cache(
        self,
        mock_multi_query_mq,
        mock_parent_retriever,
        mock_get_hybrid_retriever,
        mock_adaptive_mq,
        mock_adaptive_parent,
        mock_adaptive_hybrid,
        mock_load_query_constructor_chain
    ):
        # Setup mocks
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_load_query_constructor_chain.return_value = mock_chain

        # Structured query parsing mock
        mock_chain.invoke.return_value = {
            "query": "Finance reports after 2022",
            "text": {
                "query": "Finance reports",
                "filter": {"comparator": "gt", "attribute": "year", "value": 2022}
            }
        }

        # Setup sub-retrievers and mock outputs using MockRetriever class
        mock_hybrid_call = MagicMock()
        mock_mq_call = MagicMock()
        mock_parent_call = MagicMock()

        mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
        mock_mq = MockRetriever(invoke_mock=mock_mq_call)
        mock_parent = MockRetriever(invoke_mock=mock_parent_call)

        # Mock the constructors in adaptive service
        mock_adaptive_hybrid.return_value = mock_hybrid
        mock_adaptive_parent_wrapper = MagicMock()
        mock_adaptive_parent_wrapper.parent_document_retriever = mock_parent
        mock_adaptive_parent.return_value = mock_adaptive_parent_wrapper
        mock_adaptive_mq.return_value = mock_mq

        # Mock the constructors for method-level imports
        mock_get_hybrid_retriever.return_value = mock_hybrid
        mock_parent_wrapper_inst = MagicMock()
        mock_parent_wrapper_inst.parent_document_retriever = mock_parent
        mock_parent_retriever.return_value = mock_parent_wrapper_inst
        mock_multi_query_mq.return_value = mock_mq

        # Mock final return docs with metadata and citation preserved
        expected_docs = [
            Document(
                page_content="Finance revenue details for year 2024.",
                metadata={
                    "owner_id": 999,
                    "document_id": 811001,
                    "filename": "finance_report_24.pdf",
                    "citation_key": "fin24"
                }
            )
        ]
        mock_mq_call.return_value = expected_docs

        pipeline = RetrievalPipeline(llm=mock_llm)
        user_context = {"owner_id": 999, "top_k": 2}

        # Patch ENABLE_CONVERSATION_MEMORY to False so that history side effects
        # do not mutate document metadata and cause comparison assertion failures.
        from app.config import settings
        with patch.object(settings, "ENABLE_CONVERSATION_MEMORY", False):
            # 1. Run pipeline (first request)
            results1 = pipeline.retrieve("Finance reports after 2022", user_context)
            
            # Verify result content, metadata, and citation key
            self.assertEqual(len(results1), 1)
            self.assertEqual(results1[0].page_content, "Finance revenue details for year 2024.")
            self.assertEqual(results1[0].metadata["citation_key"], "fin24")
            self.assertEqual(results1[0].metadata["owner_id"], 999)
            self.assertEqual(mock_chain.invoke.call_count, 1)

            # 2. Run pipeline again with the exact same inputs (but separate requests - separate Cache context)
            # Should invoke LLM again because each pipeline.retrieve has its own request context.
            results2 = pipeline.retrieve("Finance reports after 2022", user_context)
            self.assertEqual(results2, results1)
            self.assertEqual(mock_chain.invoke.call_count, 2)

    def test_retrieval_analytics(self):
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
        from app.config import settings

        analytics_service = RetrievalAnalyticsService()
        analytics = analytics_service.get_analytics()
        analytics.reset()

        # 1. Analytics Enabled
        with patch.object(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            rid = analytics.start_request(
                owner_id=456,
                session_id="session_test",
                original_query="Test query"
            )
            self.assertTrue(len(rid) > 0)
            
            # 2. Retriever tracking & Latency Recording
            analytics.record_retriever(
                retriever_name="hybrid",
                executed=True,
                skipped=False,
                latency=15.0,
                returned_document_count=3,
                success=True,
                failure=False
            )
            
            # 3. Cache Tracking
            analytics.record_cache(enabled=True, hit=True, miss=False, lookup_latency=0.5)
            
            # 4. Ranking Metrics
            analytics.record_documents(before=10, after=5, duplicates=5)
            analytics.record_metadata(metadata_applied=True, threshold_applied=True, final_count=5)
            
            # 5. Summary Generation
            summary = analytics.build_summary()
            self.assertIn("Request ID:", summary)
            
            # 6. Request completion
            metrics = analytics.finish_request(
                rewritten_query="Rewritten query",
                final_query="Final query",
                docs_before=10,
                docs_after=5
            )
            self.assertEqual(len(analytics.history), 1)
            self.assertEqual(metrics.owner_id, 456)
            self.assertEqual(metrics.cache_hit, True)
            self.assertEqual(metrics.retrievers["hybrid"].executed, True)
            self.assertEqual(metrics.duplicate_removal_count, 5)

        # 7. Analytics Disabled
        analytics.reset()
        with patch.object(settings, "ENABLE_RETRIEVAL_ANALYTICS", False):
            rid = analytics.start_request(
                owner_id=456,
                session_id="session_test",
                original_query="Test query"
            )
            self.assertEqual(rid, "")
            
            # If disabled, finish_request returns None
            res = analytics.finish_request()
            self.assertEqual(res, None)
            self.assertEqual(len(analytics.history), 0)

        # 8. Multiple Requests & Reset Functionality
        analytics.reset()
        with patch.object(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            analytics.start_request(original_query="Req 1")
            analytics.finish_request()
            
            analytics.start_request(original_query="Req 2")
            analytics.finish_request()
            
            self.assertEqual(len(analytics.history), 2)
            
            # Reset
            analytics.reset()
            self.assertEqual(len(analytics.history), 0)

    def test_health_monitoring(self):
        from app.services.langchain.health_monitor import HealthMonitor, HealthReport
        from app.services.langchain.health_service import HealthService
        from app.config import settings

        service = HealthService()
        monitor = service.get_monitor()
        monitor.reset()

        # 1. Health Monitor initialization
        self.assertEqual(len(monitor.reports), 0)

        # 2. Component monitoring
        monitor.start_monitoring(request_id="test_req_820")
        monitor.record_component("hybrid", "HEALTHY", latency=10.0)
        state = monitor._get_or_create_state()
        self.assertEqual(state["components"]["hybrid"]["status"], "HEALTHY")

        # 3. Warning detection
        monitor.record_component("parent", "HEALTHY", latency=900.0) # > 500ms
        self.assertEqual(state["components"]["parent"]["status"], "WARNING")
        self.assertEqual(len(state["warnings"]), 1)

        # 4. Failure detection & Diagnostics generation
        monitor.record_failure("metadata_ranker", "Ranker timeout error")
        self.assertEqual(state["components"]["metadata_ranker"]["status"], "FAILED")

        # 5. Health report generation & Diagnostics summary
        report = monitor.build_health_report()
        self.assertEqual(report.request_id, "test_req_820")
        self.assertEqual(report.overall_status, "DEGRADED")
        self.assertTrue(any("Slow Component" in d for d in report.diagnostics_summary))
        self.assertTrue(any("Ranker timeout error" in d for d in report.diagnostics_summary))

        # 6. Reset functionality
        monitor.finish_monitoring()
        self.assertEqual(len(monitor.reports), 1)
        monitor.reset()
        self.assertEqual(len(monitor.reports), 0)

        # 7. Multiple request handling
        monitor.start_monitoring(request_id="req1")
        monitor.finish_monitoring()
        monitor.start_monitoring(request_id="req2")
        monitor.finish_monitoring()
        self.assertEqual(len(monitor.reports), 2)
        monitor.reset()


if __name__ == "__main__":
    unittest.main()
