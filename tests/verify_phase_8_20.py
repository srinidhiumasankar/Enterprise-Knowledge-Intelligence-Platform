# tests/verify_phase_8_20.py
# --------------------------
# Verification script for Phase 8.20 (Observability, Health Monitoring & Diagnostics).
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

from app.services.langchain.health_monitor import HealthMonitor, HealthReport
from app.services.langchain.health_service import HealthService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.adaptive import AdaptiveRetriever
from app.services.langchain.retrieval_analytics import RetrievalAnalytics

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_20")


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


def verify_phase_8_20():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.20 HEALTH MONITOR VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Health Monitor Initialization ---")
    service = HealthService()
    monitor = service.get_monitor()
    monitor.reset()
    
    self.assertEqual(len(monitor.reports), 0)
    logger.info("✓ HealthService and HealthMonitor initialized correctly.")

    # 2. Component monitoring
    logger.info("\n--- Verifying Component Monitoring ---")
    monitor.start_monitoring(request_id="req_test_123")
    
    monitor.record_component("hybrid", "HEALTHY", latency=120.0)
    monitor.record_component("parent", "HEALTHY", latency=80.0)
    
    state = monitor._get_or_create_state()
    self.assertEqual(state["components"]["hybrid"]["status"], "HEALTHY")
    self.assertEqual(state["components"]["hybrid"]["latency"], 120.0)
    logger.info("✓ Component status and latency recorded correctly.")

    # 3. Warning detection (Slow Component)
    logger.info("\n--- Verifying Warning Detection ---")
    # Record a slow component (exceeds default 500ms warning threshold)
    monitor.record_component("self_query", "HEALTHY", latency=650.0)
    self.assertEqual(state["components"]["self_query"]["status"], "WARNING")
    self.assertEqual(len(state["warnings"]), 1)
    self.assertTrue("Slow Component" in state["warnings"][0]["message"])
    logger.info("✓ Slow component warning threshold triggers recorded correctly.")

    # 4. Failure detection
    logger.info("\n--- Verifying Failure Detection ---")
    monitor.record_failure("metadata_ranker", "Database connection timeout")
    self.assertEqual(state["components"]["metadata_ranker"]["status"], "FAILED")
    self.assertEqual(len(state["errors"]), 1)
    logger.info("✓ Component failures recorded correctly.")

    # 5. Diagnostics generation
    logger.info("\n--- Verifying Diagnostics Generation ---")
    report = monitor.build_health_report()
    self.assertEqual(report.request_id, "req_test_123")
    self.assertEqual(report.error_count, 1)
    self.assertEqual(report.warning_count, 1)
    self.assertEqual(report.overall_status, "DEGRADED")
    self.assertTrue(any("Slow Component" in d for d in report.diagnostics_summary))
    self.assertTrue(any("Pipeline Failure" in d for d in report.diagnostics_summary))
    logger.info("✓ Diagnostic audits and overall status classifications verified successfully.")

    # 6. Pipeline integration
    logger.info("\n--- Verifying Pipeline Integration ---")
    monitor.reset()
    
    mock_hybrid_call = MagicMock()
    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    
    doc_x = Document(page_content="Text chunk A", metadata={"document_id": "doc_x", "similarity_score": 0.80})
    mock_hybrid_call.return_value = [doc_x]

    with patch("app.services.langchain.adaptive_service.AdaptiveRetrieverService.get_adaptive_retriever") as mock_get_adaptive:
        from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
        analytics_service = RetrievalAnalyticsService()
        
        adaptive_retriever = AdaptiveRetriever(
            hybrid_retriever=mock_hybrid,
            self_query_retriever=mock_hybrid,
            parent_retriever=mock_hybrid,
            multi_query_retriever=mock_hybrid,
            ensemble_retriever=mock_hybrid,
            enable_adaptive=True
        )
        # Wrap it
        monitored_adaptive = analytics_service.wrap_retriever(adaptive_retriever, "adaptive")
        mock_get_adaptive.return_value = monitored_adaptive

        from app.config import settings
        with patch.object(settings, "HEALTH_WARNING_LATENCY_MS", 50000.0):
            pipeline = RetrievalPipeline()
            
            # Run pipeline
            pipeline_results = pipeline.retrieve("Test health", {"owner_id": 999, "session_id": "session_999"})
        
        # Verify that health reports were saved
        self.assertEqual(len(monitor.reports), 1)
        logged_report = monitor.reports[0]
        self.assertEqual(logged_report.overall_status, "HEALTHY")
        self.assertEqual(logged_report.pipeline_status, "HEALTHY")
        self.assertEqual(logged_report.conversation_memory_status, "HEALTHY")
        self.assertEqual(logged_report.metadata_ranker_status, "HEALTHY")
        self.assertEqual(logged_report.result_scorer_status, "HEALTHY")
        self.assertEqual(logged_report.adaptive_status, "HEALTHY")
        
    logger.info("✓ RetrievalPipeline health monitoring integration verified successfully.")

    # 7. Cleanup
    logger.info("\nCleaning up test resources...")
    monitor.reset()
    RetrievalAnalytics.get_instance().reset()
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.20 HEALTH MONITOR VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 8.20 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.20 HEALTH MONITOR VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 8.20 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_20()
