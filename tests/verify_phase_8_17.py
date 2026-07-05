# tests/verify_phase_8_17.py
# --------------------------
# Verification script for Phase 8.17 (Intelligent Result Scoring).
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

from app.services.langchain.result_scorer import ResultScorer
from app.services.langchain.result_scorer_service import ResultScorerService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.adaptive import AdaptiveRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_17")


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


def verify_phase_8_17():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.17 RESULT SCORER VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Scorer Initialization ---")
    service = ResultScorerService()
    scorer = service.get_result_scorer()
    self.assertEqual(scorer.enable_scorer, True)
    self.assertTrue("semantic" in scorer.weights)
    logger.info("✓ ResultScorerService and ResultScorer initialized correctly.")

    # 2. Retriever agreement
    logger.info("\n--- Verifying Retriever Agreement ---")
    doc_agreement = Document(page_content="Text content", metadata={"retrieval_sources": ["hybrid", "parent"]})
    doc_no_agreement = Document(page_content="Text content", metadata={"retrieval_source": "hybrid"})
    
    self.assertEqual(scorer.score_agreement(doc_agreement), 1.0)
    self.assertEqual(scorer.score_agreement(doc_no_agreement), 0.0)
    logger.info("✓ Retriever agreement checks verified successfully.")

    # 3. Chunk completeness
    logger.info("\n--- Verifying Chunk Completeness ---")
    doc_long = Document(page_content="A" * 1200)
    doc_med = Document(page_content="A" * 600)
    doc_short = Document(page_content="A" * 300)
    doc_vshort = Document(page_content="A" * 100)

    self.assertEqual(scorer.score_chunk_completeness(doc_long), 1.0)
    self.assertEqual(scorer.score_chunk_completeness(doc_med), 0.8)
    self.assertEqual(scorer.score_chunk_completeness(doc_short), 0.5)
    self.assertEqual(scorer.score_chunk_completeness(doc_vshort), 0.1)
    logger.info("✓ Chunk completeness penalization & reward verified successfully.")

    # 4. Citation completeness
    logger.info("\n--- Verifying Citation Completeness ---")
    doc_cit_all = Document(page_content="Txt", metadata={"filename": "doc.pdf", "page_number": 1, "citation_key": "c1"})
    doc_cit_part = Document(page_content="Txt", metadata={"filename": "doc.pdf"})
    doc_cit_none = Document(page_content="Txt", metadata={})

    self.assertEqual(scorer.score_citation_completeness(doc_cit_all), 1.0)
    self.assertEqual(scorer.score_citation_completeness(doc_cit_part), 1.0 / 3.0)
    self.assertEqual(scorer.score_citation_completeness(doc_cit_none), 0.0)
    logger.info("✓ Citation completeness checks verified successfully.")

    # 5. History boost
    logger.info("\n--- Verifying History Topic Boost ---")
    history = [{"user_query": "Explain Machine Learning concepts.", "rewritten_query": "Explain Machine Learning concepts."}]
    doc_ml = Document(page_content="This document covers machine learning applications and algorithms.")
    doc_other = Document(page_content="Weather is beautiful today in Paris.")

    self.assertTrue(scorer.score_history_boost(doc_ml, history) > 0.0)
    self.assertEqual(scorer.score_history_boost(doc_other, history), 0.0)
    logger.info("✓ Conversational history boosts verified successfully.")

    # 6. Rewrite boost
    logger.info("\n--- Verifying Rewrite Boost ---")
    doc_rewritten = Document(page_content="Txt", metadata={"original_query": "ml", "rewritten_query": "Machine Learning"})
    doc_no_rewrite = Document(page_content="Txt", metadata={"original_query": "ml", "rewritten_query": "ml"})

    self.assertEqual(scorer.score_rewrite_boost(doc_rewritten), 1.0)
    self.assertEqual(scorer.score_rewrite_boost(doc_no_rewrite), 0.0)
    logger.info("✓ Query rewriting boosts verified successfully.")

    # 7. Confidence levels
    logger.info("\n--- Verifying Confidence Levels mapping ---")
    self.assertEqual(scorer.get_confidence_level(0.90), "Very High")
    self.assertEqual(scorer.get_confidence_level(0.75), "High")
    self.assertEqual(scorer.get_confidence_level(0.55), "Medium")
    self.assertEqual(scorer.get_confidence_level(0.35), "Low")
    self.assertEqual(scorer.get_confidence_level(0.15), "Very Low")
    logger.info("✓ Confidence level categorizations verified successfully.")

    # 8. Sorting and filtering
    logger.info("\n--- Verifying Scorer Ranking and Filtering ---")
    doc_a = Document(page_content="Short", metadata={"document_id": "doc_a", "semantic_score": 0.30, "metadata_score": 0.30})
    doc_b = Document(page_content="A" * 1100, metadata={"document_id": "doc_b", "semantic_score": 0.80, "metadata_score": 0.85})
    
    # 8.1 Sorting order
    results = scorer.score_documents([doc_a, doc_b])
    self.assertEqual(results[0].metadata["document_id"], "doc_b", "Best confidence doc should rank first")
    
    # 8.2 Minimum confidence threshold filtering
    strict_scorer = ResultScorer(enable_scorer=True, min_confidence_score=0.60)
    strict_results = strict_scorer.score_documents([doc_a, doc_b])
    self.assertEqual(len(strict_results), 1, "Only doc_b should pass confidence filter")
    self.assertEqual(strict_results[0].metadata["document_id"], "doc_b")
    logger.info("✓ Scorer sorting and threshold filtering verified successfully.")

    # 9. Fallback behavior (on exception)
    logger.info("\n--- Verifying Fallback Behavior ---")
    faulty_scorer = ResultScorer(enable_scorer=True)
    # Force exception inside scoring loops
    faulty_scorer.score_agreement = MagicMock(side_effect=Exception("Critical Agreement Error"))
    
    fallback_res = faulty_scorer.score_documents([doc_a, doc_b])
    # Verify both documents are still returned (never fail retrieval)
    self.assertEqual(len(fallback_res), 2)
    logger.info("✓ Fallback behavior verified successfully.")

    # 10. Pipeline Integration
    logger.info("\n--- Verifying Pipeline Integration ---")
    mock_hybrid_call = MagicMock()
    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    
    doc_x = Document(page_content="A" * 1200, metadata={"document_id": "doc_x", "similarity_score": 0.80})
    doc_y = Document(page_content="Short", metadata={"document_id": "doc_y", "similarity_score": 0.40})
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
        mock_get_adaptive.return_value = adaptive_retriever

        pipeline = RetrievalPipeline()
        pipeline_results = pipeline.retrieve("Machine Learning", {"owner_id": "user123", "session_id": "s1"})
        
        # Check that metadata decorators are present
        self.assertTrue("confidence_score" in pipeline_results[0].metadata)
        self.assertTrue("confidence_level" in pipeline_results[0].metadata)
        self.assertTrue("ranking_reason" in pipeline_results[0].metadata)
        self.assertTrue("retrieval_sources" in pipeline_results[0].metadata)

    logger.info("✓ RetrievalPipeline result scoring integration verified successfully.")

    # 11. Cleanup
    logger.info("\nCleaning up test resources...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.17 RESULT SCORER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.17 RESULT SCORER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_17()
