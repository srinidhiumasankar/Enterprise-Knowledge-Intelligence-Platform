# tests/verify_phase_8_16.py
# --------------------------
# Verification script for Phase 8.16 (Metadata Aware Ranking).
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

from app.services.langchain.metadata_ranker import MetadataRanker, parse_time
from app.services.langchain.metadata_ranker_service import MetadataRankerService
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.adaptive import AdaptiveRetriever

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_16")


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


def verify_phase_8_16():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.16 METADATA RANKER VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Initialization and Config ---")
    service = MetadataRankerService()
    ranker = service.get_metadata_ranker()
    self.assertEqual(ranker.enable_ranker, True)
    self.assertTrue("semantic" in ranker.weights)
    logger.info("✓ MetadataRankerService and MetadataRanker initialized correctly.")

    # 2. Freshness scoring (Decay of timestamp)
    logger.info("\n--- Verifying Freshness Scoring ---")
    now_ts = time.time()
    old_ts = now_ts - (60 * 24 * 3600)  # 60 days old
    
    doc_fresh = Document(page_content="Fresh doc", metadata={"created_at": now_ts})
    doc_old = Document(page_content="Old doc", metadata={"created_at": old_ts})
    doc_none = Document(page_content="No date doc", metadata={})
    
    score_fresh = ranker.score_freshness(doc_fresh)
    score_old = ranker.score_freshness(doc_old)
    score_none = ranker.score_freshness(doc_none)
    
    self.assertTrue(score_fresh > score_old, "Fresh document should score higher than old document")
    self.assertEqual(score_none, 0.5, "Missing timestamp should fall back to neutral 0.5")
    logger.info("✓ Freshness bonus and decays verified successfully.")

    # 3. Priority/importance bonus
    logger.info("\n--- Verifying Importance Scoring ---")
    doc_critical = Document(page_content="Critical doc", metadata={"priority": "critical"})
    doc_high = Document(page_content="High doc", metadata={"priority": "high"})
    doc_normal = Document(page_content="Normal doc", metadata={"priority": "normal"})
    doc_low = Document(page_content="Low doc", metadata={"priority": "low"})
    doc_no_prio = Document(page_content="No prio doc", metadata={})

    self.assertEqual(ranker.score_importance(doc_critical), 1.0)
    self.assertEqual(ranker.score_importance(doc_high), 0.8)
    self.assertEqual(ranker.score_importance(doc_normal), 0.5)
    self.assertEqual(ranker.score_importance(doc_low), 0.2)
    self.assertEqual(ranker.score_importance(doc_no_prio), 0.5)
    logger.info("✓ Importance priority scores verified successfully.")

    # 4. Document type bonus
    logger.info("\n--- Verifying Document Type Bonus ---")
    doc_policy = Document(page_content="Policy doc", metadata={"document_type": "Policy"})
    doc_guide = Document(page_content="Guide doc", metadata={"document_type": "Guide"})
    doc_notes = Document(page_content="Notes doc", metadata={"document_type": "Notes"})
    doc_misc = Document(page_content="Other doc", metadata={"document_type": "other"})

    self.assertEqual(ranker.score_doc_type(doc_policy), 1.0)
    self.assertEqual(ranker.score_doc_type(doc_guide), 1.0)
    self.assertEqual(ranker.score_doc_type(doc_notes), 0.1)
    self.assertEqual(ranker.score_doc_type(doc_misc), 0.5)
    logger.info("✓ Document type bonuses and penalties verified successfully.")

    # 5. Citation bonus
    logger.info("\n--- Verifying Citation Bonus ---")
    doc_citation_1 = Document(page_content="Doc with page", metadata={"page_number": 12})
    doc_citation_2 = Document(page_content="Doc with file", metadata={"filename": "doc.pdf"})
    doc_no_citation = Document(page_content="Doc with nothing", metadata={})

    self.assertEqual(ranker.score_citation(doc_citation_1), 1.0)
    self.assertEqual(ranker.score_citation(doc_citation_2), 1.0)
    self.assertEqual(ranker.score_citation(doc_no_citation), 0.0)
    logger.info("✓ Citation detail checks verified successfully.")

    # 6. Metadata completeness
    logger.info("\n--- Verifying Metadata Completeness ---")
    doc_complete = Document(page_content="Full metadata", metadata={
        "owner_id": "owner1", "document_id": "doc1", "created_at": now_ts, "updated_at": now_ts,
        "priority": "high", "document_type": "Policy", "page_number": 1, "filename": "x.pdf", "citation_key": "c1"
    })
    doc_sparse = Document(page_content="Few metadata", metadata={"owner_id": "owner1"})
    
    self.assertEqual(ranker.score_completeness(doc_complete), 1.0)
    self.assertEqual(ranker.score_completeness(doc_sparse), 1.0 / 9.0)
    logger.info("✓ Metadata completeness calculation verified successfully.")

    # 7. Weighted ranking and sorting
    logger.info("\n--- Verifying Weighted Ranking and Sorting ---")
    
    # Document A: high semantic, but notes, low priority
    doc_a = Document(page_content="Doc A", metadata={
        "document_id": "doc_a",
        "similarity_score": 0.90,
        "rrf_score": 0.80,
        "document_type": "Notes",
        "priority": "low"
    })
    
    # Document B: normal semantic, but policy, critical priority, full citation
    doc_b = Document(page_content="Doc B", metadata={
        "document_id": "doc_b",
        "similarity_score": 0.70,
        "rrf_score": 0.70,
        "document_type": "Policy",
        "priority": "critical",
        "page_number": 15,
        "filename": "policy.pdf",
        "citation_key": "policy_key"
    })
    
    ranked_list = ranker.rank_documents([doc_a, doc_b])
    
    # Verify Doc B is sorted first because of high metadata scores
    self.assertEqual(ranked_list[0].metadata["document_id"], "doc_b")
    # Verify metadata fields are populated
    self.assertTrue("semantic_score" in ranked_list[0].metadata)
    self.assertTrue("rrf_score" in ranked_list[0].metadata)
    self.assertTrue("metadata_score" in ranked_list[0].metadata)
    self.assertTrue("final_score" in ranked_list[0].metadata)
    self.assertTrue("ranking_reason" in ranked_list[0].metadata)
    logger.info("✓ Multi-factor sorting and metadata decorators verified successfully.")

    # 8. Pipeline Integration
    logger.info("\n--- Verifying Pipeline Integration ---")
    mock_hybrid_call = MagicMock()
    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    
    mock_hybrid_call.return_value = [doc_a, doc_b]

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
        results = pipeline.retrieve("Test search", {"owner_id": "user123", "session_id": "s1"})
        
        # Check that we received ranked documents (Doc B first)
        self.assertEqual(results[0].metadata["document_id"], "doc_b")

    logger.info("✓ RetrievalPipeline integration verified successfully.")

    # 9. Cleanup
    logger.info("\nCleaning up test resources...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.16 METADATA RANKER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.16 METADATA RANKER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_16()
