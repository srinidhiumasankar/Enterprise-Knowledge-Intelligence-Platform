# tests/verify_phase_8_14.py
# --------------------------
# Verification script for Phase 8.14 (Query Rewriter).
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

from app.services.langchain.query_rewriter import QueryRewriter
from app.services.langchain.query_rewriter_service import QueryRewriterService
from app.services.langchain.adaptive import AdaptiveRetriever
from app.services.langchain.pipeline import RetrievalPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_14")


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


def verify_phase_8_14():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.14 QUERY REWRITER VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Initialization and Config ---")
    custom_rules = {
        "finance": "finance reports and financial documents",
        "vacation": "employee vacation leave policy"
    }
    custom_synonyms = {
        "vacation": "vacation leave policy",
        "leave": "employee leave policy"
    }
    custom_abbreviations = {
        "ai": "Artificial Intelligence",
        "pto": "Paid Time Off"
    }
    
    rewriter = QueryRewriter(
        enable_rewriter=True,
        rewrite_rules=custom_rules,
        synonym_map=custom_synonyms,
        abbreviation_map=custom_abbreviations
    )
    self.assertEqual(rewriter.enable_rewriter, True)
    logger.info("✓ QueryRewriter initialized correctly.")

    # 2. Normalization
    logger.info("\n--- Verifying Normalization ---")
    norm_res = rewriter.rewrite("  What   is   the   capital   of   France?  ")
    self.assertEqual(norm_res, "What is the capital of France?", "Whitespace normalization failed")
    logger.info("✓ Whitespace normalization verified successfully.")

    # 3. Abbreviation expansion
    logger.info("\n--- Verifying Abbreviation Expansion ---")
    abbr_res = rewriter.rewrite("What is our PTO policy?")
    self.assertEqual(abbr_res, "What is our Paid Time Off policy?", "Abbreviation expansion failed")
    logger.info("✓ Abbreviation expansion verified successfully.")

    # 4. Synonym expansion
    logger.info("\n--- Verifying Synonym Expansion ---")
    syn_res = rewriter.rewrite("Request sick leave today")
    self.assertEqual(syn_res, "Request sick employee leave policy today", "Synonym expansion failed")
    logger.info("✓ Synonym expansion verified successfully.")

    # 5. Direct query-level rewriting rules
    logger.info("\n--- Verifying Direct Query-level Rewriting Rules ---")
    rew_res_1 = rewriter.rewrite("finance")
    self.assertEqual(rew_res_1, "finance reports and financial documents")

    rew_res_2 = rewriter.rewrite("  vacation  ")
    self.assertEqual(rew_res_2, "employee vacation leave policy")
    logger.info("✓ Direct query-level rewriting rules verified successfully.")

    # 6. Numeric year prepending
    logger.info("\n--- Verifying Year Prepending ---")
    year_res = rewriter.rewrite("leave details 2022")
    self.assertEqual(year_res, "employee leave policy details year 2022")
    logger.info("✓ Prepending 'year ' to numeric years verified successfully.")

    # 7. Fallback behavior (if rewriter fails or disabled)
    logger.info("\n--- Verifying Fallback Behavior ---")
    
    # 7.1 If disabled
    disabled_rewriter = QueryRewriter(enable_rewriter=False)
    dis_res = disabled_rewriter.rewrite("PTO request")
    self.assertEqual(dis_res, "PTO request", "Disabled rewriter modified the query")
    
    # 7.2 If exception is thrown during rewrite, falls back to original query
    faulty_rewriter = QueryRewriter(enable_rewriter=True)
    # Mock self.rewrite_rules to throw an error on lookup
    faulty_rewriter.rewrite_rules = MagicMock(side_effect=Exception("Rewrite dictionary locked"))
    fallback_res = faulty_rewriter.rewrite("original search text")
    self.assertEqual(fallback_res, "original search text", "Faulty rewriter did not fallback to original query")
    logger.info("✓ Fallback behavior verified successfully.")

    # 8. Service Wrapper initialization
    logger.info("\n--- Verifying Service Wrapper ---")
    service = QueryRewriterService()
    service_rewriter = service.get_query_rewriter()
    self.assertEqual(service_rewriter.enable_rewriter, True)
    logger.info("✓ QueryRewriterService successfully built a configured QueryRewriter.")

    # 9. Pipeline Integration (via AdaptiveRetriever)
    logger.info("\n--- Verifying Pipeline Integration (AdaptiveRetriever) ---")
    
    # Mock sub-retrievers
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

    # Return mocked documents
    mock_doc = Document(page_content="Rewritten result content", metadata={"source": "mock_retriever"})
    mock_hybrid_call.return_value = [mock_doc]

    adaptive_retriever = AdaptiveRetriever(
        hybrid_retriever=mock_hybrid,
        self_query_retriever=mock_sq,
        parent_retriever=mock_parent,
        multi_query_retriever=mock_mq,
        ensemble_retriever=mock_ensemble,
        enable_adaptive=True
    )

    # Patch the service call inside AdaptiveRetriever
    with patch("app.services.langchain.query_rewriter_service.QueryRewriterService.get_query_rewriter") as mock_get_rewriter:
        # Configure the mock rewriter to expand "vacation" to "employee vacation leave policy"
        mock_rewriter_inst = QueryRewriter(
            enable_rewriter=True,
            rewrite_rules={"vacation": "employee vacation leave policy"},
            synonym_map=custom_synonyms,
            abbreviation_map=custom_abbreviations
        )
        mock_get_rewriter.return_value = mock_rewriter_inst

        # Trigger search query "vacation" -> simple factual category -> routes to Hybrid
        docs = adaptive_retriever.invoke("vacation")
        
        # Verify the downstream hybrid retriever received the rewritten query "employee vacation leave policy"
        mock_hybrid_call.assert_called_once_with("employee vacation leave policy")
        
        # Verify document metadata populated with original and rewritten query
        self.assertEqual(docs[0].metadata["original_query"], "vacation")
        self.assertEqual(docs[0].metadata["rewritten_query"], "employee vacation leave policy")

    logger.info("✓ Pipeline integration verifies query rewriting and metadata tracking successfully.")

    # 10. Cleanup
    logger.info("\nCleaning up test artifacts...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.14 QUERY REWRITER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.14 QUERY REWRITER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_14()
