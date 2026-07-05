# tests/verify_phase_8_15.py
# --------------------------
# Verification script for Phase 8.15 (History Aware Retrieval).
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

from app.services.langchain.conversation_memory import ConversationMemory
from app.services.langchain.conversation_memory_service import ConversationMemoryService, get_memory_service
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.adaptive import AdaptiveRetriever
from app.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_15")


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


def verify_phase_8_15():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.15 HISTORY AWARE RETRIVAL VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Initialization and Config ---")
    memory = ConversationMemory(max_history=5)
    self.assertEqual(memory.max_history, 5)
    self.assertEqual(len(memory.history), 0)
    logger.info("✓ ConversationMemory initialized correctly.")

    # 2. Storing history
    logger.info("\n--- Verifying Storing History ---")
    memory.add_message("What is the capital of France?", "What is the capital of France?")
    self.assertEqual(len(memory.history), 1)
    self.assertEqual(memory.history[0]["user_query"], "What is the capital of France?")
    self.assertEqual(memory.history[0]["rewritten_query"], "What is the capital of France?")
    self.assertTrue(memory.history[0]["timestamp"] > 0)
    logger.info("✓ Storing history items verified successfully.")

    # 3. Limiting history size
    logger.info("\n--- Verifying Limiting History Size (Turn limit) ---")
    small_memory = ConversationMemory(max_history=3)
    for i in range(5):
        small_memory.add_message(f"Query {i}", f"Rewritten {i}")
    self.assertEqual(len(small_memory.history), 3)
    # Check that it contains the last 3 items
    self.assertEqual(small_memory.history[0]["user_query"], "Query 2")
    self.assertEqual(small_memory.history[2]["user_query"], "Query 4")
    logger.info("✓ Limiting history size verified successfully.")

    # 4. History aware rewriting & follow-up question handling
    logger.info("\n--- Verifying Rule-based Follow-up Handling ---")
    
    # Example 1: PTO policy / How many days
    pto_memory = ConversationMemory(max_history=5)
    pto_memory.add_message("What is the PTO policy?", "What is the PTO policy?")
    res_1 = pto_memory.build_context_aware_query("How many days?")
    self.assertEqual(res_1, "How many PTO policy days are employees allowed?", "PTO follow-up rephrasing failed")
    
    # Example 2: Finance reports / After 2022
    fin_memory = ConversationMemory(max_history=5)
    fin_memory.add_message("Tell me about Finance reports.", "Tell me about Finance reports.")
    res_2 = fin_memory.build_context_aware_query("After 2022?")
    self.assertEqual(res_2, "Finance reports After 2022", "Temporal relative rephrasing failed")
    
    # Example 3: Machine Learning / applications
    ml_memory = ConversationMemory(max_history=5)
    ml_memory.add_message("Explain Machine Learning.", "Explain Machine Learning.")
    res_3 = ml_memory.build_context_aware_query("What are its applications?")
    self.assertEqual(res_3, "Applications of Machine Learning", "Pronoun-noun rephrasing failed")
    logger.info("✓ Rule-based follow-up question rules verified successfully.")

    # 5. No-history fallback
    logger.info("\n--- Verifying No-history Fallback ---")
    empty_memory = ConversationMemory(max_history=5)
    fallback_res = empty_memory.build_context_aware_query("How many days?")
    self.assertEqual(fallback_res, "How many days?", "No-history fallback failed to return original query")
    logger.info("✓ No-history fallback verified successfully.")

    # 6. Pipeline Integration (RetrievalPipeline + Memory + Rewriter + Adaptive Retriever)
    logger.info("\n--- Verifying Pipeline Integration ---")
    
    mock_hybrid_call = MagicMock()
    mock_hybrid = MockRetriever(invoke_mock=mock_hybrid_call)
    
    # Setup mock document returns
    mock_doc = Document(
        page_content="Finance details for 2022",
        metadata={"source": "fin_doc", "rewritten_query": "Finance reports after year 2022"}
    )
    mock_hybrid_call.return_value = [mock_doc]

    # Patch target components in adaptive.py to use mock hybrid retriever
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

        # Start with fresh memory singleton state
        service = get_memory_service()
        service.sessions.clear()
        
        pipeline = RetrievalPipeline()
        
        # Turn 1: Establish context
        user_ctx = {"owner_id": "user123", "session_id": "test_session_999"}
        pipeline.retrieve("Tell me about Finance reports.", user_ctx)
        
        # Turn 2: Follow up query
        # Since 'after' is a preposition, it rewrites to: 'Finance reports After 2022?' -> 'Finance reports After 2022'
        # The query rewriter then processes: 'Finance reports After 2022' -> prepends 'year ' to '2022' -> 'Finance reports After year 2022'
        docs = pipeline.retrieve("After 2022?", user_ctx)
        
        # Verify downstream retriever received fully resolved query
        mock_hybrid_call.assert_called_with("Finance reports After year 2022")
        
        # Verify history holds both messages
        history = service.get_history("test_session_999")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["user_query"], "Tell me about Finance reports.")
        self.assertEqual(history[1]["user_query"], "After 2022?")
        
    logger.info("✓ Pipeline integration verified successfully.")

    # 7. Cleanup
    logger.info("\nCleaning up test artifacts...")
    service.sessions.clear()
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.15 HISTORY AWARE RETRIVAL VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.15 HISTORY AWARE RETRIVAL VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_15()
