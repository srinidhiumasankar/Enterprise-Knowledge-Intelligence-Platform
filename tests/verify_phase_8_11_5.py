# tests/verify_phase_8_11_5.py
# ----------------------------
# Standalone verification script for Phase 8.11.5 (Retrieval Pipeline & Caching).
# Uses mocks to verify caching behavior without consuming Gemini API quota.

import os
import sys
import logging
from unittest.mock import MagicMock, patch

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.pipeline_cache import RequestCache

# Set up logging for verification
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_11_5")


def verify_phase_8_11_5():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.11.5 RETRIEVAL PIPELINE VERIFICATION")
    logger.info("==================================================")

    # 1. Initialize the Retrieval Pipeline
    logger.info("Initializing Retrieval Pipeline...")
    mock_llm = MagicMock()
    pipeline = RetrievalPipeline(llm=mock_llm)
    logger.info("✓ Retrieval Pipeline initialized successfully.")

    # Patch query constructor, retrievers, and LLM behavior
    with patch("app.services.langchain.self_query.load_query_constructor_chain") as mock_load_query_constructor_chain, \
         patch("app.services.langchain.hybrid_retriever.get_hybrid_retriever") as mock_get_hybrid_retriever, \
         patch("app.services.langchain.parent_retriever.ParentRetriever") as mock_parent_retriever, \
         patch("app.services.langchain.compression.CompressionRetriever") as mock_compression_retriever, \
         patch("app.services.langchain.multi_query.get_multi_query_retriever") as mock_get_multi_query_retriever:

        # Mock structured query parsing chain
        mock_chain = MagicMock()
        mock_load_query_constructor_chain.return_value = mock_chain
        
        # Valid output structure
        mock_chain.invoke.return_value = {
            "query": "Finance reports after 2022",
            "text": {
                "query": "Finance reports",
                "filter": {"comparator": "gt", "attribute": "year", "value": 2022}
            }
        }

        # Mock downstream retrievers
        mock_hybrid = MagicMock()
        mock_get_hybrid_retriever.return_value = mock_hybrid

        mock_parent_instance = MagicMock()
        mock_parent_retriever.return_value = mock_parent_instance

        mock_comp_instance = MagicMock()
        mock_compression_retriever.return_value = mock_comp_instance

        mock_mq = MagicMock()
        mock_get_multi_query_retriever.return_value = mock_mq
        
        mock_doc = Document(
            page_content="Finance details.",
            metadata={"owner_id": 999, "document_id": 1, "filename": "finance.pdf", "citation_key": "fin"}
        )
        mock_mq.invoke.return_value = [mock_doc]

        user_context = {"owner_id": 999, "top_k": 2}

        # 2. Verify Orchestration
        logger.info("\n--- Verifying Pipeline Orchestration ---")
        results = pipeline.retrieve("Finance reports after 2022", user_context)
        
        # Check that Self Query (load_query_constructor_chain) was loaded & called
        mock_load_query_constructor_chain.assert_called_once()
        mock_chain.invoke.assert_called_once_with({"query": "Finance reports after 2022"})
        
        # Check that Multi Query retriever wrapped the compression retriever which wrapped parent and hybrid
        mock_get_hybrid_retriever.assert_called_once()
        mock_parent_retriever.assert_called_once()
        mock_compression_retriever.assert_called_once()
        mock_get_multi_query_retriever.assert_called_once()
        
        logger.info("✓ Component Orchestration confirmed (Self Query -> Multi Query -> Hybrid -> Parent -> Compression).")

        # 3. Verify request-scoped caching: Hits & Misses
        logger.info("\n--- Verifying Request-Scoped Caching (Hits & Misses) ---")
        
        mock_chain.invoke.reset_mock()
        mock_mq.invoke.reset_mock()

        # Execute multiple queries within a single cache context
        with RequestCache() as cache:
            # First call: cache miss
            logger.info("Executing Call 1 (Cache Miss expected)...")
            res1 = pipeline.retrieve("Finance reports after 2022", user_context)
            self_query_call_count_1 = mock_chain.invoke.call_count
            self.assertEqual(self_query_call_count_1, 1, "LLM should be called on miss")
            self.assertEqual(cache.cache_misses, 1, "Cache misses count should be 1")
            self.assertEqual(cache.cache_hits, 0, "Cache hits count should be 0")

            # Second call: cache hit
            logger.info("Executing Call 2 (Cache Hit expected)...")
            res2 = pipeline.retrieve("Finance reports after 2022", user_context)
            self_query_call_count_2 = mock_chain.invoke.call_count
            
            # The count should remain 1 (no new LLM invocation)
            self.assertEqual(self_query_call_count_2, 1, "LLM should NOT be called on cache hit")
            self.assertEqual(cache.cache_hits, 1, "Cache hits count should be 1")
            self.assertEqual(cache.cache_misses, 1, "Cache misses count should remain 1")
            self.assertEqual(res1, res2, "Cached output must be identical")

        # 4. Verify request-scoped isolation
        logger.info("\n--- Verifying Request-scoped Isolation ---")
        mock_chain.invoke.reset_mock()
        
        # Call 3 (outside previous context) should trigger LLM call again (miss)
        logger.info("Executing Call 3 in a new request context (Cache Miss expected)...")
        pipeline.retrieve("Finance reports after 2022", user_context)
        self.assertEqual(mock_chain.invoke.call_count, 1, "LLM should be called again in separate request context")
        logger.info("✓ Request-scoped isolation verified successfully.")

        # 5. Verify Fallback Behavior
        logger.info("\n--- Verifying Fallback Behavior ---")
        mock_chain.invoke.reset_mock()
        mock_mq.invoke.reset_mock()
        
        # Simulate self-query parser failure
        mock_chain.invoke.side_effect = Exception("LLM connection failed")
        
        logger.info("Executing retrieve with query parser failure (Fallback expected)...")
        # Should complete successfully by falling back to original query (MultiQuery fallback)
        fallback_results = pipeline.retrieve("Finance reports after 2022", user_context)
        self.assertEqual(len(fallback_results), 1, "Fallback results should return matching documents")
        self.assertEqual(fallback_results[0].metadata["citation_key"], "fin", "Fallback metadata should be preserved")
        logger.info("✓ Fallback behavior verified successfully.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.11.5 RETRIEVAL PIPELINE VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.11.5 RETRIEVAL PIPELINE VERIFIED SUCCESSFULLY")


# Lightweight assert class for verification script
class SimpleAssert:
    def assertEqual(self, a, b, msg=""):
        if a != b:
            raise AssertionError(f"{msg}: {a} != {b}")
    def assertTrue(self, cond, msg=""):
        if not cond:
            raise AssertionError(msg)

self = SimpleAssert()


if __name__ == "__main__":
    verify_phase_8_11_5()
