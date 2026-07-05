# app/services/langchain/adaptive.py
# ----------------------------------
# Adaptive Retriever that analyzes query characteristics and dynamically routes to the best retrieval strategy.
# Employs a lightweight rule-based query classifier and implements robust cascading fallback behavior.

import logging
import time
from typing import Any, List, Dict, Tuple, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


def classify_query(query: str) -> Tuple[str, str]:
    """
    Lightweight rule-based query classifier.
    Does NOT use LLM.
    Returns (detected_query_type, routing_reason).
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)
    
    # 1. Temporal & Metadata Indicator Keywords
    temporal_indicators = ["after", "before", "since", "until", "during", "year", "date", "created"]
    metadata_indicators = [
        "department", "author", "filename", "owner", "owner_id", "dept", "department_id",
        "document_id", "user", "doc", "pdf", "txt", "docx", "csv", "xls", "xlsx", "file"
    ]
    # Check for numbers/years (e.g. 2022, 2024, etc.)
    has_year = any(w.isdigit() and len(w) == 4 and (w.startswith("19") or w.startswith("20")) for w in words)
    has_temporal = any(w in words for w in temporal_indicators) or has_year
    has_metadata_keyword = any(indicator in words or f"{indicator}:" in query_lower for indicator in metadata_indicators)
    # Logical Operators or Comparison Operators
    comparison_ops = [">", "<", "=", "==", ">=", "<="]
    has_comparison_ops = any(op in query_lower for op in comparison_ops)
    logical_ops = ["and", "or", "not"]
    has_logical_ops = any(op in words for op in logical_ops)

    # 2. Comparison / Multi-Topic Indicators
    comparison_indicators = ["vs", "versus", "difference", "compare", "comparison", "both", "as well as", "multi-topic", "various"]
    has_comparison = any(c in query_lower for c in comparison_indicators)

    # 3. Explanatory / Complexity Indicators
    explanatory_indicators = [
        "explain", "how do", "how does", "how to", "why did", "what are the details of",
        "describe", "understand", "background", "meaning of", "significance of",
        "summarize", "summary", "elaborate", "tell me about"
    ]
    has_explanatory = any(exp in query_lower for exp in explanatory_indicators) or word_count > 10

    # 4. Ambiguous / Vague Phrases
    ambiguous_indicators = ["about", "related to", "similar to", "something like", "topic of", "concept of", "anything on", "some info"]
    has_ambiguous = any(amb in query_lower for amb in ambiguous_indicators)

    # Routing Rules
    if has_comparison:
        return "comparison_multi_topic", f"Query contains comparison indicators (word count: {word_count})"
    elif has_metadata_keyword or has_temporal or has_comparison_ops or (has_logical_ops and has_year):
        return "metadata_filtering", "Query contains metadata attributes, temporal constraints, or query operators"
    elif has_explanatory:
        return "long_explanatory", f"Query specifies explanatory terms or exceeds length threshold (word count: {word_count})"
    elif has_ambiguous:
        return "ambiguous", "Query utilizes general/vague conceptual search keywords"
    else:
        return "simple_factual", f"Query classified as direct factual search (word count: {word_count})"


class AdaptiveRetriever(BaseRetriever):
    """
    Adaptive Retriever that routes queries to the optimal retriever based on query classification
    and handles retrieval failures with a cascading fallback chain.
    """
    hybrid_retriever: BaseRetriever = Field(description="Underlying hybrid retriever instance")
    self_query_retriever: BaseRetriever = Field(description="Underlying self query retriever instance")
    parent_retriever: BaseRetriever = Field(description="Underlying parent document retriever instance")
    multi_query_retriever: BaseRetriever = Field(description="Underlying multi-query retriever instance")
    ensemble_retriever: BaseRetriever = Field(description="Underlying ensemble retriever instance")
    
    enable_adaptive: bool = Field(default=True, description="Enable or disable adaptive routing")
    rules: Dict[str, str] = Field(
        default_factory=lambda: {
            "simple_factual": "hybrid",
            "metadata_filtering": "self_query",
            "long_explanatory": "parent",
            "ambiguous": "multi_query",
            "comparison_multi_topic": "ensemble"
        },
        description="Rules mapping classification keys to retrievers"
    )
    where_override: Any = Field(default=None, description="Injectable metadata filters")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        start_time = time.time()
        logger.info("========================================================")
        logger.info(f"Starting Adaptive Retriever execution for: '{query}'")

        # 1. Routing Decision
        if self.enable_adaptive:
            q_type, reason = classify_query(query)
            chosen_key = self.rules.get(q_type, "self_query")
            logger.info(f"Detected query type: '{q_type}'")
            logger.info(f"Routing reason: {reason}")
            logger.info(f"Chosen retriever: '{chosen_key}'")
        else:
            q_type = "disabled"
            chosen_key = "self_query"
            logger.info("Adaptive retrieval is disabled. Defaulting to: 'self_query'")

        # Create key-to-retriever map
        retriever_map = {
            "hybrid": self.hybrid_retriever,
            "self_query": self.self_query_retriever,
            "parent": self.parent_retriever,
            "multi_query": self.multi_query_retriever,
            "ensemble": self.ensemble_retriever
        }

        selected_retriever = retriever_map.get(chosen_key, self.self_query_retriever)

        # Rewrite query before retrieval
        from app.services.langchain.query_rewriter_service import QueryRewriterService
        rewriter_service = QueryRewriterService()
        rewriter = rewriter_service.get_query_rewriter()
        rewritten_query = rewriter.rewrite(query)

        # Propagate search filters if present
        if self.where_override is not None:
            if hasattr(selected_retriever, "where_override"):
                selected_retriever.where_override = self.where_override

        # Phase A: Execute chosen retriever
        try:
            logger.info(f"Invoking chosen retriever '{chosen_key}' with rewritten query...")
            r_start = time.time()
            docs = selected_retriever.invoke(rewritten_query, config={"callbacks": run_manager.get_child()})
            latency = (time.time() - r_start) * 1000
            total_latency = (time.time() - start_time) * 1000
            
            # Enrich metadata
            for doc in docs:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["original_query"] = query
                doc.metadata["rewritten_query"] = rewritten_query

            logger.info(f"Retrieval succeeded via '{chosen_key}' in {latency:.2f}ms. Returned {len(docs)} documents.")
            logger.info(f"Adaptive Retriever total latency: {total_latency:.2f}ms")
            logger.info("========================================================")
            return docs
        except Exception as e:
            logger.warning(
                f"Chosen retriever '{chosen_key}' failed: {e}. "
                f"Initiating fallback events: Ensemble -> Hybrid.",
                exc_info=True
            )

        # Phase B: Fallback to Ensemble (if chosen was not already ensemble)
        if chosen_key != "ensemble":
            try:
                logger.info("Fallback event (level 1): Invoking Ensemble Retriever...")
                if self.where_override is not None and hasattr(self.ensemble_retriever, "where_override"):
                    self.ensemble_retriever.where_override = self.where_override
                
                r_start = time.time()
                docs = self.ensemble_retriever.invoke(rewritten_query, config={"callbacks": run_manager.get_child()})
                latency = (time.time() - r_start) * 1000
                total_latency = (time.time() - start_time) * 1000
                
                # Enrich metadata
                for doc in docs:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["original_query"] = query
                    doc.metadata["rewritten_query"] = rewritten_query

                logger.info(f"Fallback retrieval succeeded via 'ensemble' in {latency:.2f}ms. Returned {len(docs)} documents.")
                logger.info(f"Adaptive Retriever total latency: {total_latency:.2f}ms")
                logger.info("========================================================")
                return docs
            except Exception as ex:
                logger.warning(f"Fallback Ensemble Retriever also failed: {ex}", exc_info=True)

        # Phase C: Fallback to Hybrid (if chosen was not already hybrid)
        if chosen_key != "hybrid":
            try:
                logger.info("Fallback event (level 2): Invoking Hybrid Retriever...")
                if self.where_override is not None and hasattr(self.hybrid_retriever, "where_override"):
                    self.hybrid_retriever.where_override = self.where_override
                
                r_start = time.time()
                docs = self.hybrid_retriever.invoke(rewritten_query, config={"callbacks": run_manager.get_child()})
                latency = (time.time() - r_start) * 1000
                total_latency = (time.time() - start_time) * 1000
                
                # Enrich metadata
                for doc in docs:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["original_query"] = query
                    doc.metadata["rewritten_query"] = rewritten_query

                logger.info(f"Fallback retrieval succeeded via 'hybrid' in {latency:.2f}ms. Returned {len(docs)} documents.")
                logger.info(f"Adaptive Retriever total latency: {total_latency:.2f}ms")
                logger.info("========================================================")
                return docs
            except Exception as ex:
                logger.error(f"All retrieval attempts failed (including Hybrid fallback): {ex}", exc_info=True)

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Adaptive Retriever completed (all failed, returned empty list) in {total_latency:.2f}ms.")
        logger.info("========================================================")
        return []
