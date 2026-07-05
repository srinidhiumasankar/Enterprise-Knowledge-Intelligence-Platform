# app/services/langchain/adaptive_service.py
# ------------------------------------------
# Service wrapper for Adaptive Retriever construction, setting up all sub-retrievers,
# and injecting configuration rules.

import logging
from typing import Any, Dict, Optional
from langchain_core.retrievers import BaseRetriever

from app.services.langchain.adaptive import AdaptiveRetriever
from app.services.langchain.hybrid_retriever import get_hybrid_retriever
from app.services.langchain.multi_query import get_multi_query_retriever
from app.services.langchain.parent_retriever import ParentRetriever
from app.services.langchain.self_query import get_self_query_retriever
from app.services.langchain.ensemble import EnsembleRetriever
from app.services.langchain.llm import get_llm
from app.config import settings

logger = logging.getLogger(__name__)


class AdaptiveRetrieverService:
    """
    Service layer responsible for configuring and instantiating the AdaptiveRetriever with its sub-retrievers.
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_llm()

    def get_adaptive_retriever(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        child_chunk_size: int = 300,
        child_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
        enable_adaptive: Optional[bool] = None,
        rules: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> BaseRetriever:
        """
        Constructs and returns an AdaptiveRetriever pre-configured with all five sub-retrievers.
        """
        logger.info("Constructing Adaptive Retriever components...")
        from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
        analytics_service = RetrievalAnalyticsService()

        # 1. Base Hybrid retriever
        hybrid_retriever = analytics_service.wrap_retriever(
            get_hybrid_retriever(
                owner_id=owner_id,
                document_id=document_id,
                top_k=top_k
            ),
            "hybrid"
        )

        # 2. Parent Document retriever wrapping hybrid
        parent_wrapper = ParentRetriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap,
            base_retriever=hybrid_retriever
        )
        parent_retriever = analytics_service.wrap_retriever(
            parent_wrapper.parent_document_retriever,
            "parent"
        )

        # 3. Multi Query retriever wrapping compression wrapping parent wrapping hybrid
        from app.services.langchain.compression import CompressionRetriever
        compression_wrapper = CompressionRetriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            base_retriever=parent_retriever
        )
        compression_retriever = analytics_service.wrap_retriever(
            compression_wrapper.compression_retriever,
            "compression"
        )
        multi_query_retriever = analytics_service.wrap_retriever(
            get_multi_query_retriever(
                owner_id=owner_id,
                document_id=document_id,
                top_k=top_k,
                llm=self.llm,
                retriever=compression_retriever
            ),
            "multi_query"
        )

        # 4. Ensemble Retriever
        ensemble_retriever = analytics_service.wrap_retriever(
            EnsembleRetriever(
                hybrid_retriever=hybrid_retriever,
                multi_query_retriever=multi_query_retriever,
                parent_retriever=parent_retriever,
                weights=weights or {"hybrid": 0.45, "multi_query": 0.35, "parent": 0.20},
                top_k=top_k
            ),
            "ensemble"
        )

        # 5. Self Query Retriever (acts as standard self-query)
        self_query_retriever = analytics_service.wrap_retriever(
            get_self_query_retriever(
                owner_id=owner_id,
                document_id=document_id,
                top_k=top_k,
                llm=self.llm,
                child_chunk_size=child_chunk_size,
                child_overlap=child_overlap,
                parent_chunk_size=parent_chunk_size,
                parent_overlap=parent_overlap
            ),
            "self_query"
        )

        # Resolve config parameters
        active_enable_adaptive = (
            enable_adaptive
            if enable_adaptive is not None
            else getattr(settings, "ENABLE_ADAPTIVE_RETRIEVAL", True)
        )
        active_rules = (
            rules
            if rules is not None
            else getattr(settings, "ADAPTIVE_RULES", {
                "simple_factual": "hybrid",
                "metadata_filtering": "self_query",
                "long_explanatory": "parent",
                "ambiguous": "multi_query",
                "comparison_multi_topic": "ensemble"
            })
        )

        adaptive_retriever = AdaptiveRetriever(
            hybrid_retriever=hybrid_retriever,
            self_query_retriever=self_query_retriever,
            parent_retriever=parent_retriever,
            multi_query_retriever=multi_query_retriever,
            ensemble_retriever=ensemble_retriever,
            enable_adaptive=active_enable_adaptive,
            rules=active_rules
        )
        
        return analytics_service.wrap_retriever(adaptive_retriever, "adaptive")
