# app/services/langchain/ensemble_service.py
# ------------------------------------------
# Service wrapper for Ensemble Retriever construction and orchestration.

import logging
from typing import Any, Dict, Optional
from langchain_core.retrievers import BaseRetriever

from app.services.langchain.ensemble import EnsembleRetriever
from app.services.langchain.hybrid_retriever import get_hybrid_retriever
from app.services.langchain.multi_query import get_multi_query_retriever
from app.services.langchain.parent_retriever import ParentRetriever
from app.services.langchain.llm import get_llm

logger = logging.getLogger(__name__)


class EnsembleRetrieverService:
    """
    Service layer responsible for configuring and instantiating the EnsembleRetriever.
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_llm()

    def get_ensemble_retriever(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        child_chunk_size: int = 300,
        child_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
    ) -> BaseRetriever:
        """
        Constructs and returns an EnsembleRetriever configured with Hybrid, Multi Query, and Parent retrievers.
        """
        logger.info("Constructing Ensemble Retriever components...")

        # 1. Base Hybrid retriever
        hybrid_retriever = get_hybrid_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k
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
        parent_retriever = parent_wrapper.parent_document_retriever

        # 3. Multi Query retriever wrapping compression wrapping parent wrapping hybrid
        from app.services.langchain.compression import CompressionRetriever
        compression_wrapper = CompressionRetriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            base_retriever=parent_retriever
        )
        
        multi_query_retriever = get_multi_query_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            llm=self.llm,
            retriever=compression_wrapper.compression_retriever
        )

        # Configure weights
        active_weights = weights or {"hybrid": 0.45, "multi_query": 0.35, "parent": 0.20}

        return EnsembleRetriever(
            hybrid_retriever=hybrid_retriever,
            multi_query_retriever=multi_query_retriever,
            parent_retriever=parent_retriever,
            weights=active_weights,
            top_k=top_k
        )
