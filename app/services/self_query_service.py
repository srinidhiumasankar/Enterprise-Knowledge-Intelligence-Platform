# app/services/self_query_service.py
# ---------------------------------
# Service wrapper for Self Querying retriever instantiation and metadata queries.

import logging
from typing import Optional, Any
from langchain_core.retrievers import BaseRetriever

from app.services.langchain.self_query import get_self_query_retriever
from app.services.langchain.llm import get_llm

logger = logging.getLogger(__name__)


class SelfQueryService:
    """
    Service responsible for building and managing the SelfQueryRetriever.
    """

    def __init__(self, llm: Optional[Any] = None):
        logger.info("Initializing Self Query Service...")
        self.llm = llm or get_llm()

    def get_self_query_retriever(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        child_chunk_size: int = 300,
        child_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
    ) -> BaseRetriever:
        """
        Builds and returns a configured ChromaSelfQueryRetriever.
        """
        return get_self_query_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            llm=self.llm,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap
        )


# Global singleton instance of SelfQueryService
_self_query_service_instance = SelfQueryService()


def get_self_query_service() -> SelfQueryService:
    """
    FastAPI dependency injection provider returning the singleton SelfQueryService instance.
    """
    return _self_query_service_instance
