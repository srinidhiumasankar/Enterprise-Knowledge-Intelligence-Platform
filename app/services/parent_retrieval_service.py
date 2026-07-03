# app/services/parent_retrieval_service.py
# ----------------------------------------
# Service wrapper for Parent Document Retriever instantiation and document store indexing.

import logging
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.services.langchain.parent_retriever import ParentRetriever, _shared_docstore

logger = logging.getLogger(__name__)


class ParentRetrievalService:
    """
    Service responsible for managing parent-child document mapping and expansions.
    """

    def __init__(self, docstore: Optional[Any] = None):
        logger.info("Initializing Parent Retrieval Service...")
        self.docstore = docstore or _shared_docstore

    def get_parent_retriever(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        child_chunk_size: int = 300,
        child_overlap: int = 30,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
        base_retriever: Optional[BaseRetriever] = None,
    ) -> ParentRetriever:
        """
        Builds and returns a configured ParentRetriever wrapper.
        """
        return ParentRetriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap,
            docstore=self.docstore,
            base_retriever=base_retriever
        )


# Global singleton instance of ParentRetrievalService
_parent_retrieval_service_instance = ParentRetrievalService()


def get_parent_retrieval_service() -> ParentRetrievalService:
    """
    FastAPI dependency injection provider returning the singleton ParentRetrievalService instance.
    """
    return _parent_retrieval_service_instance
