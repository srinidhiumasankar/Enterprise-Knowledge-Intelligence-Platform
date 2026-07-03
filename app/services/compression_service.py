# app/services/compression_service.py
# -----------------------------------
# Service wrapper for Contextual Compression and documents text trimming.

import logging
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.services.langchain.compression import CompressionRetriever, LLMBulkDocumentCompressor
from app.services.langchain.llm import get_llm

logger = logging.getLogger(__name__)


class CompressionService:
    """
    Service responsible for contextual document compression.
    """

    def __init__(self, llm: Optional[Any] = None):
        logger.info("Initializing Compression Service...")
        self.llm = llm or get_llm()
        self.compressor = LLMBulkDocumentCompressor(llm=self.llm)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Directly compress a list of Document chunks with respect to a user query.
        """
        return list(self.compressor.compress_documents(documents, query))

    def get_compression_retriever(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        base_retriever: Optional[BaseRetriever] = None,
    ) -> CompressionRetriever:
        """
        Builds and returns a configured CompressionRetriever wrapper.
        """
        return CompressionRetriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            llm=self.llm,
            base_retriever=base_retriever
        )


# Global singleton instance of CompressionService
_compression_service_instance = CompressionService()


def get_compression_service() -> CompressionService:
    """
    FastAPI dependency injection provider returning the singleton CompressionService instance.
    """
    return _compression_service_instance
