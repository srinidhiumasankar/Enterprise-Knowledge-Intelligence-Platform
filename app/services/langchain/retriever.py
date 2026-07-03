# app/services/langchain/retriever.py
# -----------------------------------
# Custom LangChain Retriever implementation wrapping the platform's ChromaService.

import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class ChromaLangChainRetriever(BaseRetriever):
    """
    LangChain BaseRetriever wrapper for the existing ChromaService vector store.
    """
    chroma_service: Any = Field(description="The underlying ChromaService instance")
    embedding_service: Any = Field(description="The underlying EmbeddingService instance")
    owner_id: Optional[int] = Field(default=None, description="Optional owner ID filter")
    document_id: Optional[int] = Field(default=None, description="Optional document ID filter")
    top_k: int = Field(default=5, description="Number of results to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Synthesize vector embedding of query, execute filters, and fetch matching documents from ChromaDB.

        Parameters:
            query (str): The search query string.
            run_manager (CallbackManagerForRetrieverRun): Callbacks manager.

        Returns:
            List[Document]: List of LangChain Document objects with page content and metadata.
        """
        start_time = time.time()
        logger.info(f"Retriever execution started for query: '{query}'")

        try:
            # 1. Generate query embedding
            logger.info("Generating query embedding using LangChain pipeline...")
            query_embedding = self.embedding_service.generate_query_embedding(query)

            # 2. Build metadata filters (owner_id / document_id)
            filters = []
            if self.owner_id is not None:
                filters.append({"owner_id": self.owner_id})
            if self.document_id is not None:
                filters.append({"document_id": self.document_id})

            where_clause = None
            if len(filters) == 1:
                where_clause = filters[0]
            elif len(filters) > 1:
                where_clause = {"$and": filters}

            logger.info(f"Executing similarity search in ChromaDB. Filters: {where_clause}, top_k={self.top_k}")
            
            # 3. Retrieve vectors from ChromaDB
            search_results = self.chroma_service.similarity_search(
                query_embedding=query_embedding,
                n_results=self.top_k,
                where=where_clause
            )

            # Extract lists from Chroma structure
            ids = search_results.get("ids", [[]])[0] if search_results.get("ids") else []
            distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
            documents = search_results.get("documents", [[]])[0] if search_results.get("documents") else []
            metadatas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []

            # 4. Construct LangChain Documents
            docs = []
            for i in range(len(ids)):
                meta = metadatas[i].copy() if metadatas else {}
                meta["chunk_id"] = ids[i]
                meta["distance"] = distances[i] if distances else 0.0
                
                doc = Document(
                    page_content=documents[i],
                    metadata=meta
                )
                docs.append(doc)

            latency = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(docs)} documents. Latency: {latency:.2f}ms")
            return docs

        except Exception as e:
            logger.error(f"Retriever execution encountered an error: {e}", exc_info=True)
            raise RuntimeError(f"ChromaLangChainRetriever retrieval failed: {e}") from e


def get_retriever(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    chroma_service: Optional[Any] = None,
    embedding_service: Optional[Any] = None
) -> BaseRetriever:
    """
    Factory function to construct a ChromaHybridRetriever.
    """
    logger.info("Initializing LangChain retriever (redirecting to Hybrid)...")
    from app.services.langchain.hybrid_retriever import get_hybrid_retriever
    return get_hybrid_retriever(
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        chroma_service=chroma_service,
        embedding_service=embedding_service
    )
