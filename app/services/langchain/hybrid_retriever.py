# app/services/langchain/hybrid_retriever.py
# ------------------------------------------
# LangChain Hybrid Retriever combining ChromaDB vector search and BM25 keyword search.

import logging
import time
import re
import math
from typing import Any, List, Dict, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config.settings import settings

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Lightweight alphanumeric tokenization converting text to lowercase.
    """
    return [word for word in re.findall(r'\w+', text.lower()) if word]


class BM25:
    """
    Lightweight, self-contained Python BM25 implementation.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avg_doc_len = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._initialize(corpus)

    def _initialize(self, corpus: List[List[str]]):
        for doc in corpus:
            self.doc_lens.append(len(doc))
            frequencies: Dict[str, int] = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                self.df[word] = self.df.get(word, 0) + 1

        for word, freq in self.df.items():
            # BM25 standard IDF formulation
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: List[str]) -> List[float]:
        scores = []
        for i in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lens[i]
            frequencies = self.doc_freqs[i]
            for word in query:
                if word not in frequencies:
                    continue
                freq = frequencies[word]
                idf = self.idf.get(word, 0.0)
                numerator = idf * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += numerator / denominator
            scores.append(score)
        return scores


class ChromaHybridRetriever(BaseRetriever):
    """
    Custom LangChain Hybrid Retriever merging Chroma vector similarity with local BM25 keyword matching.
    """
    chroma_service: Any = Field(description="The underlying ChromaService instance")
    embedding_service: Any = Field(description="The underlying EmbeddingService instance")
    owner_id: Optional[int] = Field(default=None, description="Optional owner ID filter")
    document_id: Optional[Any] = Field(default=None, description="Optional document ID filter")
    top_k: int = Field(default=5, description="Number of results to retrieve")
    semantic_weight: float = Field(default_factory=lambda: settings.HYBRID_SEMANTIC_WEIGHT)
    keyword_weight: float = Field(default_factory=lambda: settings.HYBRID_KEYWORD_WEIGHT)
    where_override: Optional[Dict[str, Any]] = Field(default=None, description="Optional Chroma filter clause override")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieves top documents using a combination of vector similarity and keyword relevance.
        """
        start_time = time.time()
        logger.info(f"HybridRetriever execution started for query: '{query}'")

        try:
            # Build filters
            if self.where_override is not None:
                where_clause = self.where_override
            else:
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

            # 1. Semantic Search
            semantic_start = time.time()
            if hasattr(self.embedding_service, "generate_query_embedding"):
                query_embedding = self.embedding_service.generate_query_embedding(query)
            elif hasattr(self.embedding_service, "embed_query"):
                query_embedding = self.embedding_service.embed_query(query)
            else:
                raise AttributeError("The provided embedding_service does not support generate_query_embedding or embed_query.")

            n_semantic = max(self.top_k * 2, 20)
            
            logger.info(f"Retrieving semantic matches from ChromaDB (limit={n_semantic})...")
            semantic_results = self.chroma_service.similarity_search(
                query_embedding=query_embedding,
                n_results=n_semantic,
                where=where_clause
            )

            ids_s = semantic_results.get("ids", [[]])[0] if semantic_results.get("ids") else []
            distances_s = semantic_results.get("distances", [[]])[0] if semantic_results.get("distances") else []
            documents_s = semantic_results.get("documents", [[]])[0] if semantic_results.get("documents") else []
            metadatas_s = semantic_results.get("metadatas", [[]])[0] if semantic_results.get("metadatas") else []

            semantic_docs = []
            for i in range(len(ids_s)):
                meta = metadatas_s[i].copy() if metadatas_s else {}
                meta["chunk_id"] = ids_s[i]
                meta["distance"] = distances_s[i] if distances_s else 0.0
                meta["similarity"] = 1.0 / (1.0 + distances_s[i]) if distances_s else 0.0
                
                doc = Document(page_content=documents_s[i], metadata=meta)
                semantic_docs.append(doc)

            semantic_latency = (time.time() - semantic_start) * 1000
            logger.info(f"Semantic search returned {len(semantic_docs)} chunks in {semantic_latency:.2f}ms")

            # 2. Keyword Search (BM25)
            keyword_start = time.time()
            logger.info(f"Fetching all chunks for filters {where_clause} to calculate keyword relevance...")
            all_chunks = self.chroma_service.collection.get(where=where_clause)
            
            ids_all = all_chunks.get("ids", []) if all_chunks.get("ids") else []
            documents_all = all_chunks.get("documents", []) if all_chunks.get("documents") else []
            metadatas_all = all_chunks.get("metadatas", []) if all_chunks.get("metadatas") else []

            keyword_docs = []
            if documents_all:
                corpus = [tokenize(doc_text) for doc_text in documents_all]
                bm25 = BM25(corpus)
                query_tokens = tokenize(query)
                scores = bm25.get_scores(query_tokens)

                scored_chunks = []
                for i in range(len(ids_all)):
                    scored_chunks.append({
                        "id": ids_all[i],
                        "document": documents_all[i],
                        "metadata": metadatas_all[i] if metadatas_all else {},
                        "score": scores[i]
                    })
                
                scored_chunks.sort(key=lambda x: x["score"], reverse=True)
                n_keyword = max(self.top_k * 2, 20)
                top_keyword_chunks = scored_chunks[:n_keyword]

                for chunk in top_keyword_chunks:
                    if chunk["score"] <= 0.0:
                        continue
                    meta = chunk["metadata"].copy()
                    meta["chunk_id"] = chunk["id"]
                    meta["bm25_score"] = chunk["score"]
                    
                    doc = Document(page_content=chunk["document"], metadata=meta)
                    keyword_docs.append(doc)

            keyword_latency = (time.time() - keyword_start) * 1000
            logger.info(f"Keyword search returned {len(keyword_docs)} chunks in {keyword_latency:.2f}ms")

            # 3. Reciprocal Rank Fusion (RRF)
            logger.info(f"Merging results using RRF (weights: semantic={self.semantic_weight}, keyword={self.keyword_weight})...")
            rrf_constant = 60.0
            rrf_scores = {}
            doc_map = {}

            # Process semantic ranks
            for rank, doc in enumerate(semantic_docs):
                chunk_id = doc.metadata["chunk_id"]
                doc_map[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self.semantic_weight * (1.0 / (rank + rrf_constant))

            # Process keyword ranks
            for rank, doc in enumerate(keyword_docs):
                chunk_id = doc.metadata["chunk_id"]
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                else:
                    doc_map[chunk_id].metadata.update(doc.metadata)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + self.keyword_weight * (1.0 / (rank + rrf_constant))

            semantic_ids = {doc.metadata["chunk_id"] for doc in semantic_docs}
            keyword_ids = {doc.metadata["chunk_id"] for doc in keyword_docs}

            fused_docs = []
            for chunk_id, score in rrf_scores.items():
                doc = doc_map[chunk_id]
                in_s = chunk_id in semantic_ids
                in_k = chunk_id in keyword_ids
                
                if in_s and in_k:
                    source = "hybrid"
                elif in_s:
                    source = "semantic"
                else:
                    source = "keyword"
                
                doc.metadata["retrieval_source"] = source
                doc.metadata["rrf_score"] = score
                fused_docs.append(doc)

            # Sort by RRF score descending and limit to top-k
            fused_docs.sort(key=lambda x: x.metadata["rrf_score"], reverse=True)
            final_docs = fused_docs[:self.top_k]

            total_latency = (time.time() - start_time) * 1000
            logger.info(
                f"Hybrid search complete: merged {len(fused_docs)} unique chunks. "
                f"Returning top {len(final_docs)} results. "
                f"Total hybrid retrieval latency: {total_latency:.2f}ms"
            )
            return final_docs

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"ChromaHybridRetriever retrieval failed: {e}") from e


def get_hybrid_retriever(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    semantic_weight: Optional[float] = None,
    keyword_weight: Optional[float] = None,
    chroma_service: Optional[Any] = None,
    embedding_service: Optional[Any] = None,
) -> BaseRetriever:
    """
    Factory function to construct a ChromaHybridRetriever.

    Parameters:
        owner_id (Optional[int]): Owner context filter.
        document_id (Optional[int]): Document ID filter.
        top_k (int): Number of top results to return.
        semantic_weight (Optional[float]): Custom semantic RRF weight (loads from settings if None).
        keyword_weight (Optional[float]): Custom keyword RRF weight (loads from settings if None).
        chroma_service (Optional[Any]): Underlying Chroma Service.
        embedding_service (Optional[Any]): Underlying Embedding Service.

    Returns:
        BaseRetriever: The configured Hybrid Retriever.
    """
    logger.info("Initializing LangChain Hybrid retriever...")
    from app.embeddings.chroma_service import ChromaService
    from app.embeddings.embedding_service import EmbeddingService

    active_chroma = chroma_service or ChromaService()
    active_embeddings = embedding_service or EmbeddingService()

    active_semantic_weight = semantic_weight if semantic_weight is not None else settings.HYBRID_SEMANTIC_WEIGHT
    active_keyword_weight = keyword_weight if keyword_weight is not None else settings.HYBRID_KEYWORD_WEIGHT

    return ChromaHybridRetriever(
        chroma_service=active_chroma,
        embedding_service=active_embeddings,
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        semantic_weight=active_semantic_weight,
        keyword_weight=active_keyword_weight
    )
