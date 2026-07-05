# app/services/langchain/ensemble.py
# ----------------------------------
# Custom Ensemble Retriever combining Hybrid, Multi Query, and Parent Document retrievers using weighted RRF.

import logging
import time
import hashlib
from typing import Any, List, Dict, Optional, Tuple
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


def get_doc_key(doc: Document) -> str:
    """
    Generate a unique key for a document to facilitate deduplication.
    Uses chunk_id if present, fallback to hash of page_content and document_id.
    """
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    
    # Fallback content hashing
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    doc_id = doc.metadata.get("document_id", "")
    return f"{doc_id}_{content_hash}"


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble Retriever combining Hybrid, Multi Query, and Parent Document Retrievers.
    Applies weighted Reciprocal Rank Fusion (RRF) for ranking.
    """
    hybrid_retriever: BaseRetriever = Field(description="Underlying hybrid retriever instance")
    multi_query_retriever: BaseRetriever = Field(description="Underlying multi-query retriever instance")
    parent_retriever: BaseRetriever = Field(description="Underlying parent document retriever instance")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"hybrid": 0.45, "multi_query": 0.35, "parent": 0.20},
        description="Weights for RRF fusion"
    )
    top_k: int = Field(default=5, description="Number of results to retrieve")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        start_time = time.time()
        logger.info("========================================================")
        logger.info(f"Starting Ensemble Retriever execution for: '{query}'")

        retrieved_results = {}
        execution_times = {}

        # 1. Execute Hybrid retriever
        try:
            h_start = time.time()
            docs = self.hybrid_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            retrieved_results["hybrid"] = docs
            latency = (time.time() - h_start) * 1000
            execution_times["hybrid"] = latency
            logger.info(f"Hybrid Retriever returned {len(docs)} documents in {latency:.2f}ms")
        except Exception as e:
            logger.warning(f"Hybrid Retriever failed in Ensemble: {e}", exc_info=True)
            retrieved_results["hybrid"] = []
            execution_times["hybrid"] = 0.0

        # 2. Execute Multi Query retriever
        try:
            mq_start = time.time()
            docs = self.multi_query_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            retrieved_results["multi_query"] = docs
            latency = (time.time() - mq_start) * 1000
            execution_times["multi_query"] = latency
            logger.info(f"Multi Query Retriever returned {len(docs)} documents in {latency:.2f}ms")
        except Exception as e:
            logger.warning(f"Multi Query Retriever failed in Ensemble: {e}", exc_info=True)
            retrieved_results["multi_query"] = []
            execution_times["multi_query"] = 0.0

        # 3. Execute Parent Document retriever
        try:
            p_start = time.time()
            docs = self.parent_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            retrieved_results["parent"] = docs
            latency = (time.time() - p_start) * 1000
            execution_times["parent"] = latency
            logger.info(f"Parent Retriever returned {len(docs)} documents in {latency:.2f}ms")
        except Exception as e:
            logger.warning(f"Parent Retriever failed in Ensemble: {e}", exc_info=True)
            retrieved_results["parent"] = []
            execution_times["parent"] = 0.0

        # Log document counts per retriever
        for k, v in retrieved_results.items():
            logger.info(f"Retriever '{k}' returned {len(v)} docs.")

        # 4. Merge and RRF Rank Fusion
        logger.info("Merging retrieved documents and applying weighted RRF...")
        rrf_constant = 60.0
        rrf_scores = {}
        doc_map = {}
        duplicate_count = 0

        # Resolve weights
        w_hybrid = self.weights.get("hybrid", 0.45)
        w_mq = self.weights.get("multi_query", 0.35)
        w_parent = self.weights.get("parent", 0.20)

        weights_map = {
            "hybrid": w_hybrid,
            "multi_query": w_mq,
            "parent": w_parent,
        }

        for name, docs in retrieved_results.items():
            weight = weights_map.get(name, 1.0)
            for rank, doc in enumerate(docs):
                doc_key = get_doc_key(doc)
                if doc_key not in doc_map:
                    new_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                    if "retrieval_source" not in new_doc.metadata:
                        new_doc.metadata["retrieval_source"] = name
                    doc_map[doc_key] = new_doc
                else:
                    duplicate_count += 1
                    # Merge metadata, preserving existing
                    current_meta = doc_map[doc_key].metadata
                    for k, v in doc.metadata.items():
                        if current_meta.get(k) is None:
                            current_meta[k] = v
                    # Keep track of multiple sources
                    sources = current_meta.get("retrieval_sources", [])
                    if name not in sources:
                        sources.append(name)
                    current_meta["retrieval_sources"] = sources
                
                # Accumulate weighted RRF score
                rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + weight * (1.0 / (rank + rrf_constant))

        logger.info(f"Duplicate removal count: {duplicate_count}. Unique documents remaining: {len(doc_map)}")

        # Build list for ranking
        ranked_docs = []
        for doc_key, score in rrf_scores.items():
            doc = doc_map[doc_key]
            doc.metadata["rrf_score"] = score
            ranked_docs.append(doc)

        # Ranking sorting function prioritizing RRF, similarity, confidence, completeness
        def ranking_key(d: Document) -> Tuple[float, float, float, int]:
            rrf_score = d.metadata.get("rrf_score", 0.0)
            
            similarity_score = d.metadata.get("score", 0.0)
            if "distance" in d.metadata:
                dist = d.metadata["distance"]
                if dist is not None:
                    similarity_score = max(similarity_score, 1.0 / (1.0 + dist))

            confidence = d.metadata.get("confidence", 0.0)
            
            citation_fields = [
                "citation_key",
                "page_number",
                "filename",
                "owner_id",
                "document_id",
                "parent_document_id"
            ]
            completeness = sum(1 for field in citation_fields if d.metadata.get(field) is not None)
            
            return (rrf_score, similarity_score, confidence, completeness)

        # Sort descending
        ranked_docs.sort(key=ranking_key, reverse=True)
        final_docs = ranked_docs[:self.top_k]

        # Log final ranking
        logger.info("Final Ranked Results:")
        for idx, doc in enumerate(final_docs, 1):
            logger.info(
                f"{idx}: content_len={len(doc.page_content)} | "
                f"RRF={doc.metadata.get('rrf_score', 0.0):.6f} | "
                f"source={doc.metadata.get('retrieval_source')} | "
                f"citation={doc.metadata.get('citation_key')}"
            )

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Ensemble Retriever completed in {total_latency:.2f}ms. Returning top {len(final_docs)} documents.")
        logger.info("========================================================")

        return final_docs
