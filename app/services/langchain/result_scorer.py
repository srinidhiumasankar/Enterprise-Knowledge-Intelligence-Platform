# app/services/langchain/result_scorer.py
# --------------------------------------
# Core Result Scorer module to evaluate and score confidence levels of retrieved documents.

import logging
import time
import re
from typing import Any, List, Dict, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ResultScorer:
    """
    Intelligent confidence scorer assessing multiple retrieval signals (semantic, RRF, metadata, agreement,
    chunk size, citations, and conversation query history boosts) to assign confidence scores and levels.
    """
    def __init__(
        self,
        enable_scorer: bool = True,
        weights: Dict[str, float] = None,
        min_confidence_score: float = 0.0
    ):
        self.enable_scorer = enable_scorer
        self.weights = weights or {
            "semantic": 0.30,
            "metadata": 0.15,
            "rrf": 0.15,
            "agreement": 0.10,
            "chunk_completeness": 0.10,
            "citation_completeness": 0.10,
            "history_boost": 0.05,
            "rewrite_boost": 0.05,
        }
        self.min_confidence_score = min_confidence_score

    def score_agreement(self, doc: Document) -> float:
        """
        Rewards documents retrieved by multiple retrievers.
        Checks 'retrieval_sources' array and 'retrieval_source' values.
        """
        meta = doc.metadata or {}
        sources = meta.get("retrieval_sources") or []
        primary = meta.get("retrieval_source")
        
        sources_set = set(sources)
        if primary:
            sources_set.add(primary)
            
        # If returned by multiple retrievers, assign 1.0; else 0.0
        return 1.0 if len(sources_set) > 1 else 0.0

    def score_chunk_completeness(self, doc: Document) -> float:
        """
        Rewards longer complete document chunks and penalizes short chunks.
        """
        content_len = len(doc.page_content or "")
        if content_len > 1000:
            return 1.0
        if content_len > 500:
            return 0.8
        if content_len > 200:
            return 0.5
        return 0.1  # Penalize short chunks

    def score_citation_completeness(self, doc: Document) -> float:
        """
        Rewards documents with rich citation indicators (filename, page_number, citation_key).
        """
        meta = doc.metadata or {}
        citation_keys = ["filename", "page_number", "citation_key"]
        
        present = sum(1 for k in citation_keys if k in meta and meta[k] is not None)
        return present / len(citation_keys)

    def score_history_boost(self, doc: Document, history: List[Dict[str, Any]]) -> float:
        """
        Performs clean word overlap matching between the document text and conversation history queries.
        """
        if not history:
            return 0.0

        history_queries = []
        for h in history:
            history_queries.append(h.get("user_query", ""))
            history_queries.append(h.get("rewritten_query", ""))

        # Tokenize unique context terms (> 3 characters)
        history_words = set()
        for q in history_queries:
            for w in q.lower().split():
                if len(w) > 3:
                    clean = re.sub(r'[^\w]', '', w)
                    if clean:
                        history_words.add(clean)

        if not history_words:
            return 0.0

        # Tokenize document words
        doc_words = set(re.sub(r'[^\w\s]', '', doc.page_content.lower()).split())
        overlap = doc_words.intersection(history_words)
        
        # Scale boost based on number of intersecting terms
        return min(len(overlap) / 5.0, 1.0)

    def score_rewrite_boost(self, doc: Document) -> float:
        """
        Rewards document if query rewriting altered the query.
        """
        meta = doc.metadata or {}
        orig = meta.get("original_query")
        rewritten = meta.get("rewritten_query")
        if orig and rewritten and orig != rewritten:
            return 1.0
        return 0.0

    def get_confidence_level(self, score: float) -> str:
        """
        Maps a numeric confidence score to its classification level.
        """
        if score >= 0.85:
            return "Very High"
        if score >= 0.70:
            return "High"
        if score >= 0.50:
            return "Medium"
        if score >= 0.30:
            return "Low"
        return "Very Low"

    def score_documents(
        self,
        docs: List[Document],
        history: List[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Computes a unified confidence score for all documents, ranks them,
        applies threshold filtering, and decorates document metadata.
        """
        if not docs:
            return []

        if not self.enable_scorer:
            logger.info("Result Scorer is disabled. Returning documents unmodified.")
            return docs

        start_time = time.time()
        scored_docs = []

        # Find RRF boundaries for normalization
        rrf_values = [float(d.metadata.get("rrf_score", 0.5)) for d in docs]
        max_rrf = max(rrf_values) if rrf_values else 1.0
        min_rrf = min(rrf_values) if rrf_values else 0.0

        for doc in docs:
            meta = doc.metadata or {}
            
            # Base retriever scores
            semantic_score = float(meta.get("semantic_score", 0.5))
            metadata_score = float(meta.get("metadata_score", 0.5))
            
            # Normalize RRF score between 0.0 and 1.0
            rrf_raw = float(meta.get("rrf_score", 0.5))
            rrf_score = (
                (rrf_raw - min_rrf) / (max_rrf - min_rrf)
                if (max_rrf > min_rrf)
                else 1.0
            )

            # Signal calculations
            try:
                agreement_score = self.score_agreement(doc)
                chunk_score = self.score_chunk_completeness(doc)
                citation_score = self.score_citation_completeness(doc)
                history_score = self.score_history_boost(doc, history)
                rewrite_score = self.score_rewrite_boost(doc)
            except Exception as e:
                logger.warning(f"Error calculating confidence signals: {e}. Fallback to neutral values.", exc_info=True)
                agreement_score = 0.0
                chunk_score = 0.5
                citation_score = 0.5
                history_score = 0.0
                rewrite_score = 0.0

            # Sum weighted scores
            confidence_score = (
                (semantic_score * self.weights.get("semantic", 0.30)) +
                (metadata_score * self.weights.get("metadata", 0.15)) +
                (rrf_score * self.weights.get("rrf", 0.15)) +
                (agreement_score * self.weights.get("agreement", 0.10)) +
                (chunk_score * self.weights.get("chunk_completeness", 0.10)) +
                (citation_score * self.weights.get("citation_completeness", 0.10)) +
                (history_score * self.weights.get("history_boost", 0.05)) +
                (rewrite_score * self.weights.get("rewrite_boost", 0.05))
            )

            confidence_level = self.get_confidence_level(confidence_score)

            # Re-evaluate retrieval sources list
            sources = meta.get("retrieval_sources") or []
            primary = meta.get("retrieval_source")
            sources_set = set(sources)
            if primary:
                sources_set.add(primary)

            # Build scoring reason
            reasons = []
            if agreement_score > 0.7:
                reasons.append("Retriever agreement consensus")
            if chunk_score > 0.7:
                reasons.append("Highly complete document chunk")
            if citation_score > 0.7:
                reasons.append("Complete citation details")
            if history_score > 0.3:
                reasons.append("Matches conversation topic history")
            if rewrite_score > 0.7:
                reasons.append("Query expansion boost")
                
            ranking_reason = ", ".join(reasons) if reasons else "Standard confidence match"

            # Filter by minimum confidence score
            if confidence_score < self.min_confidence_score:
                logger.info(
                    f"Filtering out document ID '{meta.get('document_id', 'unknown')}' "
                    f"with confidence score {confidence_score:.3f} < threshold {self.min_confidence_score:.3f}"
                )
                continue

            # Decorate document metadata
            doc.metadata["semantic_score"] = semantic_score
            doc.metadata["metadata_score"] = metadata_score
            doc.metadata["confidence_score"] = confidence_score
            doc.metadata["confidence_level"] = confidence_level
            doc.metadata["ranking_reason"] = ranking_reason
            doc.metadata["retrieval_sources"] = list(sources_set)

            scored_docs.append(doc)

        # Sort descending by confidence_score
        scored_docs.sort(key=lambda d: d.metadata.get("confidence_score", 0.0), reverse=True)

        latency = (time.time() - start_time) * 1000
        logger.info(f"Result scoring executed in {latency:.2f}ms. Ranked {len(scored_docs)} documents.")

        # Log details
        for i, doc in enumerate(scored_docs):
            doc_id = doc.metadata.get("document_id", "unknown")
            logger.info(
                f"Scored Rank {i+1}: Doc ID: {doc_id} | "
                f"Confidence Score: {doc.metadata['confidence_score']:.3f} | "
                f"Confidence Level: {doc.metadata['confidence_level']} | "
                f"Reason: {doc.metadata['ranking_reason']} | "
                f"Sources: {doc.metadata['retrieval_sources']}"
            )

        return scored_docs
