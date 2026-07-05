# app/services/langchain/metadata_ranker.py
# ----------------------------------------
# Core Metadata Ranker to score, rank, and sort documents based on multi-factor metadata signals.

import logging
import time
import datetime
from typing import Any, List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def parse_time(val: Any) -> float:
    """
    Parses numeric, string, or datetime values into a unix epoch timestamp float.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, datetime.datetime):
        return val.timestamp()
    if isinstance(val, datetime.date):
        return datetime.datetime.combine(val, datetime.time.min).timestamp()
    if isinstance(val, str):
        val_clean = val.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(val_clean, fmt).timestamp()
            except ValueError:
                continue
    return None


class MetadataRanker:
    """
    Evaluates multi-factor metadata signals (freshness, importance, type, citation, completeness)
    to calculate a weighted score and rank documents after retrieval.
    """
    def __init__(self, enable_ranker: bool = True, weights: Dict[str, float] = None):
        self.enable_ranker = enable_ranker
        self.weights = weights or {
            "semantic": 0.45,
            "rrf": 0.20,
            "freshness": 0.10,
            "importance": 0.10,
            "type": 0.05,
            "citation": 0.05,
            "completeness": 0.05,
        }
        
    def score_freshness(self, doc: Document) -> float:
        """
        Calculates freshness score decaying over document age. Range: 0.0 to 1.0.
        Returns 0.5 (neutral) if timestamps are missing to prevent penalty.
        """
        meta = doc.metadata or {}
        time_val = meta.get("created_at") or meta.get("updated_at")
        doc_time = parse_time(time_val)
        if doc_time is None:
            return 0.5

        # Calculate document age in days
        age_days = (time.time() - doc_time) / (24 * 3600)
        if age_days < 0:
            age_days = 0.0
            
        # Sigmoid decay logic over 30 days
        return 1.0 / (1.0 + (age_days / 30.0))

    def score_importance(self, doc: Document) -> float:
        """
        Scores document priority values (critical: 1.0, high: 0.8, normal: 0.5, low: 0.2).
        """
        meta = doc.metadata or {}
        priority = str(meta.get("priority", "normal")).strip().lower()
        
        mapping = {
            "critical": 1.0,
            "high": 0.8,
            "normal": 0.5,
            "low": 0.2
        }
        return mapping.get(priority, 0.5)

    def score_doc_type(self, doc: Document) -> float:
        """
        Scores document type (Policies/Manuals: 1.0, Notes/Drafts: 0.1, others: 0.5).
        """
        meta = doc.metadata or {}
        doc_type = str(meta.get("document_type") or meta.get("doc_type") or meta.get("type", "other")).strip().lower()
        
        prefer_list = {"policy", "handbook", "manual", "specification", "guide"}
        demote_list = {"notes", "miscellaneous", "draft"}
        
        if doc_type in prefer_list:
            return 1.0
        if doc_type in demote_list:
            return 0.1
        return 0.5

    def score_citation(self, doc: Document) -> float:
        """
        Scores citation details existence. Returns 1.0 if keys are present; else 0.0.
        """
        meta = doc.metadata or {}
        citation_keys = {"page_number", "filename", "citation_key"}
        if any(k in meta for k in citation_keys):
            return 1.0
        return 0.0

    def score_completeness(self, doc: Document) -> float:
        """
        Scores completeness portion of 9 expected metadata fields.
        """
        meta = doc.metadata or {}
        expected_keys = [
            "owner_id", "document_id", "created_at", "updated_at",
            "priority", "document_type", "page_number", "filename", "citation_key"
        ]
        
        present_count = sum(1 for k in expected_keys if k in meta and meta[k] is not None)
        return present_count / len(expected_keys)

    def rank_documents(self, docs: List[Document]) -> List[Document]:
        """
        Executes metadata ranking, sorts documents descending, and adds final details to metadata.
        """
        if not docs:
            return []

        if not self.enable_ranker:
            logger.info("Metadata Ranker is disabled. Returning documents unsorted.")
            return docs

        start_time = time.time()
        ranked_docs = []

        for doc in docs:
            meta = doc.metadata or {}
            
            # 1. Base search scores (from query/vector store)
            semantic_score = float(meta.get("similarity_score") or meta.get("score") or 0.5)
            rrf_score = float(meta.get("rrf_score", 0.5))

            # 2. Metadata signals calculations (guaranteed no exceptions)
            try:
                freshness_score = self.score_freshness(doc)
                importance_score = self.score_importance(doc)
                type_score = self.score_doc_type(doc)
                citation_score = self.score_citation(doc)
                completeness_score = self.score_completeness(doc)
            except Exception as e:
                logger.warning(f"Error calculating metadata signals: {e}. Fallback to neutral values.", exc_info=True)
                freshness_score = 0.5
                importance_score = 0.5
                type_score = 0.5
                citation_score = 0.5
                completeness_score = 0.5

            # 3. Sum sub-weights
            metadata_score = (
                (freshness_score * self.weights.get("freshness", 0.10)) +
                (importance_score * self.weights.get("importance", 0.10)) +
                (type_score * self.weights.get("type", 0.05)) +
                (citation_score * self.weights.get("citation", 0.05)) +
                (completeness_score * self.weights.get("completeness", 0.05))
            )

            final_score = (
                (semantic_score * self.weights.get("semantic", 0.45)) +
                (rrf_score * self.weights.get("rrf", 0.20)) +
                metadata_score
            )

            # 4. Generate dynamic ranking reasons
            reasons = []
            if freshness_score > 0.7:
                reasons.append("Highly fresh document")
            if importance_score > 0.7:
                reasons.append(f"High importance ({meta.get('priority', 'high')})")
            if type_score > 0.7:
                reasons.append(f"Preferred document type ({meta.get('document_type') or meta.get('doc_type')})")
            if citation_score > 0.7:
                reasons.append("Citation details available")
            if completeness_score > 0.7:
                reasons.append("Rich metadata completeness")
                
            ranking_reason = ", ".join(reasons) if reasons else "Standard metadata score"

            # 5. Decorate document metadata
            doc.metadata["semantic_score"] = semantic_score
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["metadata_score"] = metadata_score
            doc.metadata["final_score"] = final_score
            doc.metadata["ranking_reason"] = ranking_reason

            ranked_docs.append(doc)

        # 6. Sort docs descending by final_score
        ranked_docs.sort(key=lambda d: d.metadata.get("final_score", 0.0), reverse=True)

        latency = (time.time() - start_time) * 1000
        logger.info(f"Metadata ranking executed in {latency:.2f}ms for {len(ranked_docs)} documents.")

        # Log individual details
        for i, doc in enumerate(ranked_docs):
            doc_id = doc.metadata.get("document_id", "unknown")
            logger.info(
                f"Rank {i+1}: Doc ID: {doc_id} | "
                f"Semantic: {doc.metadata['semantic_score']:.3f} | "
                f"Metadata: {doc.metadata['metadata_score']:.3f} | "
                f"Final: {doc.metadata['final_score']:.3f} | "
                f"Reason: {doc.metadata['ranking_reason']}"
            )

        return ranked_docs
