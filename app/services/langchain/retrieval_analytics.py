# app/services/langchain/retrieval_analytics.py
# ----------------------------------------------
# Subsystem to monitor and measure performance of the RAG pipeline.
# Collects high-resolution metrics for retrievers, cache hits/misses, and latency breakdown.

import logging
import time
import uuid
import threading
from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field, asdict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrieverMetric:
    """
    Performance and diagnostic metrics for an individual retriever execution.
    """
    executed: bool = False
    skipped: bool = True
    latency: float = 0.0
    returned_document_count: int = 0
    success: bool = False
    failure: bool = False


@dataclass
class RetrievalMetrics:
    """
    Aggregated metrics payload for a single retrieval request.
    """
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    owner_id: Optional[int] = None
    session_id: str = ""
    original_query: str = ""
    rewritten_query: str = ""
    final_query: str = ""
    query_length: int = 0

    # Retriever breakdown
    retrievers: Dict[str, RetrieverMetric] = field(default_factory=lambda: {
        "hybrid": RetrieverMetric(),
        "parent": RetrieverMetric(),
        "multi_query": RetrieverMetric(),
        "compression": RetrieverMetric(),
        "self_query": RetrieverMetric(),
        "ensemble": RetrieverMetric(),
        "adaptive": RetrieverMetric(),
    })

    # Cache breakdown
    cache_enabled: bool = False
    cache_hit: bool = False
    cache_miss: bool = False
    cache_lookup_latency: float = 0.0

    # Document filtering/ranking breakdown
    docs_before_ranking: int = 0
    docs_after_ranking: int = 0
    duplicate_removal_count: int = 0
    metadata_ranking_applied: bool = False
    confidence_threshold_applied: bool = False
    final_ranked_document_count: int = 0

    # Latencies (in milliseconds)
    total_latency: float = 0.0
    hybrid_latency: float = 0.0
    parent_latency: float = 0.0
    multi_query_latency: float = 0.0
    compression_latency: float = 0.0
    self_query_latency: float = 0.0
    ensemble_latency: float = 0.0
    adaptive_latency: float = 0.0
    query_rewrite_latency: float = 0.0
    metadata_ranker_latency: float = 0.0
    result_scorer_latency: float = 0.0
    answer_verifier_latency: float = 0.0
    conversation_memory_latency: float = 0.0
    total_llm_latency: float = 0.0
    embedding_latency: float = 0.0

    # Errors recorded
    errors: Dict[str, str] = field(default_factory=dict)
    status: str = "Success"


class RetrievalAnalytics:
    """
    Observability manager tracking performance latency, queries, cache,
    and retriever status across concurrent request threads.
    """
    _instance: Optional["RetrievalAnalytics"] = None
    _lock = threading.Lock()
    _local = threading.local()

    def __init__(self):
        self.history: List[RetrievalMetrics] = []
        self.history_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RetrievalAnalytics":
        """
        Retrieves the thread-safe singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _get_or_create_metrics(self) -> RetrievalMetrics:
        """
        Gets the current thread-scoped request metrics object.
        """
        if not hasattr(self._local, "current_metrics") or self._local.current_metrics is None:
            self._local.current_metrics = RetrievalMetrics(request_id=str(uuid.uuid4()))
        return self._local.current_metrics

    def start_request(
        self,
        request_id: Optional[str] = None,
        owner_id: Optional[int] = None,
        session_id: Optional[str] = None,
        original_query: Optional[str] = None
    ) -> str:
        """
        Registers the start of a retrieval request.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return ""

        rid = request_id or str(uuid.uuid4())
        metrics = RetrievalMetrics(
            request_id=rid,
            timestamp=time.time(),
            owner_id=owner_id,
            session_id=session_id or "",
            original_query=original_query or "",
            query_length=len(original_query) if original_query else 0
        )
        self._local.current_metrics = metrics
        self._local.start_perf_counter = time.perf_counter()
        
        logger.info(f"Retrieval Analytics started for request: {rid}")
        return rid

    def finish_request(
        self,
        rewritten_query: Optional[str] = None,
        final_query: Optional[str] = None,
        docs_before: int = 0,
        docs_after: int = 0
    ) -> Optional[RetrievalMetrics]:
        """
        Completes the current request, calculates total latency, and logs findings.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return None

        metrics = self._get_or_create_metrics()
        
        # Calculate overall latency
        if hasattr(self._local, "start_perf_counter"):
            metrics.total_latency = (time.perf_counter() - self._local.start_perf_counter) * 1000

        if rewritten_query:
            metrics.rewritten_query = rewritten_query
        if final_query:
            metrics.final_query = final_query
            
        metrics.docs_before_ranking = docs_before
        metrics.docs_after_ranking = docs_after

        # Auto-complete unexecuted retrievers in metrics
        for name, ret_metric in metrics.retrievers.items():
            if not ret_metric.executed:
                ret_metric.skipped = True
                ret_metric.executed = False

        # Add to history
        max_history = getattr(settings, "MAX_ANALYTICS_HISTORY", 1000)
        with self.history_lock:
            self.history.append(metrics)
            if len(self.history) > max_history:
                self.history.pop(0)

        # Print structured logging summary if enabled
        if getattr(settings, "ANALYTICS_VERBOSE", True):
            summary = self.build_summary()
            logger.info(summary)

        # Clear local thread scope
        self._local.current_metrics = None
        if hasattr(self._local, "start_perf_counter"):
            del self._local.start_perf_counter

        return metrics

    def record_latency(self, name: str, val: float):
        """
        Saves high-resolution subcomponent latencies (in ms).
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        if hasattr(metrics, name):
            setattr(metrics, name, getattr(metrics, name) + val)
        elif name == "llm_latency":
            metrics.total_llm_latency += val
        elif name == "embedding_latency":
            metrics.embedding_latency += val

    def record_query(self, original_query: str, rewritten_query: str, final_query: str):
        """
        Registers query alterations.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        metrics.original_query = original_query
        metrics.rewritten_query = rewritten_query
        metrics.final_query = final_query
        metrics.query_length = len(original_query)

    def record_retriever(
        self,
        retriever_name: str,
        executed: bool,
        skipped: bool,
        latency: float,
        returned_document_count: int,
        success: bool,
        failure: bool
    ):
        """
        Registers individual retriever operation facts.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        
        # Save to retriever breakdown
        if retriever_name in metrics.retrievers:
            metric = metrics.retrievers[retriever_name]
            metric.executed = executed
            metric.skipped = skipped
            metric.latency = latency
            metric.returned_document_count = returned_document_count
            metric.success = success
            metric.failure = failure

        # Update matching latency field in metrics
        latency_field = f"{retriever_name}_latency"
        if hasattr(metrics, latency_field):
            setattr(metrics, latency_field, latency)

    def record_cache(self, enabled: bool, hit: bool, miss: bool, lookup_latency: float):
        """
        Registers retrieval cache behavior.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        metrics.cache_enabled = enabled
        if hit:
            metrics.cache_hit = True
        if miss:
            metrics.cache_miss = True
        metrics.cache_lookup_latency += lookup_latency

    def record_documents(self, before: int, after: int, duplicates: int):
        """
        Registers duplicate counts and document set sizes.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        metrics.docs_before_ranking = before
        metrics.docs_after_ranking = after
        metrics.duplicate_removal_count = duplicates

    def record_metadata(self, metadata_applied: bool, threshold_applied: bool, final_count: int):
        """
        Registers post-retrieval ranking outcomes.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        metrics.metadata_ranking_applied = metadata_applied
        metrics.confidence_threshold_applied = threshold_applied
        metrics.final_ranked_document_count = final_count

    def record_error(self, retriever_name: str, error_message: str):
        """
        Registers retriever exceptions.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return

        metrics = self._get_or_create_metrics()
        metrics.errors[retriever_name] = error_message
        metrics.status = "Failed"
        if retriever_name in metrics.retrievers:
            metrics.retrievers[retriever_name].failure = True
            metrics.retrievers[retriever_name].success = False

    def build_summary(self) -> str:
        """
        Compiles current request stats into a structured log presentation.
        """
        metrics = self._get_or_create_metrics()
        
        # Determine used retrievers list
        used = [k for k, v in metrics.retrievers.items() if v.executed]
        retrievers_str = ", ".join(used) if used else "None"
        
        # Determine cache status
        cache_str = "Disabled"
        if metrics.cache_enabled:
            if metrics.cache_hit:
                cache_str = "Hit"
            elif metrics.cache_miss:
                cache_str = "Miss"
            else:
                cache_str = "Enabled"

        summary = f"""
====================================================
Retrieval Analytics
====================================================
Request ID:          {metrics.request_id}
Original Query:      '{metrics.original_query}'
Final Query:         '{metrics.final_query}'
Retriever Used:      {retrievers_str}
Total Latency:       {metrics.total_latency:.2f}ms
Cache:               {cache_str}
Documents Retrieved: {metrics.docs_before_ranking}
Documents Ranked:    {metrics.docs_after_ranking}
Verification Status: {metrics.status}
====================================================
"""
        return summary.strip()

    def export_dict(self) -> List[Dict[str, Any]]:
        """
        Exports history items to serializable dictionaries.
        """
        with self.history_lock:
            return [asdict(m) for m in self.history]

    def reset(self):
        """
        Purges request metrics and resets logs queue.
        """
        with self.history_lock:
            self.history.clear()
        self._local.current_metrics = None
        if hasattr(self._local, "start_perf_counter"):
            del self._local.start_perf_counter
        logger.info("Retrieval analytics memory reset completed.")


class MonitoredRetriever(BaseRetriever):
    """
    Subclass of BaseRetriever wrapping an underlying retriever to measure
    execution latencies and register document outputs in RetrievalAnalytics.
    """
    underlying: BaseRetriever = Field(description="The underlying base retriever")
    retriever_name: str = Field(description="Identifier name for the retriever component")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        analytics = RetrievalAnalytics.get_instance()
        
        t_start = time.perf_counter()
        success = False
        docs = []
        try:
            # Pass children callbacks properly to nested runners
            docs = self.underlying.invoke(
                query,
                config={"callbacks": run_manager.get_child() if run_manager else None}
            )
            success = True
            return docs
        except Exception as e:
            analytics.record_error(self.retriever_name, str(e))
            raise e
        finally:
            latency = (time.perf_counter() - t_start) * 1000
            analytics.record_retriever(
                retriever_name=self.retriever_name,
                executed=True,
                skipped=False,
                latency=latency,
                returned_document_count=len(docs),
                success=success,
                failure=not success
            )
            
            # Record component health status
            from app.services.langchain.health_monitor import HealthMonitor
            HealthMonitor.get_instance().record_component(
                name=self.retriever_name,
                status="HEALTHY" if success else "FAILED",
                latency=latency,
                error_message=None if success else "Exception during retriever execution"
            )

    def __getattr__(self, name: str) -> Any:
        if name == "underlying":
            raise AttributeError()
        return getattr(self.underlying, name)

    def __setattr__(self, name: str, value: Any):
        if name in ("where_override", "parent_document_retriever", "compression_retriever"):
            setattr(self.underlying, name, value)
        else:
            super().__setattr__(name, value)
