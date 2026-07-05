# app/services/langchain/retrieval_analytics_service.py
# ----------------------------------------------------
# Service layer managing configuration, wraps, exports, and lifespan of RetrievalAnalytics.

import logging
from typing import Any, Dict, List, Optional
from langchain_core.retrievers import BaseRetriever

from app.config import settings
from app.services.langchain.retrieval_analytics import RetrievalAnalytics, MonitoredRetriever

logger = logging.getLogger(__name__)


class RetrievalAnalyticsService:
    """
    Service wrapper managing telemetry resets, serializations, and dynamic wraps for retrievers.
    """
    def __init__(self):
        self.analytics = RetrievalAnalytics.get_instance()

    def get_analytics(self) -> RetrievalAnalytics:
        """
        Returns the core RetrievalAnalytics manager instance.
        """
        return self.analytics

    def wrap_retriever(self, retriever: BaseRetriever, name: str) -> BaseRetriever:
        """
        Wraps a retriever in a telemetry collector.
        If analytics are disabled, returns the original retriever directly.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return retriever
        
        # Don't double wrap
        if isinstance(retriever, MonitoredRetriever):
            return retriever

        return MonitoredRetriever(
            underlying=retriever,
            retriever_name=name
        )

    def wrap_embeddings(self, embeddings: Any) -> Any:
        """
        Wraps embedding calls to track embedding latency.
        """
        if not getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True):
            return embeddings

        class MonitoredEmbeddings:
            def __init__(self, underlying: Any):
                self._underlying = underlying

            def embed_query(self, text: str) -> List[float]:
                import time
                from app.services.langchain.retrieval_analytics import RetrievalAnalytics
                analytics = RetrievalAnalytics.get_instance()
                
                t_start = time.perf_counter()
                res = self._underlying.embed_query(text)
                latency = (time.perf_counter() - t_start) * 1000
                analytics.record_latency("embedding_latency", latency)
                return res

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                import time
                from app.services.langchain.retrieval_analytics import RetrievalAnalytics
                analytics = RetrievalAnalytics.get_instance()
                
                t_start = time.perf_counter()
                res = self._underlying.embed_documents(texts)
                latency = (time.perf_counter() - t_start) * 1000
                analytics.record_latency("embedding_latency", latency)
                return res

            def __getattr__(self, name: str) -> Any:
                return getattr(self._underlying, name)

        return MonitoredEmbeddings(embeddings)

    def export_analytics(self) -> List[Dict[str, Any]]:
        """
        Exports metrics logs to list of dictionaries.
        """
        return self.analytics.export_dict()

    def reset_analytics(self):
        """
        Clears telemetry metrics logs.
        """
        self.analytics.reset()
