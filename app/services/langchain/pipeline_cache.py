# app/services/langchain/pipeline_cache.py
# ----------------------------------------
# Request-scoped caching and tracking layer using contextvars.
# Prevents duplicate LLM and retriever queries within the lifetime of a single request.

import contextvars
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Request-scoped cache container ContextVar
_request_cache: contextvars.ContextVar[Optional["RequestCache"]] = contextvars.ContextVar("request_cache", default=None)


def get_current_request_cache() -> Optional["RequestCache"]:
    """
    Returns the active request-scoped cache context if one is set.
    """
    return _request_cache.get()


def freeze_value(val: Any) -> Any:
    """
    Recursively converts mutable inputs (dicts, lists, sets) into immutable,
    hashable tuple representations to prevent caching key validation errors.
    """
    if isinstance(val, dict):
        return tuple(sorted((k, freeze_value(v)) for k, v in val.items()))
    elif isinstance(val, list):
        return tuple(freeze_value(x) for x in val)
    elif isinstance(val, set):
        return tuple(sorted(freeze_value(x) for x in val))
    else:
        try:
            hash(val)
            return val
        except TypeError:
            return str(val)


class RequestCache:
    """
    Context manager for request-scoped caching and diagnostic metrics.
    Accumulates cache hit/miss statistics and counts LLM operations.
    """
    def __init__(self):
        self.cache: Dict[Tuple[Any, ...], Any] = {}
        self.llm_call_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._token = None

    def increment_llm_calls(self) -> None:
        self.llm_call_count += 1

    def log_hit(self, msg: str) -> None:
        self.cache_hits += 1
        logger.info(f"[CACHE HIT] {msg} (Hits: {self.cache_hits}, Misses: {self.cache_misses})")

    def log_miss(self, msg: str) -> None:
        self.cache_misses += 1
        logger.info(f"[CACHE MISS] {msg} (Hits: {self.cache_hits}, Misses: {self.cache_misses})")

    def __enter__(self) -> "RequestCache":
        self._token = _request_cache.set(self)
        logger.info("Started request-scoped retrieval cache context.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token:
            _request_cache.reset(self._token)
        logger.info(
            f"Finished request-scoped retrieval cache context. "
            f"Total LLM calls: {self.llm_call_count}, Cache hits: {self.cache_hits}, Cache misses: {self.cache_misses}"
        )


class CachingLLM:
    """
    Proxy wrapper around a LangChain LLM instance that intercepts invoke
    and generate calls to cache their results if a request cache is active.
    """
    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, *args, **kwargs) -> Any:
        import time
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        analytics = RetrievalAnalytics.get_instance()

        cache = get_current_request_cache()
        if cache is None:
            t_start = time.perf_counter()
            result = self._llm.invoke(*args, **kwargs)
            latency = (time.perf_counter() - t_start) * 1000
            analytics.record_latency("llm_latency", latency)
            return result

        t_lookup_start = time.perf_counter()
        key = ("llm_invoke", freeze_value(args), freeze_value(kwargs))
        in_cache = key in cache.cache
        lookup_latency = (time.perf_counter() - t_lookup_start) * 1000
        analytics.record_cache(enabled=True, hit=in_cache, miss=not in_cache, lookup_latency=lookup_latency)

        if in_cache:
            cache.log_hit(f"LLM invoke (args={args}, kwargs={kwargs})")
            return cache.cache[key]

        cache.log_miss(f"LLM invoke (args={args}, kwargs={kwargs})")
        cache.increment_llm_calls()
        
        t_start = time.perf_counter()
        result = self._llm.invoke(*args, **kwargs)
        latency = (time.perf_counter() - t_start) * 1000
        analytics.record_latency("llm_latency", latency)

        cache.cache[key] = result
        return result

    def generate(self, *args, **kwargs) -> Any:
        import time
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        analytics = RetrievalAnalytics.get_instance()

        cache = get_current_request_cache()
        if cache is None:
            t_start = time.perf_counter()
            result = self._llm.generate(*args, **kwargs)
            latency = (time.perf_counter() - t_start) * 1000
            analytics.record_latency("llm_latency", latency)
            return result

        t_lookup_start = time.perf_counter()
        key = ("llm_generate", freeze_value(args), freeze_value(kwargs))
        in_cache = key in cache.cache
        lookup_latency = (time.perf_counter() - t_lookup_start) * 1000
        analytics.record_cache(enabled=True, hit=in_cache, miss=not in_cache, lookup_latency=lookup_latency)

        if in_cache:
            cache.log_hit(f"LLM generate (args={args}, kwargs={kwargs})")
            return cache.cache[key]

        cache.log_miss(f"LLM generate (args={args}, kwargs={kwargs})")
        cache.increment_llm_calls()
        
        t_start = time.perf_counter()
        result = self._llm.generate(*args, **kwargs)
        latency = (time.perf_counter() - t_start) * 1000
        analytics.record_latency("llm_latency", latency)

        cache.cache[key] = result
        return result

    async def ainvoke(self, *args, **kwargs) -> Any:
        import time
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        analytics = RetrievalAnalytics.get_instance()

        cache = get_current_request_cache()
        if cache is None:
            t_start = time.perf_counter()
            result = await self._llm.ainvoke(*args, **kwargs)
            latency = (time.perf_counter() - t_start) * 1000
            analytics.record_latency("llm_latency", latency)
            return result

        t_lookup_start = time.perf_counter()
        key = ("llm_invoke", freeze_value(args), freeze_value(kwargs))
        in_cache = key in cache.cache
        lookup_latency = (time.perf_counter() - t_lookup_start) * 1000
        analytics.record_cache(enabled=True, hit=in_cache, miss=not in_cache, lookup_latency=lookup_latency)

        if in_cache:
            cache.log_hit(f"LLM ainvoke (args={args}, kwargs={kwargs})")
            return cache.cache[key]

        cache.log_miss(f"LLM ainvoke (args={args}, kwargs={kwargs})")
        cache.increment_llm_calls()
        
        t_start = time.perf_counter()
        result = await self._llm.ainvoke(*args, **kwargs)
        latency = (time.perf_counter() - t_start) * 1000
        analytics.record_latency("llm_latency", latency)

        cache.cache[key] = result
        return result

    async def agenerate(self, *args, **kwargs) -> Any:
        import time
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        analytics = RetrievalAnalytics.get_instance()

        cache = get_current_request_cache()
        if cache is None:
            t_start = time.perf_counter()
            result = await self._llm.agenerate(*args, **kwargs)
            latency = (time.perf_counter() - t_start) * 1000
            analytics.record_latency("llm_latency", latency)
            return result

        t_lookup_start = time.perf_counter()
        key = ("llm_generate", freeze_value(args), freeze_value(kwargs))
        in_cache = key in cache.cache
        lookup_latency = (time.perf_counter() - t_lookup_start) * 1000
        analytics.record_cache(enabled=True, hit=in_cache, miss=not in_cache, lookup_latency=lookup_latency)

        if in_cache:
            cache.log_hit(f"LLM agenerate (args={args}, kwargs={kwargs})")
            return cache.cache[key]

        cache.log_miss(f"LLM agenerate (args={args}, kwargs={kwargs})")
        cache.increment_llm_calls()
        
        t_start = time.perf_counter()
        result = await self._llm.agenerate(*args, **kwargs)
        latency = (time.perf_counter() - t_start) * 1000
        analytics.record_latency("llm_latency", latency)

        cache.cache[key] = result
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
