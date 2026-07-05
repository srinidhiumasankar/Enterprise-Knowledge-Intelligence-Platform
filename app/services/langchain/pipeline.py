# app/services/langchain/pipeline.py
# ----------------------------------
# Central orchestrator for the platform's multi-stage retrieval pipeline.
# Integrates Self Query, Multi Query, Hybrid Search, Parent Retriever,
# and Contextual Compression with request-scoped caching.

import logging
import time
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from app.services.langchain.llm import get_llm
from app.services.langchain.self_query import get_self_query_retriever
from app.services.langchain.pipeline_cache import RequestCache, CachingLLM, get_current_request_cache

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """
    Orchestration pipeline that coordinates the multi-stage retriever flow:
    Self Query -> Multi Query -> Hybrid -> Parent -> Contextual Compression.
    Provides automatic request-scoped caching to save Gemini API credits.
    """
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def retrieve(self, query: str, user_context: Any) -> List[Document]:
        """
        Executes search using the optimized retrieval pipeline under request cache context.
        """
        start_time = time.time()
        logger.info(f"RetrievalPipeline.retrieve started for query: '{query}'")

        # Parse user context attributes safely
        if isinstance(user_context, dict):
            owner_id = user_context.get("owner_id")
            document_id = user_context.get("document_id")
            top_k = user_context.get("top_k", 5)
            child_chunk_size = user_context.get("child_chunk_size", 300)
            child_overlap = user_context.get("child_overlap", 30)
            parent_chunk_size = user_context.get("parent_chunk_size", 1024)
            parent_overlap = user_context.get("parent_overlap", 100)
            session_id = user_context.get("session_id")
        else:
            owner_id = getattr(user_context, "owner_id", None)
            document_id = getattr(user_context, "document_id", None)
            top_k = getattr(user_context, "top_k", 5)
            child_chunk_size = getattr(user_context, "child_chunk_size", 300)
            child_overlap = getattr(user_context, "child_overlap", 30)
            parent_chunk_size = getattr(user_context, "parent_chunk_size", 1024)
            parent_overlap = getattr(user_context, "parent_overlap", 100)
            session_id = getattr(user_context, "session_id", None)

        if not session_id:
            session_id = f"owner_{owner_id}" if owner_id else "default_session"

        # Collection and Workspace filtering integration
        if isinstance(user_context, dict):
            collection_ids = user_context.get("collection_ids")
            workspace_id = user_context.get("workspace_id")
        else:
            collection_ids = getattr(user_context, "collection_ids", None)
            workspace_id = getattr(user_context, "workspace_id", None)

        from app.config import settings
        enable_filtering = getattr(settings, "ENABLE_COLLECTION_FILTERING", True)

        if enable_filtering and (collection_ids is not None or workspace_id is not None):
            from app.database import SessionLocal
            from app.services.collection.collection_filter_service import CollectionFilterService
            from sqlalchemy import select

            db = SessionLocal()
            try:
                # Attempt to resolve workspace_id from owner_id if not present
                if not workspace_id and owner_id:
                    from app.models.workspace import Workspace
                    ws = db.scalars(select(Workspace).where(Workspace.owner_id == owner_id)).first()
                    workspace_id = ws.id if ws else None

                if workspace_id:
                    filter_service = CollectionFilterService(db)
                    resolved_doc_ids = filter_service.validate_and_resolve_filters(
                        user_id=owner_id,
                        workspace_id=workspace_id,
                        collection_ids=collection_ids
                    )
                    # Override document_id filter scope
                    if len(resolved_doc_ids) == 1:
                        document_id = resolved_doc_ids[0]
                    else:
                        document_id = {"$in": resolved_doc_ids}
            finally:
                db.close()

        # 0. Start Retrieval Analytics
        from app.services.langchain.retrieval_analytics import RetrievalAnalytics
        analytics = RetrievalAnalytics.get_instance()
        analytics.start_request(
            owner_id=owner_id,
            session_id=session_id,
            original_query=query
        )

        # 0.5 Start Health Monitoring
        from app.services.langchain.health_monitor import HealthMonitor
        monitor = HealthMonitor.get_instance()
        monitor.start_monitoring(request_id=analytics._get_or_create_metrics().request_id)

        # 1. Fetch History & Build Context Aware Query
        from app.services.langchain.conversation_memory_service import get_memory_service
        from app.config import settings
        
        enable_memory = getattr(settings, "ENABLE_CONVERSATION_MEMORY", True)
        memory_service = get_memory_service()
        
        t_memory_start = time.perf_counter()
        if enable_memory:
            memory = memory_service.get_memory(session_id)
            history = list(memory.history)  # snapshot for logging
            context_aware_query = memory.build_context_aware_query(query)
        else:
            history = []
            context_aware_query = query
        memory_latency = (time.perf_counter() - t_memory_start) * 1000
        analytics.record_latency("conversation_memory_latency", memory_latency)
        monitor.record_component("conversation_memory", "HEALTHY", memory_latency)

        # 2. Retrieve documents
        active_cache = get_current_request_cache()
        if active_cache is not None:
            results = self._execute_retrieval(
                context_aware_query, owner_id, document_id, top_k,
                child_chunk_size, child_overlap, parent_chunk_size, parent_overlap
            )
        else:
            with RequestCache() as cache:
                results = self._execute_retrieval(
                    context_aware_query, owner_id, document_id, top_k,
                    child_chunk_size, child_overlap, parent_chunk_size, parent_overlap
                )

        docs_retrieved_count = len(results) if results else 0
        analytics.record_documents(before=docs_retrieved_count, after=docs_retrieved_count, duplicates=0)

        # 2.5 Rank retrieved documents
        t_rank_start = time.perf_counter()
        from app.services.langchain.metadata_ranker_service import MetadataRankerService
        ranker_service = MetadataRankerService()
        ranker = ranker_service.get_metadata_ranker()
        results = ranker.rank_documents(results)
        rank_latency = (time.perf_counter() - t_rank_start) * 1000
        analytics.record_latency("metadata_ranker_latency", rank_latency)
        analytics.record_metadata(metadata_applied=True, threshold_applied=False, final_count=len(results))
        monitor.record_component("metadata_ranker", "HEALTHY", rank_latency)

        # 2.7 Score document confidence levels
        t_score_start = time.perf_counter()
        from app.services.langchain.result_scorer_service import ResultScorerService
        scorer_service = ResultScorerService()
        scorer = scorer_service.get_result_scorer()
        results = scorer.score_documents(results, history)
        score_latency = (time.perf_counter() - t_score_start) * 1000
        analytics.record_latency("result_scorer_latency", score_latency)
        analytics.record_metadata(metadata_applied=True, threshold_applied=True, final_count=len(results))
        monitor.record_component("result_scorer", "HEALTHY", score_latency)

        # 3. Determine final rewritten query
        final_rewritten_query = context_aware_query
        if results:
            final_rewritten_query = results[0].metadata.get("rewritten_query", context_aware_query)
        else:
            t_rewrite_start = time.perf_counter()
            from app.services.langchain.query_rewriter_service import QueryRewriterService
            rewriter_service = QueryRewriterService()
            rewriter = rewriter_service.get_query_rewriter()
            final_rewritten_query = rewriter.rewrite(context_aware_query)
            rewrite_latency = (time.perf_counter() - t_rewrite_start) * 1000
            analytics.record_latency("query_rewrite_latency", rewrite_latency)
            monitor.record_component("query_rewriter", "HEALTHY", rewrite_latency)

        # 4. Log stats
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Original Query: '{query}'")
        logger.info(f"Conversation Context: {[{'user_query': h['user_query'], 'rewritten_query': h['rewritten_query']} for h in history]}")
        logger.info(f"History Aware Query: '{context_aware_query}'")
        logger.info(f"Final Rewritten Query: '{final_rewritten_query}'")
        logger.info(f"History Size: {len(history)}")
        logger.info(f"Retrieval latency: {latency_ms:.2f}ms")

        # 5. Save current query to memory
        if enable_memory:
            memory_service.add_message(session_id, query, final_rewritten_query)

        # Record cache and analytics status
        cache_active = active_cache is not None or getattr(settings, "ENABLE_REQUEST_CACHE", True)
        monitor.record_component("cache", "HEALTHY" if cache_active else "WARNING", 0.0)

        analytics_active = getattr(settings, "ENABLE_RETRIEVAL_ANALYTICS", True)
        monitor.record_component("analytics", "HEALTHY" if analytics_active else "WARNING", 0.0)

        # 6. Finish Retrieval Analytics
        analytics.finish_request(
            rewritten_query=final_rewritten_query,
            final_query=final_rewritten_query,
            docs_before=docs_retrieved_count,
            docs_after=len(results) if results else 0
        )

        # 7. Finish Health Monitoring
        monitor.finish_monitoring()

        return results

    def _execute_retrieval(
        self,
        query: str,
        owner_id: Optional[int],
        document_id: Optional[int],
        top_k: int,
        child_chunk_size: int,
        child_overlap: int,
        parent_chunk_size: int,
        parent_overlap: int,
    ) -> List[Document]:
        # Wrap standard LLM in the request cache proxy
        active_llm = self.llm or get_llm()
        proxied_llm = CachingLLM(active_llm)

        logger.info("Initializing multi-stage retrieval chain components...")
        from app.services.langchain.adaptive_service import AdaptiveRetrieverService
        adaptive_service = AdaptiveRetrieverService(llm=proxied_llm)
        retriever = adaptive_service.get_adaptive_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            child_chunk_size=child_chunk_size,
            child_overlap=child_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_overlap=parent_overlap
        )

        return retriever.invoke(query)

    def chat(self, query: str, user_context: Any) -> str:
        """
        Executes a complete conversational RAG query:
        1. Validates ownership and loads history from ConversationMemoryService.
        2. Executes retrieval pipeline to find relevant documents.
        3. Formulates a context prompt using PromptBuilder.
        4. Invokes Gemini LLM to generate the final response.
        5. Saves both the user message and the assistant response to database.
        """
        if isinstance(user_context, dict):
            conversation_id = user_context.get("conversation_id")
            user_id = user_context.get("user_id") or user_context.get("owner_id")
            model_name = user_context.get("model_name", "gemini-2.5-flash")
        else:
            conversation_id = getattr(user_context, "conversation_id", None)
            user_id = getattr(user_context, "user_id", None) or getattr(user_context, "owner_id", None)
            model_name = getattr(user_context, "model_name", "gemini-2.5-flash")

        if not conversation_id or not user_id:
            raise ValueError("conversation_id and user_id/owner_id must be provided in user_context for chat flow")

        from app.database import SessionLocal
        from app.services.conversation.conversation_memory_service import ConversationMemoryService
        from app.ai.prompt_builder import PromptBuilder
        from app.ai.gemini_service import GeminiService

        db = SessionLocal()
        try:
            memory_service = ConversationMemoryService(db)
            
            # Load conversation history context
            history_text, _ = memory_service.load_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id
            )

            # Store the user's current query first
            memory_service.repo.add_message(
                conversation_id=conversation_id,
                role="user",
                content=query,
                token_count=memory_service.estimate_tokens(query)
            )

            # Run existing retrieval pipeline
            retrieved_docs = self.retrieve(query, user_context)

            # Build LLM prompt combining history, retrieved knowledge, and question
            prompt = PromptBuilder.build_prompt(
                question=query,
                chunks=retrieved_docs,
                conversation_history=history_text
            )

            # Call Gemini
            gemini = GeminiService()
            response = gemini.generate_answer(prompt=prompt, model_name=model_name)

            # Store assistant response
            memory_service.repo.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response,
                token_count=memory_service.estimate_tokens(response),
                model_name=model_name
            )

            logger.info("Conversational RAG response generated and persisted successfully.")
            return response

        finally:
            db.close()

