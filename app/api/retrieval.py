# app/api/retrieval.py
# --------------------
# Router implementing search API endpoints for collection-scoped semantic retrieval.

import logging
import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_retrieval_service, get_gemini_service, get_db
from app.models.user import User
from app.schemas.retrieval import CollectionRetrievalRequest
from app.schemas.search import SearchResponse, Citation, SearchResult
from app.services.retrieval_service import RetrievalService
from app.ai.prompt_builder import PromptBuilder
from app.ai.gemini_service import GeminiService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/retrieval", tags=["Collection Retrieval"])


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Collection-Aware Semantic Search",
    description="Perform semantic search on chunks within target collections inside the active workspace.",
)
async def search_collection_chunks(
    request_data: CollectionRetrievalRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    db: Session = Depends(get_db)
) -> Any:
    """
    Perform semantic vector query filtered to a specific set of document collections or workspace boundary.
    """
    from app.services.workspace.workspace_service import WorkspaceService
    from app.utils.activity_logger import record_rag_search_history

    query_str = request_data.query
    logger.info(f"User '{current_user.email}' requested collection-scoped semantic search for query '{query_str}'")

    if not query_str or not query_str.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace-only.",
        )

    # Determine active workspace for logging
    ws_service = WorkspaceService(db)
    active_ws = ws_service.get_active_workspace(current_user.id)
    workspace_id = request_data.workspace_id or (active_ws.id if active_ws else None)

    # Track metrics
    start_time = time.perf_counter()
    retrieval_latency = 0.0
    generation_latency = 0.0
    results = []

    try:
        # Retrieve chunks with collection and workspace filter parameters
        retrieval_start = time.perf_counter()
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=query_str,
            top_k=request_data.top_k or 5,
            collection_ids=request_data.collection_ids,
            workspace_id=workspace_id
        )
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000

        logger.info(f"Retrieved {len(results)} chunks for collection-scoped query: '{query_str}'")

        if not results:
            total_latency = (time.perf_counter() - start_time) * 1000
            if workspace_id:
                background_tasks.add_task(
                    record_rag_search_history,
                    current_user.id,
                    workspace_id,
                    query_str,
                    retrieval_latency,
                    0.0,
                    total_latency,
                    0,
                    [],
                    [],
                    "Success"
                )
            return SearchResponse(
                query=query_str,
                results=[],
                citations=[],
                answer="I cannot determine the answer from the uploaded documents.",
                message=None
            )

        # Extract structured citations
        seen_chunks = set()
        citations = []
        search_results = []
        for r in results:
            chunk_id = r["chunk_id"]
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                citations.append(
                    Citation(
                        document_id=r["document_id"],
                        filename=r["metadata"]["filename"],
                        chunk_id=chunk_id,
                        score=r["score"]
                    )
                )
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=r["document_id"],
                    text=r["text"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
            )

        # Build prompt incorporating multi-turn conversation memory if provided
        conversation_history = None
        memory_tokens = 0
        loaded_msg_count = 0
        if request_data.conversation_id is not None:
            try:
                from app.services.conversation.conversation_memory_service import ConversationMemoryService
                mem_service = ConversationMemoryService(db)
                conversation_history, memory_tokens = mem_service.load_conversation_history(
                    conversation_id=request_data.conversation_id,
                    user_id=current_user.id,
                    max_messages=11 # Fetch up to 11 to keep 10 history items after excluding current
                )
                if conversation_history:
                    lines = conversation_history.strip().split("\n")
                    if lines and lines[-1].startswith("User: ") and lines[-1][6:].strip().lower() == query_str.strip().lower():
                        lines.pop()
                    conversation_history = "\n".join(lines) if lines else None
                if conversation_history:
                    loaded_msg_count = len(conversation_history.strip().split("\n"))
            except Exception as mem_e:
                logger.error(f"Failed to load conversation history for RAG query: {mem_e}")
                conversation_history = None
                memory_tokens = 0
                loaded_msg_count = 0

        prompt = PromptBuilder.build_prompt(
            question=query_str,
            chunks=results,
            conversation_history=conversation_history
        )

        # Call Gemini Service to generate answer
        gemini_start = time.perf_counter()
        try:
            answer = gemini_service.generate_answer(prompt=prompt, model_name="gemini-2.5-flash")
            generation_latency = (time.perf_counter() - gemini_start) * 1000
            total_latency = (time.perf_counter() - start_time) * 1000

            # Log Phase 10.2 conversation memory & retrieval execution metrics
            logger.info(
                f"RAG search multi-turn metrics: conversation_id={request_data.conversation_id}, "
                f"loaded_message_count={loaded_msg_count}, retrieved_chunk_count={len(results)}, "
                f"retrieval_latency={retrieval_latency:.2f}ms, generation_latency={generation_latency:.2f}ms, "
                f"total_latency={total_latency:.2f}ms, prompt_size={len(prompt)}, memory_size={memory_tokens}"
            )

            doc_names = list(set(r["metadata"]["filename"] for r in results if r.get("metadata", {}).get("filename")))
            scores = [r["score"] for r in results]

            if workspace_id:
                background_tasks.add_task(
                    record_rag_search_history,
                    current_user.id,
                    workspace_id,
                    query_str,
                    retrieval_latency,
                    generation_latency,
                    total_latency,
                    len(results),
                    doc_names,
                    scores,
                    "Success"
                )

            return SearchResponse(
                query=query_str,
                results=search_results,
                citations=citations,
                answer=answer,
                message=None
            )
        except Exception as gemini_e:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
            total_latency = (time.perf_counter() - start_time) * 1000
            doc_names = list(set(r["metadata"]["filename"] for r in results if r.get("metadata", {}).get("filename")))
            scores = [r["score"] for r in results]
            if workspace_id:
                background_tasks.add_task(
                    record_rag_search_history,
                    current_user.id,
                    workspace_id,
                    query_str,
                    retrieval_latency,
                    generation_latency,
                    total_latency,
                    len(results),
                    doc_names,
                    scores,
                    "Failure"
                )
            raise gemini_e

    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        total_latency = (time.perf_counter() - start_time) * 1000
        if workspace_id:
            background_tasks.add_task(
                record_rag_search_history,
                current_user.id,
                workspace_id,
                query_str,
                retrieval_latency,
                generation_latency,
                total_latency,
                len(results),
                list(set(r["metadata"]["filename"] for r in results if r.get("metadata", {}).get("filename"))),
                [r["score"] for r in results],
                "Failure"
            )
        logger.error(f"Error during collection-scoped search: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search retrieval failure")
