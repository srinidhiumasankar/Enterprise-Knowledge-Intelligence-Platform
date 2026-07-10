# app/api/search.py
# -----------------
# Router implementing search API endpoints.

import logging
import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_retrieval_service, get_gemini_service, get_db
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResponse, Citation
from app.services.retrieval_service import RetrievalService
from app.ai.prompt_builder import PromptBuilder
from app.ai.gemini_service import (
    GeminiService,
    GeminiError,
    GeminiConfigurationError,
    GeminiQuotaExceededError,
    GeminiTimeoutError,
    GeminiAPIError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["Semantic Search"])


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search Chunks",
    description="Perform semantic search on chunks belonging to the authenticated user using vector embeddings.",
)
async def search_chunks(
    request_data: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
    db: Session = Depends(get_db)
) -> Any:
    """
    Generate query embedding, perform vector similarity search, filter by user ownership,
    and return ranked results with LLM generated answer and structured citations.
    """
    from app.services.workspace.workspace_service import WorkspaceService
    from app.utils.activity_logger import record_rag_search_history

    query_str = request_data.query
    logger.info(f"User '{current_user.email}' requested semantic search for query '{query_str}' with top_k={request_data.top_k}")
    
    # Reject empty or whitespace-only query
    if not query_str or not query_str.strip():
        logger.warning(f"Rejecting empty or whitespace search query from user '{current_user.email}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace-only.",
        )

    # Pre-fetch active workspace context to associate logging records
    ws_service = WorkspaceService(db)
    active_ws = ws_service.get_active_workspace(current_user.id)
    workspace_id = active_ws.id if active_ws else None

    # Track metrics
    start_time = time.perf_counter()
    retrieval_latency = 0.0
    generation_latency = 0.0
    results = []

    try:
        # 1. Retrieve chunks
        retrieval_start = time.perf_counter()
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=query_str,
            top_k=request_data.top_k,
            workspace_id=workspace_id
        )
        retrieval_latency = (time.perf_counter() - retrieval_start) * 1000
        
        # Log retrieved chunk count
        logger.info(f"Retrieved {len(results)} chunks for query: '{query_str}'")

        if not results:
            logger.info("No chunks retrieved. Skipping Gemini call and returning default fallback answer.")
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

        # Build prompt
        logger.info(f"Creating prompt for user query: '{query_str}'")
        prompt = PromptBuilder.build_prompt(question=query_str, chunks=results)

        # Call Gemini Service
        logger.info(f"Gemini API request started for query '{query_str}' using model 'gemini-2.5-flash'")
        gemini_start = time.perf_counter()
        
        try:
            answer = gemini_service.generate_answer(
                prompt=prompt,
                model_name="gemini-2.5-flash"
            )
            generation_latency = (time.perf_counter() - gemini_start) * 1000
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Gemini API response received in {generation_latency/1000:.4f}s")
            
            # Extract document names & similarity scores for granular logging
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
            
            logger.info("Final RAG response constructed and returned")
            return SearchResponse(
                query=query_str,
                results=results,
                answer=answer,
                citations=citations,
                message=None
            )
        except GeminiQuotaExceededError as qee:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
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
            logger.error(f"Gemini quota exceeded error: {qee}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini API rate limit or quota exceeded.",
            )
        except GeminiConfigurationError as gce:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
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
            logger.error(f"Gemini configuration error: {gce}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini service configuration error: {str(gce)}",
            )
        except GeminiTimeoutError as gte:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
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
            logger.error(f"Gemini timeout error: {gte}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini API request timed out.",
            )
        except GeminiAPIError as gae:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
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
            logger.error(f"Gemini API error: {gae}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API error: {str(gae)}",
            )
        except GeminiError as ge:
            generation_latency = (time.perf_counter() - gemini_start) * 1000
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
            logger.error(f"Gemini error occurred: {ge}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini service error: {str(ge)}",
            )

    except HTTPException:
        # Re-raise FastAPIs HTTPExceptions
        raise
    except ValueError as ve:
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
        logger.warning(f"Validation error in search for user '{current_user.email}': {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except RuntimeError as re:
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
        error_msg = str(re).lower()
        if "chromadb" in error_msg or "connection" in error_msg:
            logger.error(f"ChromaDB connection unavailable during search: {re}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector storage service is temporarily unavailable.",
            )
        elif "embedding" in error_msg or "model" in error_msg:
            logger.error(f"Embedding model failure during search: {re}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate query embeddings.",
            )
        else:
            logger.error(f"Semantic search retrieval error: {re}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Semantic retrieval failed: {str(re)}",
            )
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
        logger.error(f"Unexpected error in semantic search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected retrieval error occurred: {str(e)}",
        )

