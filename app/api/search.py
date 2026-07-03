# app/api/search.py
# -----------------
# Router implementing search API endpoints.

import logging
import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user, get_retrieval_service, get_gemini_service
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
    current_user: User = Depends(get_current_active_user),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
) -> Any:
    """
    Generate query embedding, perform vector similarity search, filter by user ownership,
    and return ranked results with LLM generated answer and structured citations.
    """
    query_str = request_data.query
    logger.info(f"User '{current_user.email}' requested semantic search for query '{query_str}' with top_k={request_data.top_k}")
    
    # Reject empty or whitespace-only query
    if not query_str or not query_str.strip():
        logger.warning(f"Rejecting empty or whitespace search query from user '{current_user.email}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace-only.",
        )

    try:
        # Retrieve chunks
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=query_str,
            top_k=request_data.top_k,
        )
        
        # Log retrieved chunk count
        logger.info(f"Retrieved {len(results)} chunks for query: '{query_str}'")

        if not results:
            logger.info("No chunks retrieved. Skipping Gemini call and returning default fallback answer.")
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
            latency = time.perf_counter() - gemini_start
            logger.info(f"Gemini API response received in {latency:.4f}s")
            
            logger.info("Final RAG response constructed and returned")
            return SearchResponse(
                query=query_str,
                results=results,
                answer=answer,
                citations=citations,
                message=None
            )
        except GeminiQuotaExceededError as qee:
            logger.error(f"Gemini quota exceeded error: {qee}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini API rate limit or quota exceeded.",
            )
        except GeminiConfigurationError as gce:
            logger.error(f"Gemini configuration error: {gce}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini service configuration error: {str(gce)}",
            )
        except GeminiTimeoutError as gte:
            logger.error(f"Gemini timeout error: {gte}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Gemini API request timed out.",
            )
        except GeminiAPIError as gae:
            logger.error(f"Gemini API error: {gae}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini API error: {str(gae)}",
            )
        except GeminiError as ge:
            logger.error(f"Gemini error occurred: {ge}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Gemini service error: {str(ge)}",
            )

    except HTTPException:
        # Re-raise FastAPIs HTTPExceptions to prevent them from being swallowed by general except blocks
        raise
    except ValueError as ve:
        logger.warning(f"Validation error in search for user '{current_user.email}': {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except RuntimeError as re:
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
        logger.error(f"Unexpected error in semantic search: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected retrieval error occurred: {str(e)}",
        )

