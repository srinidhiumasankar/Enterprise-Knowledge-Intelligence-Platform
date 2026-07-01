# app/api/search.py
# -----------------
# Router implementing search API endpoints.

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user, get_retrieval_service
from app.models.user import User
from app.schemas.search import SearchRequest, SearchResponse
from app.services.retrieval_service import RetrievalService

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
) -> Any:
    """
    Generate query embedding, perform vector similarity search, filter by user ownership,
    and return ranked results.
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
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=query_str,
            top_k=request_data.top_k,
        )
        
        message = None
        if not results:
            message = "No relevant information found."

        return SearchResponse(
            query=query_str,
            results=results,
            message=message
        )
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
