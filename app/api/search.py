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
    logger.info(f"User '{current_user.email}' requested semantic search for query '{request_data.query}' with top_k={request_data.top_k}")
    try:
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=request_data.query,
            top_k=request_data.top_k,
        )
        return SearchResponse(
            query=request_data.query,
            results=results,
        )
    except Exception as e:
        logger.error(f"Semantic search endpoint error for user '{current_user.email}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {str(e)}",
        )
