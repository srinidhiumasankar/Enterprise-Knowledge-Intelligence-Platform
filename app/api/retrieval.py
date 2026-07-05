# app/api/retrieval.py
# --------------------
# Router implementing search API endpoints for collection-scoped semantic retrieval.

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user, get_retrieval_service, get_gemini_service
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
    current_user: User = Depends(get_current_active_user),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    gemini_service: GeminiService = Depends(get_gemini_service),
) -> Any:
    """
    Perform semantic vector query filtered to a specific set of document collections or workspace boundary.
    """
    query_str = request_data.query
    logger.info(f"User '{current_user.email}' requested collection-scoped semantic search for query '{query_str}'")

    if not query_str or not query_str.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty or whitespace-only.",
        )

    try:
        # Retrieve chunks with collection and workspace filter parameters
        results = await retrieval_service.retrieve(
            user_id=current_user.id,
            query=query_str,
            top_k=request_data.top_k or 5,
            collection_ids=request_data.collection_ids,
            workspace_id=request_data.workspace_id
        )

        logger.info(f"Retrieved {len(results)} chunks for collection-scoped query: '{query_str}'")

        if not results:
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

        # Build prompt
        prompt = PromptBuilder.build_prompt(question=query_str, chunks=results)

        # Call Gemini Service to generate answer
        answer = gemini_service.generate_answer(prompt=prompt, model_name="gemini-2.5-flash")

        return SearchResponse(
            query=query_str,
            results=search_results,
            citations=citations,
            answer=answer,
            message=None
        )

    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error during collection-scoped search: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search retrieval failure")
