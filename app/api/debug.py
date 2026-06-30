# app/api/debug.py
# ----------------
# Debugging / verification router for Phase 6.2.

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.embeddings.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["Debug / Verification"])


class DebugEmbedRequest(BaseModel):
    """Request schema for testing text embedding generation."""
    text: str = Field(..., description="Text input to embed", min_length=1)


class DebugEmbedResponse(BaseModel):
    """Response schema containing details and a subset of the generated vector."""
    dimension: int = Field(..., description="Dimension of the embedding vector")
    embedding_preview: List[float] = Field(..., description="First 10 values of the generated embedding")


def get_embedding_service() -> EmbeddingService:
    """Dependency injection generator for EmbeddingService."""
    return EmbeddingService()


@router.post(
    "/embed",
    response_model=DebugEmbedResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Debug Embedding Preview",
    description="Generate vector embeddings for a given text and return the first 10 dimensions for verification.",
)
async def generate_debug_embedding(
    request_data: DebugEmbedRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> DebugEmbedResponse:
    """
    Validate inputs, request single-vector embedding from the EmbeddingService,
    extract a 10-dimension preview list, and return metadata to the requester.
    """
    # Enforce non-whitespace check
    if not request_data.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty or whitespace-only."
        )

    try:
        logger.info("Debug endpoint requested embedding generation.")
        # Ensure model is initialized (handles loading failures inside service)
        embedding_service.load_model()
        embedding = embedding_service.generate_embedding(request_data.text)
        preview = embedding[:10]
        
        return DebugEmbedResponse(
            dimension=len(embedding),
            embedding_preview=preview
        )
    except ValueError as ve:
        logger.warning(f"Validation failure in debug embed endpoint: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Internal error generating debug embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )
