# app/schemas/search.py
# ---------------------
# Pydantic validation schemas for semantic search.

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    """
    Request payload schema for semantic search.
    """
    query: str = Field(..., description="Query string for semantic search", min_length=1)
    top_k: int = Field(default=5, description="Number of top results to retrieve", ge=1)


class SearchResult(BaseModel):
    """
    Schema for a single semantic search result chunk.
    """
    document_id: int = Field(..., description="ID of the document")
    chunk_id: int = Field(..., description="Database primary ID of the chunk")
    score: float = Field(..., description="Similarity score calculated from distance")
    text: str = Field(..., description="Text content of the matching chunk")
    metadata: dict = Field(..., description="Metadata dictionary containing chunk details")


class SearchResponse(BaseModel):
    """
    Response schema for the semantic search endpoint.
    """
    query: str = Field(..., description="The queried text")
    results: List[SearchResult] = Field(..., description="Ranked list of matching chunks")
    message: Optional[str] = Field(default=None, description="Optional system message indicating search status")
