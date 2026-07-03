# app/schemas/search.py
# ---------------------
# Pydantic validation schemas for semantic search.

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    """
    Request payload schema for semantic search.
    """
    query: str = Field(..., description="Query string for semantic search")
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


class Citation(BaseModel):
    """
    Schema for a structured source citation.
    """
    document_id: int = Field(..., description="ID of the document")
    filename: str = Field(..., description="Name of the source document file")
    chunk_id: int = Field(..., description="Database primary ID of the chunk")
    score: float = Field(..., description="Similarity score calculated from distance")


class SearchResponse(BaseModel):
    """
    Response schema for the semantic search endpoint.
    """
    query: str = Field(..., description="The queried text")
    results: List[SearchResult] = Field(..., description="Ranked list of matching chunks")
    answer: Optional[str] = None
    citations: List[Citation] = Field(default=[], description="Structured citations of the source chunks used for the answer")
    message: Optional[str] = Field(default=None, description="Optional system message indicating search status")
