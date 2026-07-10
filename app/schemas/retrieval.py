# app/schemas/retrieval.py
# ------------------------
# Pydantic schemas validating scoped document collection search payloads.

from typing import List, Optional
from pydantic import BaseModel, Field


class CollectionRetrievalRequest(BaseModel):
    """
    Validation request schema for collection-scoped semantic retrieval search.
    """
    query: str = Field(..., description="Semantic search query text")
    collection_ids: Optional[List[int]] = Field(None, description="Optional document collection IDs to filter search scope")
    workspace_id: Optional[int] = Field(None, description="Target workspace ID boundary filter")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="Max matching documents list size to return")
    conversation_id: Optional[int] = Field(None, description="Active conversation ID context for loading recent chat history.")
