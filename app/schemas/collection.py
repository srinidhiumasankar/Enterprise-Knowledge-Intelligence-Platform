# app/schemas/collection.py
# -----------------------
# Pydantic validation schemas for document collection organization.

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    """
    Schema for creating a new document collection.
    """
    name: str = Field(..., min_length=1, max_length=255, description="Unique collection name within the workspace")
    description: Optional[str] = Field(None, description="Optional text description of the collection")
    workspace_id: Optional[int] = Field(None, description="Target workspace ID, defaults to user's active workspace")


class CollectionUpdate(BaseModel):
    """
    Schema for modifying metadata of an existing collection.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated collection name")
    description: Optional[str] = Field(None, description="Updated description text")


class CollectionResponse(BaseModel):
    """
    Detailed serialization schema for document collections.
    """
    id: int
    uuid: str
    workspace_id: int
    owner_id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class CollectionSummary(BaseModel):
    """
    Summarized serialization schema for collections list results.
    """
    id: int
    uuid: str
    workspace_id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class CollectionListResponse(BaseModel):
    """
    Wrapper schema for paginated collections listings.
    """
    items: List[CollectionSummary]
    page: int
    page_size: int
    total_records: int
    total_pages: int


class AddDocumentRequest(BaseModel):
    """
    Payload for adding an uploaded document to the collection.
    """
    document_id: int = Field(..., description="The database ID of the document to link")


class RemoveDocumentRequest(BaseModel):
    """
    Payload for removing a document linkage from the collection.
    """
    document_id: int = Field(..., description="The database ID of the document to unlink")


class CollectionStatistics(BaseModel):
    """
    Metrics and metadata details for a collection.
    """
    document_count: int = Field(..., description="Total documents currently inside the collection")
    total_size: int = Field(..., description="Aggregate size of all documents in bytes")
    first_uploaded: Optional[datetime] = Field(None, description="Creation time of the oldest document in collection")
    last_uploaded: Optional[datetime] = Field(None, description="Creation time of the newest document in collection")
    last_modified: datetime = Field(..., description="Timestamp of the collection's last update")
    owner_email: str = Field(..., description="Email address of the collection owner")
    workspace_name: str = Field(..., description="Name of the partitioned workspace")
