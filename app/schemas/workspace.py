# app/schemas/workspace.py
# ----------------------
# Pydantic schemas validating workspace configuration, updates, and metrics reporting.

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    """
    Schema for creating a new Workspace.
    """
    name: str = Field(..., min_length=1, max_length=255, description="Unique workspace name per user")
    description: Optional[str] = Field(None, max_length=500, description="Optional text description of the workspace")


class WorkspaceUpdate(BaseModel):
    """
    Schema for updating workspace metadata or status.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated workspace name")
    description: Optional[str] = Field(None, max_length=500, description="Updated workspace description")
    is_active: Optional[bool] = Field(None, description="Toggle active status of the workspace")


class WorkspaceResponse(BaseModel):
    """
    Detailed workspace serialization schema.
    """
    id: int
    uuid: str
    owner_id: int
    name: str
    description: Optional[str]
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }


class WorkspaceSummary(BaseModel):
    """
    Summarized serialization schema for listing workspaces.
    """
    id: int
    uuid: str
    name: str
    description: Optional[str]
    is_default: bool
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class WorkspaceListResponse(BaseModel):
    """
    Paginated workspace response structure.
    """
    items: List[WorkspaceSummary]
    page: int
    page_size: int
    total_records: int
    total_pages: int


class WorkspaceStatistics(BaseModel):
    """
    Metrics details for a workspace.
    """
    workspace_id: int = Field(..., description="Target workspace ID")
    document_count: int = Field(..., description="Total documents stored inside workspace")
    collection_count: int = Field(..., description="Total collections established in workspace")
    conversation_count: int = Field(..., description="Total conversations in workspace")
    search_count: int = Field(..., description="Total search history records in workspace")
    storage_usage: int = Field(..., description="Cumulative file size of documents in bytes")
    created_at: datetime = Field(..., description="Timestamp of workspace creation")
    last_activity: datetime = Field(..., description="Timestamp of most recent activity in workspace")
