# app/schemas/dashboard.py
# ------------------------
# Pydantic schemas for structural validation of Enterprise Dashboard statistics.

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class WorkspaceOverview(BaseModel):
    """
    Schema for workspace overview details.
    """
    workspace_name: str
    owner_name: str
    created_at: datetime
    last_activity_at: Optional[datetime] = None


class ConversationMetrics(BaseModel):
    """
    Schema for conversation metrics.
    """
    total_conversations: int
    active_conversations: int
    archived_conversations: int
    total_messages: int
    average_conversation_length: float


class CollectionMetrics(BaseModel):
    """
    Schema for collections metrics.
    """
    total_collections: int
    largest_collection_name: Optional[str] = None
    largest_collection_size: int = 0
    average_documents_per_collection: float


class SearchMetrics(BaseModel):
    """
    Schema for search query metrics.
    """
    searches_today: int
    searches_this_week: int
    most_frequent_query: Optional[str] = None
    average_retrieval_time_ms: float


class DocumentMetrics(BaseModel):
    """
    Schema for document upload metrics.
    """
    uploaded_today: int
    uploaded_this_week: int
    total_documents: int
    total_chunks: int
    total_embeddings: int


class StorageMetrics(BaseModel):
    """
    Schema for storage usage details.
    """
    total_storage_bytes: int
    average_document_size_bytes: float
    largest_document_name: Optional[str] = None
    largest_document_size_bytes: int = 0
    total_embeddings: int
    vector_db_size_bytes: int = 0


class ActivityItem(BaseModel):
    """
    Schema representing a single entry in the recent activity timeline.
    """
    type: str  # 'document_upload', 'conversation_start', 'search_query', 'collection_update'
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class RecentActivity(BaseModel):
    """
    Schema for recent activity timeline collection.
    """
    items: List[ActivityItem]


class DashboardResponse(BaseModel):
    """
    Combined metrics and details for workspace dashboard dashboard.
    """
    overview: WorkspaceOverview
    conversation_metrics: ConversationMetrics
    collection_metrics: CollectionMetrics
    search_metrics: SearchMetrics
    document_metrics: DocumentMetrics
    storage_metrics: StorageMetrics
    recent_activity: List[ActivityItem]


class DashboardMetricsResponse(BaseModel):
    """
    Schema representing the workspace metrics payload response.
    """
    conversations: ConversationMetrics
    collections: CollectionMetrics
    searches: SearchMetrics
    documents: DocumentMetrics
    storage: StorageMetrics
