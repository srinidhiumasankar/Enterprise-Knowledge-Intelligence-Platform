# app/schemas/search_history.py
# -----------------------------
# Pydantic schemas validating payload and responses for Search History.

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchHistoryResponse(BaseModel):
    """
    Schema representing a single search history record entry.
    """
    id: int
    user_id: int
    workspace_id: int
    query: str
    filters_json: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None
    result_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SearchHistoryListResponse(BaseModel):
    """
    Schema for paginated list response of search history records.
    """
    items: List[SearchHistoryResponse]
    page: int
    page_size: int
    total_records: int
    total_pages: int


class RecentSearchResponse(BaseModel):
    """
    Schema representing a recent search query.
    """
    query: str
    created_at: datetime
    workspace_id: int

    class Config:
        from_attributes = True


class FrequentSearchResponse(BaseModel):
    """
    Schema representing a frequently searched query.
    """
    query: str
    count: int

    class Config:
        from_attributes = True


class SearchStatisticsResponse(BaseModel):
    """
    Schema detailing statistics metrics and aggregations for search history.
    """
    total_searches: int
    today_searches: int
    weekly_searches: int
    monthly_searches: int
    average_query_length: float
    average_latency_ms: float
    most_frequent_query: Optional[str] = None
    last_search_time: Optional[datetime] = None
    top_queries: List[FrequentSearchResponse] = []
    daily_query_trend: Dict[str, int] = Field(default_factory=dict)


class DeleteHistoryResponse(BaseModel):
    """
    Schema representing status of a deletion operations.
    """
    success: bool
    message: str
