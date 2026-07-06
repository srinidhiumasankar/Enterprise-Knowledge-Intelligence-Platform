# app/schemas/__init__.py
# Marks the `schemas` directory as a Python package.
# Pydantic request/response schemas will be defined here.

from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.schemas.token import Token, TokenPayload, LoginRequest, LoginResponse
from app.schemas.upload import UploadResponse, DocumentResponse, DocumentListResponse, ProcessResponse, ChunkResponse
from app.schemas.search import SearchRequest, SearchResult, SearchResponse
from app.schemas.retrieval import CollectionRetrievalRequest
from app.schemas.collection import CollectionCreate, CollectionUpdate, CollectionResponse, CollectionSummary, CollectionListResponse
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationListResponse, ConversationSummary
from app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceSummary,
    WorkspaceListResponse,
    WorkspaceStatistics,
)
from app.schemas.search_history import (
    SearchHistoryResponse,
    SearchHistoryListResponse,
    RecentSearchResponse,
    FrequentSearchResponse,
    SearchStatisticsResponse,
    DeleteHistoryResponse,
)
from app.schemas.dashboard import (
    DashboardResponse,
    WorkspaceOverview,
    ConversationMetrics,
    CollectionMetrics,
    SearchMetrics,
    DocumentMetrics,
    StorageMetrics,
    RecentActivity,
    DashboardMetricsResponse,
)


