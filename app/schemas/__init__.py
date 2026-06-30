# app/schemas/__init__.py
# Marks the `schemas` directory as a Python package.
# Pydantic request/response schemas will be defined here.

from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.schemas.token import Token, TokenPayload, LoginRequest, LoginResponse
from app.schemas.upload import UploadResponse, DocumentResponse, DocumentListResponse, ProcessResponse, ChunkResponse
from app.schemas.search import SearchRequest, SearchResult, SearchResponse


