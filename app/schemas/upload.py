# app/schemas/upload.py
# ---------------------
# Pydantic validation schemas for document upload and metadata responses.

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    """
    Response schema returned after a successful file upload.
    """
    message: str = Field(default="Upload successful")
    document_id: int
    filename: str
    status: str


class DocumentResponse(BaseModel):
    """
    Detailed document metadata response schema.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    uuid: str
    filename: str
    stored_filename: str
    file_extension: str
    mime_type: str
    file_size: int
    file_path: str
    status: str
    owner_id: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """
    Response schema holding a list of document metadata responses.
    """
    documents: list[DocumentResponse]


class ProcessResponse(BaseModel):
    """
    Response schema returned after successfully extracting text from a document.
    """
    document_id: int
    status: str
    characters: int
    preview: str


class ChunkResponse(BaseModel):
    """
    Response schema returned after successfully chunking a document.
    """
    document_id: int
    total_chunks: int
    average_chunk_size: int
