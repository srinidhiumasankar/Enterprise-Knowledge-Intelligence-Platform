# app/api/upload.py
# ------------------
# Router implementing upload endpoints.

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status, File
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.database import DocumentRepository, ChunkRepository
from app.models.user import User
from app.schemas.upload import UploadResponse, DocumentResponse, DocumentListResponse, ProcessResponse, ChunkResponse
from app.services import UploadService, DocumentProcessorService, ChunkService

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["Document Upload"])


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload a document file (PDF, DOCX, or TXT) up to 20 MB. Authenticated users only.",
)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Handle document upload, perform validation, save to disk, and write DB metadata.
    """
    logger.info(f"User '{current_user.email}' is uploading file '{file.filename}'")
    try:
        upload_service = UploadService(DocumentRepository(db))
        db_doc = await upload_service.save_upload(file=file, owner_id=current_user.id)
        logger.info(f"Upload success for file '{file.filename}' (Doc ID: {db_doc.id})")
        return UploadResponse(
            message="Upload successful",
            document_id=db_doc.id,
            filename=db_doc.filename,
            status=db_doc.status,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during file upload for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during upload",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve metadata for all documents uploaded by the authenticated user.",
)
def list_documents(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    List user's uploaded documents.
    """
    logger.info(f"Listing documents for user '{current_user.email}' (skip: {skip}, limit: {limit})")
    try:
        doc_repo = DocumentRepository(db)
        docs = doc_repo.get_user_documents(owner_id=current_user.id, skip=skip, limit=limit)
        return DocumentListResponse(documents=[DocumentResponse.model_validate(d) for d in docs])
    except Exception as e:
        logger.error(f"Unexpected error listing documents for '{current_user.email}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Document Details",
    description="Retrieve metadata details of a specific uploaded document if owned by the current user.",
)
def get_document_details(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Get document metadata.
    """
    logger.info(f"User '{current_user.email}' requested document metadata for ID '{document_id}'")
    try:
        upload_service = UploadService(DocumentRepository(db))
        doc = upload_service.get_document_metadata(document_id=document_id, owner_id=current_user.id)
        return DocumentResponse.model_validate(doc)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error retrieving document '{document_id}' metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete the physical document file from disk and its metadata record from the database.",
)
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Delete document.
    """
    logger.info(f"User '{current_user.email}' requested deletion of document ID '{document_id}'")
    try:
        upload_service = UploadService(DocumentRepository(db))
        upload_service.delete_document(document_id=document_id, owner_id=current_user.id)
        return {"message": "Document deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error deleting document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during deletion",
        )


@router.post(
    "/{document_id}/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Document",
    description="Extract and clean plain text from the uploaded document. Return character count and text preview.",
)
async def process_document_endpoint(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Find document, extract and clean its text, and return processing metadata.
    """
    logger.info(f"User '{current_user.email}' requested processing of document ID '{document_id}'")
    try:
        processor = DocumentProcessorService(DocumentRepository(db))
        extracted_text = processor.process_document(
            document_id=document_id, owner_id=current_user.id
        )

        return ProcessResponse(
            document_id=document_id,
            status="processed",
            characters=len(extracted_text),
            preview=extracted_text[:500],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.post(
    "/{document_id}/chunk",
    response_model=ChunkResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk Document",
    description="Divide document's extracted text into chunks and save them to the database.",
)
async def chunk_document_endpoint(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> Any:
    """
    Load document text, slice it into chunks, persist to DB, and return chunk statistics.
    """
    logger.info(f"User '{current_user.email}' requested chunking of document ID '{document_id}'")
    try:
        chunk_service = ChunkService(
            chunk_repository=ChunkRepository(db),
            document_repository=DocumentRepository(db)
        )
        result = chunk_service.chunk_document(
            document_id=document_id, owner_id=current_user.id
        )
        return ChunkResponse(**result)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error chunking document '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during chunking: {str(e)}",
        )
