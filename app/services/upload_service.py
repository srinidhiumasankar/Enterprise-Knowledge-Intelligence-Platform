# app/services/upload_service.py
# ------------------------------
# Business logic service layer for validating, saving, and deleting uploaded documents.

import logging
import uuid
from pathlib import Path
from fastapi import HTTPException, status, UploadFile

from app.database import DocumentRepository
from app.models.document import Document
from app.utils.file_utils import (
    is_allowed_file,
    ensure_upload_directories,
    get_upload_path,
    MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)


class UploadService:
    """
    Service class orchestrating file saving, validation, folder creation,
    and metadata persistence through DocumentRepository.
    """

    def __init__(self, repository: DocumentRepository):
        """
        Initialize the service with a DocumentRepository instance.
        """
        self.repository = repository

    async def save_upload(
        self, file: UploadFile, owner_id: int, base_upload_dir: str = "uploads"
    ) -> Document:
        """
        Validate, save to disk, and store metadata of an uploaded file.
        """
        filename = file.filename
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename cannot be empty",
            )

        # 1. Validate file extension
        if not is_allowed_file(filename):
            logger.warning(f"Upload rejected: forbidden extension for filename '{filename}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File type not supported. Allowed formats: .pdf, .docx, .txt",
            )

        # 2. Validate file size (before saving)
        file_size = file.size
        if file_size is None:
            try:
                file.file.seek(0, 2)
                file_size = file.file.tell()
                file.file.seek(0)
            except Exception as e:
                logger.error(f"Error checking file size for '{filename}': {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not check file size",
                )

        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(
                f"Upload rejected: file '{filename}' size ({file_size} bytes) exceeds limit"
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds maximum size of 20 MB",
            )

        # 3. Ensure subfolders exist
        ensure_upload_directories(base_upload_dir)

        # 4. Generate unique UUID filename
        file_extension = Path(filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        stored_filename = f"{unique_id}{file_extension}"

        # 5. Determine destination path
        dest_folder = get_upload_path(filename, base_upload_dir)
        dest_path = dest_folder / stored_filename

        # 6. Save the file to disk
        try:
            with open(dest_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    f.write(chunk)
            logger.info(f"File successfully saved to disk at '{dest_path}'")
        except Exception as e:
            logger.error(f"Failed to save file '{filename}' to disk: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save file to disk",
            )

        # 7. Create DB record
        try:
            db_doc = self.repository.create_document(
                filename=filename,
                stored_filename=stored_filename,
                file_extension=file_extension,
                mime_type=file.content_type or "application/octet-stream",
                file_size=file_size,
                file_path=str(dest_path.resolve()),
                owner_id=owner_id,
            )
            logger.info(f"Document metadata record saved in DB (ID: {db_doc.id}, UUID: {db_doc.uuid})")
            return db_doc
        except Exception as e:
            # Rollback file creation if database write fails to keep system clean
            if dest_path.exists():
                dest_path.unlink()
            logger.error(f"Failed to write metadata record to DB for '{filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error while saving document metadata",
            )

    def get_document_metadata(self, document_id: int, owner_id: int) -> Document:
        """
        Retrieve document metadata. Validates ownership.
        """
        doc = self.repository.get_document(document_id)
        if not doc:
            logger.warning(f"Retrieve failure: Document '{document_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        if doc.owner_id != owner_id:
            logger.warning(f"Unauthorized access check failed: User '{owner_id}' requested Document '{document_id}'")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document",
            )
        return doc

    def delete_document(self, document_id: int, owner_id: int) -> None:
        """
        Delete database record and associated physical file from disk.
        """
        doc = self.get_current_user_document_or_raise(document_id, owner_id)
        
        # 1. Delete physical file
        file_path = Path(doc.file_path)
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Physical file deleted at '{file_path}'")
            else:
                logger.warning(f"File not found on disk at '{file_path}' during deletion")
        except Exception as e:
            logger.error(f"Error deleting file '{file_path}': {e}")

        # 1.5. Delete vectors from ChromaDB
        try:
            from app.embeddings.chroma_service import ChromaService
            chroma_service = ChromaService()
            chroma_service.delete_document(document_id)
        except Exception as e:
            logger.error(f"Error cleaning up ChromaDB vectors for document_id {document_id}: {e}")

        # 2. Delete database record
        try:
            self.repository.delete_document(doc)
            logger.info(f"Document metadata record ID '{document_id}' removed from DB")
        except Exception as e:
            logger.error(f"Database error deleting document record '{document_id}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document metadata record",
            )

    def get_current_user_document_or_raise(self, document_id: int, owner_id: int) -> Document:
        """
        Helper method to retrieve document and assert ownership.
        """
        return self.get_document_metadata(document_id, owner_id)
