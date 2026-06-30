# app/services/document_processor.py
# ----------------------------------
# Business logic service layer for document text extraction and cleaning.

import logging
import os
import re
from fastapi import HTTPException, status

from app.database import DocumentRepository
from app.models.document import Document
from app.utils.pdf_reader import read_pdf
from app.utils.docx_reader import read_docx
from app.utils.text_reader import read_text

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean extracted text content:
    - Normalizes line endings to \n.
    - Replaces repeated spaces and tabs with a single space.
    - Reduces consecutive newlines to at most a single blank line.
    - Strips leading/trailing whitespaces.
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Collapse repeated non-newline spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Collapse multiple blank lines to a single blank line
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


class DocumentProcessorService:
    """
    Service class orchestrating text extraction from supported document formats.
    """

    def __init__(self, repository: DocumentRepository):
        """
        Initialize the service with a DocumentRepository instance.
        """
        self.repository = repository

    def extract_text(self, file_path: str, file_extension: str) -> str:
        """
        Detect file extension, invoke the correct reader, extract text, and clean it.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found on disk at: {file_path}")

        ext = file_extension.lower()
        
        logger.info(f"Extracting text from file '{file_path}' (extension: {ext})")
        if ext == ".pdf":
            raw_text = read_pdf(file_path)
        elif ext == ".docx":
            raw_text = read_docx(file_path)
        elif ext in (".txt", ".text"):
            raw_text = read_text(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Extracted text is empty or contains only whitespace")

        return cleaned

    def process_document(self, document_id: int, owner_id: int) -> str:
        """
        Process an uploaded document: fetch from DB, verify ownership, extract text,
        and update status. Returns the cleaned text.
        """
        doc = self.repository.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        if doc.owner_id != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document",
            )

        # Set status to processing
        doc.status = "processing"
        self.repository.db.add(doc)
        self.repository.db.commit()

        try:
            cleaned_text = self.extract_text(doc.file_path, doc.file_extension)
            return cleaned_text
        except FileNotFoundError as fnfe:
            doc.status = "failed"
            self.repository.db.add(doc)
            self.repository.db.commit()
            logger.error(f"Processing failed for Document {document_id}: File not found. {fnfe}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(fnfe),
            )
        except ValueError as ve:
            doc.status = "failed"
            self.repository.db.add(doc)
            self.repository.db.commit()
            logger.error(f"Processing failed for Document {document_id}: Validation error. {ve}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(ve),
            )
        except Exception as e:
            doc.status = "failed"
            self.repository.db.add(doc)
            self.repository.db.commit()
            logger.error(f"Unexpected error processing Document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text extraction failed: {str(e)}",
            )

    def get_document_text(self, document_id: int, owner_id: int) -> str:
        """
        Convenience method to retrieve document and extract its text without status changes.
        """
        doc = self.repository.get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
        if doc.owner_id != owner_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document",
            )

        try:
            return self.extract_text(doc.file_path, doc.file_extension)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract document text: {e}",
            )
