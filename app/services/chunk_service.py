# app/services/chunk_service.py
# ----------------------------
# Business logic service layer for dividing document text into chunks and database persistence.

import logging
from typing import Sequence, Optional
from fastapi import HTTPException, status

from app.database.chunk_repository import ChunkRepository
from app.database import DocumentRepository
from app.models.document_chunk import DocumentChunk
from app.services.document_processor import DocumentProcessorService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """
    Split text into chunks of maximum chunk_size with chunk_overlap, preserving paragraphs.
    """
    if not text or not text.strip():
        return []

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        # Handle oversized paragraphs
        if para_len > chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            start = 0
            while start < para_len:
                end = min(start + chunk_size, para_len)
                chunk_part = para[start:end].strip()
                if chunk_part:
                    chunks.append(chunk_part)
                start += chunk_size - chunk_overlap
            continue

        # Check if paragraph fits in current chunk
        if current_length + para_len + (2 if current_chunk else 0) > chunk_size:
            chunks.append("\n\n".join(current_chunk))

            # Create overlap from ending paragraphs of current chunk
            overlap_chunk = []
            overlap_len = 0
            for prev_para in reversed(current_chunk):
                if overlap_len + len(prev_para) + (2 if overlap_chunk else 0) <= chunk_overlap:
                    overlap_chunk.insert(0, prev_para)
                    overlap_len += len(prev_para) + (2 if overlap_chunk else 0)
                else:
                    break
            current_chunk = overlap_chunk
            current_length = overlap_len

        current_chunk.append(para)
        current_length += para_len + (2 if len(current_chunk) > 1 else 0)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Trim and filter empty chunks
    cleaned_chunks = [c.strip() for c in chunks if c.strip()]
    return cleaned_chunks


class ChunkService:
    """
    Service class responsible for splitting document text into chunks and persisting them.
    """

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository,
        embedding_service: Optional[EmbeddingService] = None,
        vector_service: Optional[VectorService] = None,
    ):
        """
        Initialize the service with repository and service dependencies.
        """
        self.repository = chunk_repository
        self.doc_repository = document_repository
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or VectorService()

    def chunk_document(self, document_id: int, owner_id: int) -> dict:
        """
        Retrieve document text, split it into chunks, save to DB, and return stats.
        Raises HTTPException on failure.
        """
        doc = self.doc_repository.get_document(document_id)
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

        # 1. Check if document has text
        processor = DocumentProcessorService(self.doc_repository)
        try:
            cleaned_text = processor.get_document_text(document_id, owner_id)
        except Exception as e:
            logger.error(f"Cannot chunk document {document_id} as text extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot chunk document. Text extraction must succeed first.",
            )

        # 2. Delete any existing chunks for this document to ensure idempotency
        self.repository.delete_document_chunks(document_id)

        # 3. Split into chunks
        chunks = split_text(cleaned_text)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Extracted text is empty. No chunks generated.",
            )

        # 4. Save to DB in bulk
        chunks_data = []
        total_chars = 0
        for i, chunk_text in enumerate(chunks, start=1):
            char_count = len(chunk_text)
            total_chars += char_count
            chunks_data.append({
                "document_id": document_id,
                "chunk_number": i,
                "chunk_text": chunk_text,
                "character_count": char_count
            })

        self.repository.create_chunks_bulk(chunks_data)
        logger.info(f"Successfully split document {document_id} into {len(chunks)} chunks.")

        # 4.5. Generate embeddings and persist to ChromaDB
        try:
            # Retrieve chunks to get their uuids
            db_chunks = self.repository.get_document_chunks(document_id)
            chunk_texts = [c.chunk_text for c in db_chunks]
            chunk_ids = [c.uuid for c in db_chunks]
            chunk_indices = [c.chunk_number for c in db_chunks]

            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks of document_id: {document_id}")
            embeddings = self.embedding_service.embed_documents(chunk_texts)

            logger.info(f"Storing embeddings in ChromaDB for document_id: {document_id}")
            self.vector_service.insert_vectors(
                document_id=document_id,
                chunk_ids=chunk_ids,
                chunk_indices=chunk_indices,
                texts=chunk_texts,
                embeddings=embeddings,
            )
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error persisting vectors to ChromaDB for document {document_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate or persist vector embeddings: {str(e)}",
            )

        # 5. Update document status to indexed
        doc.status = "indexed"
        self.doc_repository.db.add(doc)
        self.doc_repository.db.commit()

        avg_size = int(total_chars / len(chunks)) if chunks else 0

        return {
            "document_id": document_id,
            "total_chunks": len(chunks),
            "average_chunk_size": avg_size
        }

    def get_chunks(self, document_id: int, owner_id: int) -> Sequence[DocumentChunk]:
        """
        Retrieve all stored database chunks for a document after checking ownership.
        """
        doc = self.doc_repository.get_document(document_id)
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
        return self.repository.get_document_chunks(document_id)
