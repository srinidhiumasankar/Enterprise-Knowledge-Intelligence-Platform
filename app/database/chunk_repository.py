# app/database/chunk_repository.py
# ---------------------------------
# Data access layer for the DocumentChunk model.

from typing import Sequence
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


class ChunkRepository:
    """
    Repository class encapsulating database operations for the DocumentChunk model.
    """

    def __init__(self, db: Session):
        """
        Initialize the repository with a database session instance.
        """
        self.db = db

    def create_chunk(
        self,
        document_id: int,
        chunk_number: int,
        chunk_text: str,
        character_count: int,
    ) -> DocumentChunk:
        """
        Record a new document chunk in the database.
        """
        db_chunk = DocumentChunk(
            document_id=document_id,
            chunk_number=chunk_number,
            chunk_text=chunk_text,
            character_count=character_count,
        )
        self.db.add(db_chunk)
        self.db.commit()
        self.db.refresh(db_chunk)
        return db_chunk

    def create_chunks_bulk(self, chunks_data: list[dict]) -> None:
        """
        Bulk save list of chunks dict.
        """
        db_chunks = [
            DocumentChunk(
                document_id=item["document_id"],
                chunk_number=item["chunk_number"],
                chunk_text=item["chunk_text"],
                character_count=item["character_count"],
            )
            for item in chunks_data
        ]
        self.db.add_all(db_chunks)
        self.db.commit()

    def get_document_chunks(self, document_id: int) -> Sequence[DocumentChunk]:
        """
        Retrieve all chunks associated with a specific document ID.
        """
        return self.db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_number.asc())
        ).all()

    def delete_document_chunks(self, document_id: int) -> None:
        """
        Delete all text chunks associated with a specific document ID.
        """
        self.db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        self.db.commit()
