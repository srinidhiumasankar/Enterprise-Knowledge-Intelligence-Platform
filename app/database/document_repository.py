# app/database/document_repository.py
# ------------------------------------
# Data access layer for the Document model.

from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    """
    Repository class encapsulating database operations for the Document model.
    """

    def __init__(self, db: Session):
        """
        Initialize the repository with a database session instance.
        """
        self.db = db

    def create_document(
        self,
        filename: str,
        stored_filename: str,
        file_extension: str,
        mime_type: str,
        file_size: int,
        file_path: str,
        owner_id: int,
    ) -> Document:
        """
        Record a new document upload in the database.
        """
        db_doc = Document(
            filename=filename,
            stored_filename=stored_filename,
            file_extension=file_extension,
            mime_type=mime_type,
            file_size=file_size,
            file_path=file_path,
            owner_id=owner_id,
            status="uploaded",
        )
        self.db.add(db_doc)
        self.db.commit()
        self.db.refresh(db_doc)
        return db_doc

    def get_document(self, document_id: int) -> Document | None:
        """
        Retrieve a single document by its integer ID.
        """
        return self.db.get(Document, document_id)

    def get_user_documents(
        self, owner_id: int, skip: int = 0, limit: int = 100
    ) -> Sequence[Document]:
        """
        Retrieve all documents belonging to a specific user (with pagination).
        """
        return self.db.scalars(
            select(Document)
            .where(Document.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
        ).all()

    def delete_document(self, db_document: Document) -> None:
        """
        Delete a document record from the database.
        """
        self.db.delete(db_document)
        self.db.commit()
