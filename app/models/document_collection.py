# app/models/document_collection.py
# ---------------------------------
# SQLAlchemy ORM model mapping document-collection relationships (many-to-many bridge).

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class DocumentCollection(Base):
    """
    Bridge table connecting Document records to Collection records in a many-to-many relationship.
    """
    __tablename__ = "document_collections"

    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )
