# app/models/collection.py
# ------------------------
# SQLAlchemy ORM model for Collection representing custom document groupings.

import uuid
from typing import List, Optional
from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin
from app.models.user import User


class Collection(Base, TimestampMixin):
    """
    SQLAlchemy ORM model representing collection objects mapping many documents inside a workspace.
    """
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    uuid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    owner_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="collections")
    owner: Mapped["User"] = relationship("User", back_populates="collections")
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        secondary="document_collections",
        back_populates="collections",
    )


# Inject bidirectional relationship on User model dynamically without modifying app/models/user.py
User.collections = relationship(
    "Collection",
    back_populates="owner",
    cascade="all, delete-orphan",
)
