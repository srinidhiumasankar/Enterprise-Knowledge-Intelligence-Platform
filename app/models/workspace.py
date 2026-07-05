# app/models/workspace.py
# ----------------------
# SQLAlchemy ORM model for Workspace representing collaborative working directories.

import uuid
from typing import List, Optional
from sqlalchemy import String, Integer, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin
from app.models.user import User


class Workspace(Base, TimestampMixin):
    """
    SQLAlchemy ORM model representing workspaces which partition documents,
    conversations, search history, and collections.
    """
    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    uuid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        default=lambda: str(uuid.uuid4()),
    )
    owner_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="workspaces")
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    collections: Mapped[List["Collection"]] = relationship(
        "Collection",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    search_history: Mapped[List["SearchHistory"]] = relationship(
        "SearchHistory",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )


# Inject bidirectional relationship on User model dynamically without modifying app/models/user.py
User.workspaces = relationship(
    "Workspace",
    back_populates="owner",
    cascade="all, delete-orphan",
)
