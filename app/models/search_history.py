# app/models/search_history.py
# ----------------------------
# SQLAlchemy ORM model for storing search query history.

from typing import Any, Dict, Optional
from datetime import datetime
from sqlalchemy import Integer, String, ForeignKey, DateTime, func, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.user import User


class SearchHistory(Base):
    """
    SQLAlchemy ORM model representing users search operations history.
    """
    __tablename__ = "search_histories"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    filters_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    result_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="search_histories")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="search_history")


# Inject bidirectional relationship on User model dynamically without modifying app/models/user.py
User.search_histories = relationship(
    "SearchHistory",
    back_populates="user",
    cascade="all, delete-orphan",
)
