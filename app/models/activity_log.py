# app/models/activity_log.py
# --------------------------
# SQLAlchemy ORM model for storing general workspace events and activity logs.

from datetime import datetime
from sqlalchemy import Integer, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ActivityLog(Base):
    """
    SQLAlchemy ORM model representing users dashboard/workspace recent activities.
    """
    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    workspace_id: Mapped[int] = mapped_column(Integer, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
