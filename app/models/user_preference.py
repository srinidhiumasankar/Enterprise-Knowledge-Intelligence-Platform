# app/models/user_preference.py
# -----------------------------
# SQLAlchemy ORM model for storing User preference and UI settings.

from typing import Optional
from sqlalchemy import Integer, String, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin
from app.models.user import User


class UserPreference(Base, TimestampMixin):
    """
    SQLAlchemy ORM model representing configuration settings and preferences of a user.
    """
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    theme: Mapped[str] = mapped_column(String(50), default="light", nullable=False)
    default_workspace: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    preferred_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top_k: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="preference")


# Inject bidirectional relationship on User model dynamically without modifying app/models/user.py
User.preference = relationship(
    "UserPreference",
    back_populates="user",
    uselist=False,
    cascade="all, delete-orphan",
)
