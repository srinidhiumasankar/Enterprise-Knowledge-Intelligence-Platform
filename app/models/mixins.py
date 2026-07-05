# app/models/mixins.py
# --------------------
# Reusable SQLAlchemy model mixins for handling timestamps and soft deletes.

from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

Optional_datetime = Optional[datetime]


class TimestampMixin:
    """
    Mixin adding indexed created_at and updated_at datetime fields.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Mixin adding deleted_at datetime field to support soft-deletion of records.
    """
    deleted_at: Mapped[Optional_datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
