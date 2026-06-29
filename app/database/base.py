# app/database/base.py
# --------------------
# Declarative Base for ORM models.

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    SQLAlchemy 2.0 Declarative Base.
    All application database models will inherit from this class.
    """
    pass
