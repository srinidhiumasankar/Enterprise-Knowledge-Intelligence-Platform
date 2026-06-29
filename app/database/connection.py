# app/database/connection.py
# --------------------------
# Database connection engine and session factory.

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings

# Determine if we're using SQLite to configure connect_args
is_sqlite = settings.DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency generator for database sessions.
    Yields a database session and guarantees its closure on completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
