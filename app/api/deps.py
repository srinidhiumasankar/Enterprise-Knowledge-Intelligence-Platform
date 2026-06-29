# app/api/deps.py
# ----------------
# Reusable dependencies for FastAPI endpoints.

import logging
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.database import SessionLocal, UserRepository
from app.models.user import User
from app.utils.security import decode_token

# Setup logger
logger = logging.getLogger(__name__)

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_db() -> Generator[Session, None, None]:
    """
    Yields database session local instance and closes it on completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Dependency to validate JWT access token and return the authenticated User.
    Raises 401 Unauthorized if invalid.
    """
    try:
        payload = decode_token(token)
        user_uuid: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if not user_uuid or token_type != "access":
            logger.warning("Token verification failed: missing subject or invalid token type")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError as e:
        logger.warning(f"Token decoding failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_repo = UserRepository(db)
    user = user_repo.get_by_uuid(user_uuid)
    if not user:
        logger.warning(f"User UUID '{user_uuid}' parsed from token does not exist in DB")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to verify that the authenticated User is active.
    Raises 403 Forbidden if inactive.
    """
    if not current_user.is_active:
        logger.warning(f"Active user check failed: User '{current_user.email}' is inactive")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user
