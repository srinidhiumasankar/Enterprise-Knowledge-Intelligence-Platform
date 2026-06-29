# app/utils/security.py
# ---------------------
# Cryptographic utilities for password hashing and JWT token processing.

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.config import settings

# Setup logger
logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against its bcrypt hash.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification encountered an error: {e}")
        return False


def create_access_token(subject: str | Any, expires_delta: timedelta = None) -> str:
    """
    Generate a JWT access token for a subject (usually user uuid).
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(subject: str | Any, expires_delta: timedelta = None) -> str:
    """
    Generate a JWT refresh token for a subject.
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )

    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token. Raises JWTError if invalid or expired.
    """
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError as e:
        logger.warning("JWT validation failed: Token signature has expired")
        raise e
    except JWTError as e:
        logger.warning(f"JWT validation failed: Invalid token - {e}")
        raise e


def verify_token(token: str) -> str | None:
    """
    Verify token validity and return the subject ('sub') claim if valid.
    Returns None if signature is invalid or expired.
    """
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except JWTError:
        return None


def get_token_expiry(token: str) -> datetime | None:
    """
    Retrieve the expiration datetime from a JWT token.
    """
    try:
        payload = decode_token(token)
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp, timezone.utc)
        return None
    except JWTError:
        return None
