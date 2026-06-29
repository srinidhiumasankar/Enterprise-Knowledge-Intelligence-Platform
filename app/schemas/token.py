# app/schemas/token.py
# --------------------
# Pydantic schemas for authentication tokens and login requests.

from pydantic import BaseModel, EmailStr, Field
from app.schemas.user import UserResponse


class Token(BaseModel):
    """
    Schema for JWT tokens returned upon authentication.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type, e.g. bearer")
    refresh_token: str | None = Field(default=None, description="Optional JWT refresh token")


class TokenPayload(BaseModel):
    """
    Schema for payload decrypted from JWT.
    """
    sub: str | None = Field(default=None, description="Subject (typically user UUID)")


class LoginRequest(BaseModel):
    """
    Schema for authentication credentials.
    """
    email: EmailStr
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """
    Schema for successful authentication response.
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="Bearer", description="Token type, e.g. Bearer")
    expires_in: int = Field(..., description="Token validity duration in seconds")
    user: UserResponse
    refresh_token: str | None = Field(default=None, description="Optional JWT refresh token")
