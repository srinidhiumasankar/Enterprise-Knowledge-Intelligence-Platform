# app/schemas/user.py
# --------------------
# Pydantic validation schemas for the User model.

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserCreate(BaseModel):
    """
    Schema for creating a new user.
    """
    email: EmailStr
    password: str = Field(..., min_length=8, description="User password (min length 8)")
    full_name: str | None = Field(default=None, max_length=255)


class UserUpdate(BaseModel):
    """
    Schema for updating an existing user.
    All fields are optional.
    """
    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=8)
    full_name: str | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None


class UserResponse(BaseModel):
    """
    Schema for returning user information.
    Excludes sensitive data like hashed_password.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    uuid: str
    email: str
    full_name: str | None = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
