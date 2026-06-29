# app/services/user_service.py
# ----------------------------
# Business logic service layer for User management.

from typing import Sequence
from fastapi import HTTPException, status

from app.database.user_repository import UserRepository
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserService:
    """
    Service class encapsulating business logic for User operations.
    """

    def __init__(self, repository: UserRepository):
        """
        Initialize UserService with a UserRepository instance.
        """
        self.repository = repository

    def create_user(self, user_in: UserCreate) -> User:
        """
        Register a new user. Verifies email uniqueness before creation.
        """
        existing_user = self.repository.get_by_email(user_in.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        # Note: Password hashing is skipped in this phase per instructions.
        return self.repository.create_user(user_in, hashed_password=user_in.password)

    def get_user_by_id(self, user_id: int) -> User:
        """
        Retrieve a user by their integer database ID. Raises 404 if not found.
        """
        user = self.repository.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_444_NOT_FOUND if hasattr(status, "HTTP_444_NOT_FOUND") else status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user

    def get_user_by_email(self, email: str) -> User:
        """
        Retrieve a user by their email. Raises 404 if not found.
        """
        user = self.repository.get_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return user

    def update_user(self, user_id: int, update_data: UserUpdate) -> User:
        """
        Update user information. Verifies email uniqueness if modified.
        """
        user = self.get_user_by_id(user_id)

        if update_data.email and update_data.email != user.email:
            existing_user = self.repository.get_by_email(update_data.email)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )

        return self.repository.update_user(user, update_data)

    def delete_user(self, user_id: int) -> None:
        """
        Delete a user record from the database.
        """
        user = self.get_user_by_id(user_id)
        self.repository.delete_user(user)

    def list_users(self, skip: int = 0, limit: int = 100) -> Sequence[User]:
        """
        Retrieve a list of users with pagination.
        """
        return self.repository.list_users(skip=skip, limit=limit)
