# app/database/user_repository.py
# ------------------------------
# Data access repository layer for the User model.

from typing import Sequence
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserRepository:
    """
    Repository class encapsulating database operations for the User model.
    """

    def __init__(self, db: Session):
        """
        Initialize repository with a database session.
        """
        self.db = db

    def create_user(self, user_in: UserCreate, hashed_password: str) -> User:
        """
        Create a new user record in the database.
        """
        db_user = User(
            email=user_in.email,
            hashed_password=hashed_password,
            full_name=user_in.full_name,
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def get_by_id(self, user_id: int) -> User | None:
        """
        Retrieve a user by their database integer ID.
        """
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """
        Retrieve a user by their unique email address.
        """
        return self.db.scalars(
            select(User).where(User.email == email)
        ).first()

    def get_by_uuid(self, user_uuid: str) -> User | None:
        """
        Retrieve a user by their unique UUID string.
        """
        return self.db.scalars(
            select(User).where(User.uuid == user_uuid)
        ).first()

    def update_user(self, db_user: User, update_data: UserUpdate | dict) -> User:
        """
        Update fields on an existing user record.
        """
        if isinstance(update_data, dict):
            update_dict = update_data
        else:
            update_dict = update_data.model_dump(exclude_unset=True)

        if "password" in update_dict:
            # Note: For now, password update stores password as-is (simulating hashed_password).
            hashed_password = update_dict.pop("password")
            db_user.hashed_password = hashed_password

        for key, value in update_dict.items():
            setattr(db_user, key, value)

        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def delete_user(self, db_user: User) -> None:
        """
        Delete a user record from the database.
        """
        self.db.delete(db_user)
        self.db.commit()

    def list_users(self, skip: int = 0, limit: int = 100) -> Sequence[User]:
        """
        List users with pagination offsets.
        """
        return self.db.scalars(
            select(User).offset(skip).limit(limit)
        ).all()
