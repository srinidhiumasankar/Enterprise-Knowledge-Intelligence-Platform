# app/services/auth_service.py
# ----------------------------
# Business logic service layer for user authentication and JWT session creation.

import logging
from fastapi import HTTPException, status

from app.database.user_repository import UserRepository
from app.schemas.token import Token, LoginResponse
from app.schemas.user import UserResponse
from app.utils.security import verify_password, create_access_token, create_refresh_token

# Setup logger
logger = logging.getLogger(__name__)


class AuthService:
    """
    Service class encapsulating authentication business logic.
    """

    def __init__(self, repository: UserRepository):
        """
        Initialize AuthService with a UserRepository instance.
        """
        self.repository = repository

    def authenticate_user(self, email: str, password: str) -> LoginResponse:
        """
        Authenticate a user by email and password.
        Returns a LoginResponse containing tokens and user data.
        Raises HTTPException on invalid credentials or inactive user.
        """
        user = self.repository.get_by_email(email)
        if not user:
            logger.warning(f"Authentication failure: email '{email}' not found.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify password
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failure: password mismatch for email '{email}'.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check active status
        if not user.is_active:
            logger.warning(f"Authentication failure: user account '{email}' is inactive.")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )

        # Generate tokens
        access_token = create_access_token(subject=user.uuid)
        refresh_token = create_refresh_token(subject=user.uuid)

        from app.config import settings
        expires_in_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        user_resp = UserResponse.model_validate(user)

        logger.info(f"Authentication success: user '{email}' logged in (UUID: {user.uuid}).")

        return LoginResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in_seconds,
            user=user_resp,
            refresh_token=refresh_token,
        )
