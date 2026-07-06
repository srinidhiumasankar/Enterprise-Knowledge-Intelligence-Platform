# app/api/auth.py
# ---------------
# API router implementing authentication endpoints.

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user
from app.database import UserRepository
from app.models.user import User
from app.schemas.token import LoginRequest, LoginResponse
from app.schemas.user import UserCreate, UserResponse
from app.services import UserService, AuthService

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register User",
    description="Register a new user in the platform with email and password validation.",
)
async def register(
    user_in: UserCreate,
    db: Session = Depends(get_db),
) -> Any:
    """
    Register a new user.
    """
    logger.info(f"Registration attempt for email: '{user_in.email}'")
    try:
        user_service = UserService(UserRepository(db))
        user = user_service.create_user(user_in)
        
        # Provision default workspace
        from app.services.workspace.workspace_service import WorkspaceService
        ws_service = WorkspaceService(db)
        ws_service.get_default_workspace(user.id)

        logger.info(f"Successful registration and workspace provisioning for email: '{user_in.email}' (UUID: {user.uuid})")
        return user
    except HTTPException as he:
        logger.warning(f"Failed registration for email '{user_in.email}': {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Internal error during registration of '{user_in.email}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Login User",
    description=(
        "Authenticate user email and password and return a JWT access token. "
        "Supports both JSON request bodies and application/x-www-form-urlencoded forms (for Swagger UI)."
    ),
)
async def login(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Authenticate user and return access token.
    """
    content_type = request.headers.get("content-type", "")
    email = None
    password = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            email = body.get("email")
            password = body.get("password")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid JSON request body",
            )
    else:
        # Fallback to form data parsing for Swagger authorize flow compatibility
        try:
            form = await request.form()
            email = form.get("username") or form.get("email")
            password = form.get("password")
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid form data",
            )

    if not email or not password:
        logger.warning("Failed login: missing email or password credentials")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Email and password are required",
        )

    logger.info(f"Login attempt for user: '{email}'")
    try:
        auth_service = AuthService(UserRepository(db))
        response = auth_service.authenticate_user(email=email, password=password)
        logger.info(f"Successful login for user: '{email}'")
        return response
    except HTTPException as he:
        logger.warning(f"Failed login for user '{email}': {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Internal error during login for '{email}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout User",
    description="Logout the authenticated user (stateless).",
)
async def logout(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Perform stateless logout.
    """
    logger.info(f"Successful logout for user: '{current_user.email}' (UUID: {current_user.uuid})")
    return {"message": "Logged out successfully"}


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current User",
    description="Retrieve the authenticated user's profile.",
)
async def get_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current logged in user details.
    """
    logger.info(f"Successful profile retrieval for user: '{current_user.email}'")
    return current_user
