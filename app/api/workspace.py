# app/api/workspace.py
# --------------------
# FastAPI route handlers exposing API endpoints for multi-workspace management.

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_active_user, get_workspace_service
from app.models.user import User
from app.services.workspace.workspace_service import WorkspaceService
from app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceListResponse,
    WorkspaceStatistics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(
    payload: WorkspaceCreate,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Creates a new workspace environment for the user.
    """
    try:
        return service.create_workspace(
            owner_id=current_user.id,
            name=payload.name,
            description=payload.description
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workspace: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("", response_model=WorkspaceListResponse)
def list_workspaces(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Lists workspaces owned by the user.
    """
    try:
        items, total = service.list_workspaces(current_user.id, page, page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Failed to list workspaces: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


# NOTE: Declare static routes before parameterized routes to prevent FastAPI collisions
@router.get("/default", response_model=WorkspaceResponse)
def get_default_workspace(
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Retrieves the user's active default workspace.
    """
    try:
        return service.get_default_workspace(current_user.id)
    except Exception as e:
        logger.error(f"Failed to retrieve default workspace: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{workspace_id}/default", status_code=status.HTTP_200_OK)
def set_default_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Sets designated workspace as default.
    """
    try:
        success = service.set_default_workspace(current_user.id, workspace_id)
        return {
            "success": success,
            "message": "Default workspace changed successfully" if success else "Default workspace could not be changed"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set default workspace {workspace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Retrieves workspace details.
    """
    try:
        return service.get_workspace(workspace_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to retrieve workspace {workspace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
def update_workspace(
    workspace_id: int,
    payload: WorkspaceUpdate,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Updates workspace metadata description.
    """
    try:
        return service.update_workspace(
            workspace_id=workspace_id,
            owner_id=current_user.id,
            name=payload.name,
            description=payload.description,
            is_active=payload.is_active
        )
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update workspace {workspace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{workspace_id}", status_code=status.HTTP_200_OK)
def delete_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Deletes workspace environment.
    """
    try:
        success = service.delete_workspace(workspace_id, current_user.id)
        return {
            "success": success,
            "message": "Workspace deleted successfully" if success else "Workspace could not be deleted"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete workspace {workspace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/{workspace_id}/statistics", response_model=WorkspaceStatistics)
def get_workspace_statistics(
    workspace_id: int,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Retrieves statistics metrics for workspace.
    """
    try:
        return service.get_workspace_statistics(workspace_id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate statistics for workspace {workspace_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")
