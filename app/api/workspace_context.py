# app/api/workspace_context.py
# ----------------------------
# FastAPI router exposing active workspace switching context endpoints.

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_active_user, get_workspace_service
from app.models.user import User
from app.services.workspace.workspace_service import WorkspaceService
from app.schemas.workspace import WorkspaceResponse, WorkspaceStatistics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces-context"])


@router.get("/current", response_model=WorkspaceResponse)
def get_current_active_workspace(
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Returns the user's current active workspace from context.
    """
    try:
        return service.get_active_workspace(current_user.id)
    except Exception as e:
        logger.error(f"Failed to get current active workspace: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.post("/{workspace_id}/switch", status_code=status.HTTP_200_OK)
def switch_active_workspace(
    workspace_id: int,
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Switches the active workspace environment.
    """
    try:
        ws = service.switch_workspace(current_user.id, workspace_id)
        if ws:
            from app.utils.activity_logger import log_activity
            log_activity(service.repo.db, current_user.id, ws.id, "workspace_switched", f"Switched workspace context to: '{ws.name}'")
        return {
            "success": True,
            "workspace_id": ws.id,
            "message": f"Successfully switched to workspace '{ws.name}'"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to switch workspace: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/current/statistics", response_model=WorkspaceStatistics)
def get_current_workspace_statistics(
    current_user: User = Depends(get_current_active_user),
    service: WorkspaceService = Depends(get_workspace_service)
):
    """
    Retrieves statistics metrics for the current active workspace.
    """
    try:
        active_ws = service.get_active_workspace(current_user.id)
        return service.get_workspace_statistics(active_ws.id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get current workspace stats: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")
