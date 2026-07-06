# app/api/dashboard.py
# --------------------
# FastAPI router exposing endpoints for retrieving Enterprise Dashboard statistics.

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user, get_dashboard_service
from app.models.user import User
from app.services.dashboard.dashboard_service import DashboardService
from app.services.workspace.workspace_service import WorkspaceService
from app.schemas.dashboard import (
    DashboardResponse,
    WorkspaceOverview,
    DashboardMetricsResponse,
    ActivityItem,
    StorageMetrics
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
def get_dashboard_summary(
    current_user: User = Depends(get_current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
    db: Session = Depends(get_db)
):
    """
    Returns the unified dashboard summary for the current active workspace.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        return service.build_dashboard_summary(active_ws.id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/overview", response_model=WorkspaceOverview)
def get_workspace_overview(
    current_user: User = Depends(get_current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
    db: Session = Depends(get_db)
):
    """
    Returns metadata overview of the current active workspace.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        return service.get_workspace_overview(active_ws.id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fetch workspace overview: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/metrics", response_model=DashboardMetricsResponse)
def get_dashboard_metrics(
    current_user: User = Depends(get_current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
    db: Session = Depends(get_db)
):
    """
    Returns core metrics across conversations, collections, searches, and storage.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        return service.get_metrics(active_ws.id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to fetch core metrics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/activity", response_model=List[ActivityItem])
def get_dashboard_activity(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
    db: Session = Depends(get_db)
):
    """
    Returns recent activities chronologically within the active workspace.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        return service.get_activity(active_ws.id, current_user.id, limit=limit)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load recent activity: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/storage", response_model=StorageMetrics)
def get_dashboard_storage(
    current_user: User = Depends(get_current_active_user),
    service: DashboardService = Depends(get_dashboard_service),
    db: Session = Depends(get_db)
):
    """
    Returns storage details of the active workspace.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        return service.get_storage(active_ws.id, current_user.id)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate storage statistics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")
