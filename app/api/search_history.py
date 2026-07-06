# app/api/search_history.py
# -------------------------
# FastAPI router exposing endpoints for managing Enterprise Search History.

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_current_active_user, get_search_history_service
from app.models.user import User
from app.services.search_history.search_history_service import SearchHistoryService
from app.services.workspace.workspace_service import WorkspaceService
from app.schemas.search_history import (
    SearchHistoryListResponse,
    RecentSearchResponse,
    FrequentSearchResponse,
    SearchStatisticsResponse,
    DeleteHistoryResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search-history", tags=["search-history"])


@router.get("", response_model=SearchHistoryListResponse)
def list_search_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service),
    db: Session = Depends(get_db)
):
    """
    Lists paginated search history records for the current active workspace context.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        
        items, total = service.list_history(current_user.id, active_ws.id, page, page_size)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Failed to list search history: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/recent", response_model=List[RecentSearchResponse])
def get_recent_searches(
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service),
    db: Session = Depends(get_db)
):
    """
    Returns recent search queries in the active workspace context.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        return service.get_recent(current_user.id, active_ws.id, limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch recent searches: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/frequent", response_model=List[FrequentSearchResponse])
def get_frequent_searches(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service),
    db: Session = Depends(get_db)
):
    """
    Returns frequently searched queries in the active workspace context.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        tuples = service.get_frequent(current_user.id, active_ws.id, limit=limit)
        return [{"query": item[0], "count": item[1]} for item in tuples]
    except Exception as e:
        logger.error(f"Failed to fetch frequent searches: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.get("/statistics", response_model=SearchStatisticsResponse)
def get_search_statistics(
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service),
    db: Session = Depends(get_db)
):
    """
    Retrieves aggregated search history statistics for the active workspace context.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        return service.get_statistics(current_user.id, active_ws.id)
    except Exception as e:
        logger.error(f"Failed to load search stats: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("", response_model=DeleteHistoryResponse)
def clear_search_history(
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service),
    db: Session = Depends(get_db)
):
    """
    Clears all search history records for the user in the active workspace.
    """
    try:
        ws_service = WorkspaceService(db)
        active_ws = ws_service.get_active_workspace(current_user.id)
        success = service.clear_history(current_user.id, active_ws.id)
        return {
            "success": success,
            "message": "Search history cleared successfully" if success else "Failed to clear search history"
        }
    except Exception as e:
        logger.error(f"Failed to clear search history: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")


@router.delete("/{history_id}", response_model=DeleteHistoryResponse)
def delete_search_entry(
    history_id: int,
    current_user: User = Depends(get_current_active_user),
    service: SearchHistoryService = Depends(get_search_history_service)
):
    """
    Deletes a single search history entry after validating ownership.
    """
    try:
        success = service.delete_history(history_id, current_user.id)
        return {
            "success": success,
            "message": "Search history record deleted successfully"
        }
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete search entry {history_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database failure")
