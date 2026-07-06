# app/services/search_history/search_history_service.py
# ----------------------------------------------------
# Core service layer orchestrating Search History lifecycles and validation.

import logging
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models.search_history import SearchHistory
from app.repositories.search_history_repository import SearchHistoryRepository
from app.services.workspace.workspace_service import WorkspaceService

logger = logging.getLogger(__name__)


class SearchHistoryService:
    """
    Service layer implementing search recording, retrieval, and analytical statistics.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = SearchHistoryRepository(db)
        self.ws_service = WorkspaceService(db)

    def _validate_workspace(self, user_id: int, workspace_id: int) -> None:
        """
        Helper validating user workspace ownership boundaries.
        """
        self.ws_service.validate_workspace_ownership(user_id, workspace_id)

    def record_search(
        self,
        user_id: int,
        workspace_id: int,
        query: str,
        filters_json: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        result_count: Optional[int] = None
    ) -> SearchHistory:
        """
        Persists a search query execution history after validating workspace ownership.
        """
        # Validate workspace ownership
        self._validate_workspace(user_id, workspace_id)
        
        # Enforce search history flag from settings
        from app.config.settings import settings
        enable_history = getattr(settings, "ENABLE_SEARCH_HISTORY", True)
        if not enable_history:
            logger.info("Search history recording is disabled via config.")
            return None

        history = self.repo.create(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
            filters_json=filters_json,
            execution_time_ms=execution_time_ms,
            result_count=result_count
        )
        logger.info(f"Recorded search query for user {user_id} in workspace {workspace_id}")
        return history

    def get_recent(self, user_id: int, workspace_id: int, limit: Optional[int] = None) -> List[SearchHistory]:
        """
        Returns recent search entries.
        """
        self._validate_workspace(user_id, workspace_id)
        
        from app.config.settings import settings
        max_limit = limit or getattr(settings, "MAX_RECENT_SEARCHES", 50)
        
        return self.repo.recent(user_id, workspace_id, limit=max_limit)

    def get_frequent(self, user_id: int, workspace_id: int, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Returns frequently searched queries sorted by search count.
        """
        self._validate_workspace(user_id, workspace_id)
        return self.repo.frequent(user_id, workspace_id, limit=limit)

    def get_statistics(self, user_id: int, workspace_id: int) -> Dict[str, Any]:
        """
        Retrieves search history statistics metrics.
        """
        self._validate_workspace(user_id, workspace_id)
        return self.repo.statistics(user_id, workspace_id)

    def delete_history(self, history_id: int, user_id: int) -> bool:
        """
        Deletes a single search history record.
        """
        entry = self.repo.get(history_id)
        if not entry:
            raise KeyError("Search history entry not found")
        if entry.user_id != user_id:
            raise PermissionError("Unauthorized to delete search history")

        # Validate that the user owns the workspace where search took place
        self._validate_workspace(user_id, entry.workspace_id)

        res = self.repo.delete(history_id)
        logger.info(f"Deleted search history entry: ID={history_id} for user {user_id}")
        return res

    def clear_history(self, user_id: int, workspace_id: int) -> bool:
        """
        Wipes all search history logs inside a workspace boundary.
        """
        self._validate_workspace(user_id, workspace_id)
        res = self.repo.clear(user_id, workspace_id)
        logger.info(f"Cleared search history for user {user_id} in workspace {workspace_id}")
        return res

    def list_history(self, user_id: int, workspace_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[SearchHistory], int]:
        """
        Lists search history entries.
        """
        self._validate_workspace(user_id, workspace_id)
        return self.repo.list(user_id, workspace_id, page, page_size)
