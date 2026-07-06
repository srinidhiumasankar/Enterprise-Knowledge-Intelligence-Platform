# app/services/dashboard/dashboard_service.py
# ----------------------------------------
# Core service layer coordinating dashboard summary assembly and active user validation.

import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from app.repositories.dashboard_repository import DashboardRepository
from app.services.workspace.workspace_service import WorkspaceService

logger = logging.getLogger(__name__)


class DashboardService:
    """
    Service coordinating workspace summaries, statistics, recent activities, and storage details.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = DashboardRepository(db)
        self.workspace_service = WorkspaceService(db)

    def _validate(self, owner_id: int, workspace_id: int) -> None:
        """
        Gating function ensuring security boundaries on active workspace.
        """
        self.workspace_service.validate_workspace_ownership(owner_id, workspace_id)

    def build_dashboard_summary(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Returns full dashboard statistics data payload.
        """
        logger.info(f"Dashboard summary requested for workspace {workspace_id} by user {owner_id}")
        self._validate(owner_id, workspace_id)

        overview = self.repo.get_workspace_overview(workspace_id, owner_id)
        conversation_metrics = self.repo.get_conversation_metrics(workspace_id)
        collection_metrics = self.repo.get_collection_metrics(workspace_id)
        search_metrics = self.repo.get_search_metrics(workspace_id)
        document_metrics = self.repo.get_document_metrics(workspace_id)
        storage_metrics = self.repo.get_storage_metrics(workspace_id)

        from app.config import settings
        limit = getattr(settings, "DASHBOARD_ACTIVITY_LIMIT", 20)
        recent_activity = self.repo.get_recent_activities(workspace_id, limit=limit)

        logger.info("Dashboard summary generated successfully")
        return {
            "overview": overview,
            "conversation_metrics": conversation_metrics,
            "collection_metrics": collection_metrics,
            "search_metrics": search_metrics,
            "document_metrics": document_metrics,
            "storage_metrics": storage_metrics,
            "recent_activity": recent_activity
        }

    def get_workspace_overview(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Returns workspace overview metadata details.
        """
        logger.info(f"Workspace overview requested for workspace {workspace_id} by user {owner_id}")
        self._validate(owner_id, workspace_id)
        res = self.repo.get_workspace_overview(workspace_id, owner_id)
        logger.info("Workspace overview generated successfully")
        return res

    def get_metrics(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Returns core metrics (documents, collections, conversations, searches, storage).
        """
        logger.info(f"Core metrics requested for workspace {workspace_id} by user {owner_id}")
        self._validate(owner_id, workspace_id)

        conv = self.repo.get_conversation_metrics(workspace_id)
        coll = self.repo.get_collection_metrics(workspace_id)
        search = self.repo.get_search_metrics(workspace_id)
        doc = self.repo.get_document_metrics(workspace_id)
        storage = self.repo.get_storage_metrics(workspace_id)

        logger.info("Core metrics loaded successfully")
        return {
            "conversations": conv,
            "collections": coll,
            "searches": search,
            "documents": doc,
            "storage": storage
        }

    def get_activity(self, workspace_id: int, owner_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Returns combined recent activities for workspace.
        """
        logger.info(f"Activity feed requested for workspace {workspace_id} by user {owner_id}")
        self._validate(owner_id, workspace_id)

        from app.config import settings
        max_limit = limit or getattr(settings, "DASHBOARD_ACTIVITY_LIMIT", 20)
        res = self.repo.get_recent_activities(workspace_id, limit=max_limit)

        logger.info("Activity feed loaded successfully")
        return res

    def get_storage(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Returns storage sizes and details.
        """
        logger.info(f"Storage metrics requested for workspace {workspace_id} by user {owner_id}")
        self._validate(owner_id, workspace_id)
        res = self.repo.get_storage_metrics(workspace_id)
        logger.info("Storage metrics aggregated successfully")
        return res
