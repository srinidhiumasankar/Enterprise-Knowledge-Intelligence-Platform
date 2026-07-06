# app/services/workspace/workspace_service.py
# -----------------------------------------
# Core business service layer orchestrating the Workspace lifecycle and isolation check gates.

import logging
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.workspace import Workspace
from app.repositories.workspace_repository import WorkspaceRepository

logger = logging.getLogger(__name__)


class WorkspaceService:
    """
    Service layer providing business validations and CRUD logic for Workspaces.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = WorkspaceRepository(db)

    def create_workspace(self, owner_id: int, name: str, description: Optional[str] = None) -> Workspace:
        """
        Creates a new Workspace. Enforces that workspace names must be unique per user.
        """
        # Enforce name uniqueness per user
        existing = self.repo.get_by_name(owner_id, name)
        if existing:
            logger.warning(f"Failed to create workspace: Name '{name}' already exists for user {owner_id}")
            raise ValueError(f"Workspace with name '{name}' already exists.")

        # Determine if this is the user's first workspace (if so, make it default)
        count = self.repo.count_for_user(owner_id)
        is_default = (count == 0)

        ws = self.repo.create(owner_id, name, description, is_default)
        logger.info(f"Workspace created successfully: id={ws.id}, is_default={ws.is_default}")
        return ws

    def get_workspace(self, workspace_id: int, owner_id: int) -> Workspace:
        """
        Retrieves workspace and validates owner authorization.
        """
        ws = self.repo.get(workspace_id)
        if not ws:
            logger.warning(f"Workspace {workspace_id} not found.")
            raise KeyError("Workspace not found")
        if ws.owner_id != owner_id:
            logger.warning(f"User {owner_id} unauthorized to access workspace {workspace_id}")
            raise PermissionError("Unauthorized workspace access")
        return ws

    def update_workspace(
        self,
        workspace_id: int,
        owner_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Workspace:
        """
        Modifies metadata of an active workspace.
        """
        ws = self.get_workspace(workspace_id, owner_id)

        # Enforce name uniqueness if changing name
        if name and name != ws.name:
            existing = self.repo.get_by_name(owner_id, name)
            if existing:
                raise ValueError(f"Workspace with name '{name}' already exists.")

        updated = self.repo.update(workspace_id, name, description, is_active)
        logger.info(f"Workspace updated: id={workspace_id}")
        return updated

    def delete_workspace(self, workspace_id: int, owner_id: int) -> bool:
        """
        Deletes workspace after confirming safety gates:
        - Cannot delete the last workspace.
        - Cannot delete the default workspace.
        """
        ws = self.get_workspace(workspace_id, owner_id)

        # Count workspaces
        total_count = self.repo.count_for_user(owner_id)
        if total_count <= 1:
            logger.warning(f"Attempt to delete last workspace {workspace_id} blocked for user {owner_id}")
            raise ValueError("Cannot delete last workspace")

        # Check default status
        if ws.is_default:
            logger.warning(f"Attempt to delete default workspace {workspace_id} blocked for user {owner_id}")
            raise ValueError("Cannot delete default workspace unless another default is assigned")

        res = self.repo.delete(workspace_id)
        logger.info(f"Workspace deleted: id={workspace_id}")
        return res

    def list_workspaces(self, owner_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[Workspace], int]:
        """
        Lists workspaces owned by user.
        """
        return self.repo.list(owner_id, page, page_size)

    def get_default_workspace(self, owner_id: int) -> Workspace:
        """
        Retrieves the default workspace for user, creating one if not exists.
        """
        ws = self.repo.get_default(owner_id)
        if not ws:
            # Create a default workspace
            from app.config.settings import settings
            default_name = getattr(settings, "DEFAULT_WORKSPACE_NAME", "My Workspace")
            ws = self.repo.create(owner_id, default_name, "Automatically created default workspace", is_default=True)
        return ws

    def set_default_workspace(self, owner_id: int, workspace_id: int) -> bool:
        """
        Designates a workspace as the user's default.
        """
        ws = self.get_workspace(workspace_id, owner_id)
        if not ws.is_active:
            raise ValueError("Cannot set inactive workspace as default")

        res = self.repo.set_default(owner_id, workspace_id)
        logger.info(f"Changed default workspace for user {owner_id} to workspace {workspace_id}")
        return res

    def validate_workspace_ownership(self, owner_id: int, workspace_id: int) -> None:
        """
        Helper validation check raising exceptions on ownership mismatch.
        """
        self.get_workspace(workspace_id, owner_id)

    def get_workspace_statistics(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Aggregates statistics metrics for a workspace.
        """
        self.get_workspace(workspace_id, owner_id)
        stats = self.repo.statistics(workspace_id)
        logger.info(f"Generated stats report for workspace {workspace_id}")
        return stats

    def switch_workspace(self, user_id: int, workspace_id: int) -> Workspace:
        """
        Switches the user's active workspace.
        """
        from app.services.workspace.workspace_context_service import WorkspaceContextService
        ctx_service = WorkspaceContextService(self.db)
        return ctx_service.set_active_workspace(user_id, workspace_id)

    def get_active_workspace(self, user_id: int) -> Workspace:
        """
        Retrieves the user's current active workspace.
        """
        from app.services.workspace.workspace_context_service import WorkspaceContextService
        ctx_service = WorkspaceContextService(self.db)
        return ctx_service.get_active_workspace(user_id)

    def validate_active_workspace(self, user_id: int, workspace_id: int) -> None:
        """
        Validates that workspace_id belongs to user and is currently active.
        """
        active_ws = self.get_active_workspace(user_id)
        if active_ws.id != workspace_id:
            raise PermissionError("Workspace is not the active workspace")
