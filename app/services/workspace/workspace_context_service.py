# app/services/workspace/workspace_context_service.py
# -------------------------------------------------
# Request-scoped context manager tracking and caching the active Workspace.

import contextvars
import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.workspace import Workspace
from app.models.user_preference import UserPreference

logger = logging.getLogger(__name__)

# Request-scoped ContextVar caching the resolved Workspace object
_active_workspace_context = contextvars.ContextVar("_active_workspace_context", default=None)


class WorkspaceContextService:
    """
    Service responsible for maintaining request-scoped active workspace context and caching.
    """
    def __init__(self, db: Session):
        self.db = db

    def get_active_workspace(self, user_id: int) -> Workspace:
        """
        Retrieves the active workspace for the current request scope.
        Resolves from UserPreference default_workspace or falls back to standard default.
        """
        cached = _active_workspace_context.get()
        if cached is not None:
            try:
                from sqlalchemy import inspect
                state = inspect(cached)
                if state.detached:
                    _active_workspace_context.set(None)
                else:
                    logger.info("Retrieved active workspace from request context cache.")
                    return cached
            except Exception:
                _active_workspace_context.set(None)

        # Check UserPreference default_workspace
        pref = self.db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
        workspace = None
        if pref and pref.default_workspace:
            workspace = self.db.get(Workspace, pref.default_workspace)
            if workspace and (workspace.owner_id != user_id or not workspace.is_active):
                workspace = None

        if not workspace:
            workspace = self.db.scalar(
                select(Workspace).where(Workspace.owner_id == user_id, Workspace.is_default == True)
            )

        if not workspace:
            # Lazy initialize default workspace
            from app.services.workspace.workspace_service import WorkspaceService
            ws_service = WorkspaceService(self.db)
            workspace = ws_service.get_default_workspace(user_id)

        _active_workspace_context.set(workspace)
        logger.info(f"Resolved and cached active workspace: id={workspace.id} for user {user_id}")
        return workspace

    def set_active_workspace(self, user_id: int, workspace_id: int) -> Workspace:
        """
        Switches the active workspace, updating preferences and updating context cache.
        """
        workspace = self.db.get(Workspace, workspace_id)
        if not workspace:
            logger.warning(f"Workspace {workspace_id} not found during switch.")
            raise KeyError("Workspace not found")
        if workspace.owner_id != user_id:
            logger.warning(f"User {user_id} unauthorized to switch to workspace {workspace_id}")
            raise PermissionError("Unauthorized workspace switch")
        if not workspace.is_active:
            logger.warning(f"Workspace {workspace_id} is inactive.")
            raise ValueError("Cannot switch to inactive workspace")

        # Persist switch in UserPreference
        pref = self.db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))
        if not pref:
            pref = UserPreference(user_id=user_id, default_workspace=workspace_id)
            self.db.add(pref)
        else:
            pref.default_workspace = workspace_id
            self.db.add(pref)
        self.db.commit()

        # Update cache contextVar
        _active_workspace_context.set(workspace)
        logger.info(f"Workspace context switched successfully to id={workspace_id}")
        return workspace

    def clear_context(self) -> None:
        """
        Clears the cached request context.
        """
        _active_workspace_context.set(None)
        logger.info("Cleared request workspace cache context.")

    def get_context(self) -> Optional[Workspace]:
        """
        Returns currently cached Workspace object if set.
        """
        return _active_workspace_context.get()
