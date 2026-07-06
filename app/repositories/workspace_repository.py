# app/repositories/workspace_repository.py
# -------------------------------------
# Data access repository layer for managing Workspaces.

import logging
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.models.workspace import Workspace

logger = logging.getLogger(__name__)


class WorkspaceRepository:
    """
    Repository class encapsulating database operations for the Workspace model.
    """
    def __init__(self, db: Session):
        self.db = db

    def create(self, owner_id: int, name: str, description: Optional[str] = None, is_default: bool = False) -> Workspace:
        """
        Creates and persists a new Workspace.
        If is_default is True, we must clear default flags on other user workspaces.
        """
        if is_default:
            self.clear_default_flag(owner_id)

        ws = Workspace(
            owner_id=owner_id,
            name=name,
            description=description,
            is_default=is_default,
            is_active=True
        )
        self.db.add(ws)
        self.db.commit()
        self.db.refresh(ws)
        logger.info(f"Workspace created in DB: id={ws.id}, name={ws.name}, is_default={ws.is_default}")
        return ws

    def get(self, workspace_id: int) -> Optional[Workspace]:
        """
        Retrieves a workspace by its integer ID.
        """
        return self.db.get(Workspace, workspace_id)

    def get_by_name(self, owner_id: int, name: str) -> Optional[Workspace]:
        """
        Retrieves workspace matching owner and name.
        """
        return self.db.scalar(
            select(Workspace).where(Workspace.owner_id == owner_id, Workspace.name == name)
        )

    def list(self, owner_id: int, page: int = 1, page_size: int = 20) -> Tuple[List[Workspace], int]:
        """
        Lists workspaces owned by the user with pagination.
        """
        query = select(Workspace).where(Workspace.owner_id == owner_id)
        
        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_records = self.db.scalar(count_query) or 0
        
        # Ordering & offset
        query = query.order_by(Workspace.is_default.desc(), Workspace.id.asc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        items = list(self.db.scalars(query).all())
        return items, total_records

    def update(
        self,
        workspace_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[Workspace]:
        """
        Modifies name, description, or active status on a workspace.
        """
        ws = self.db.get(Workspace, workspace_id)
        if ws:
            if name is not None:
                ws.name = name
            if description is not None:
                ws.description = description
            if is_active is not None:
                ws.is_active = is_active
            ws.updated_at = func.now()
            self.db.add(ws)
            self.db.commit()
            self.db.refresh(ws)
            logger.info(f"Workspace metadata updated: id={workspace_id}")
            return ws
        return None

    def delete(self, workspace_id: int) -> bool:
        """
        Deletes a workspace.
        """
        ws = self.db.get(Workspace, workspace_id)
        if ws:
            self.db.delete(ws)
            self.db.commit()
            logger.info(f"Workspace deleted from DB: id={workspace_id}")
            return True
        return False

    def get_default(self, owner_id: int) -> Optional[Workspace]:
        """
        Returns the user's default workspace.
        """
        return self.db.scalar(
            select(Workspace).where(Workspace.owner_id == owner_id, Workspace.is_default == True)
        )

    def set_default(self, owner_id: int, workspace_id: int) -> bool:
        """
        Sets a specific workspace as default, unsetting all other defaults for the user.
        """
        ws = self.db.get(Workspace, workspace_id)
        if not ws or ws.owner_id != owner_id:
            return False

        self.clear_default_flag(owner_id)
        ws.is_default = True
        ws.updated_at = func.now()
        self.db.add(ws)
        self.db.commit()
        logger.info(f"Set default workspace for user {owner_id} to workspace {workspace_id}")
        return True

    def clear_default_flag(self, owner_id: int):
        """
        Clears default status flag across all user workspaces.
        """
        existing_defaults = self.db.scalars(
            select(Workspace).where(Workspace.owner_id == owner_id, Workspace.is_default == True)
        ).all()
        for d_ws in existing_defaults:
            d_ws.is_default = False
            self.db.add(d_ws)
        self.db.flush()

    def count_for_user(self, owner_id: int) -> int:
        """
        Counts total workspaces owned by the user.
        """
        return self.db.scalar(
            select(func.count(Workspace.id)).where(Workspace.owner_id == owner_id)
        ) or 0

    def statistics(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        """
        Aggregates stats reporting counts and timestamps for a workspace.
        """
        ws = self.db.get(Workspace, workspace_id)
        if not ws:
            return None

        from app.models.document import Document
        from app.models.collection import Collection
        from app.models.conversation import Conversation
        from app.models.search_history import SearchHistory

        doc_count = self.db.scalar(select(func.count(Document.id)).where(Document.workspace_id == workspace_id)) or 0
        col_count = self.db.scalar(select(func.count(Collection.id)).where(Collection.workspace_id == workspace_id)) or 0
        conv_count = self.db.scalar(select(func.count(Conversation.id)).where(Conversation.workspace_id == workspace_id)) or 0
        search_count = self.db.scalar(select(func.count(SearchHistory.id)).where(SearchHistory.workspace_id == workspace_id)) or 0
        storage_usage = self.db.scalar(select(func.sum(Document.file_size)).where(Document.workspace_id == workspace_id)) or 0

        # Activity timestamp scanning
        times = [ws.updated_at or ws.created_at]

        def get_max_time(model):
            updated = self.db.scalar(select(func.max(model.updated_at)).where(model.workspace_id == workspace_id))
            if updated:
                return updated
            created = self.db.scalar(select(func.max(model.created_at)).where(model.workspace_id == workspace_id))
            return created

        for model in [Document, Collection, Conversation, SearchHistory]:
            try:
                t = get_max_time(model)
                if t:
                    times.append(t)
            except Exception:
                try:
                    created = self.db.scalar(select(func.max(model.created_at)).where(model.workspace_id == workspace_id))
                    if created:
                        times.append(created)
                except Exception:
                    pass

        last_activity = max(times) if times else (ws.updated_at or ws.created_at)

        return {
            "workspace_id": ws.id,
            "document_count": doc_count,
            "collection_count": col_count,
            "conversation_count": conv_count,
            "search_count": search_count,
            "storage_usage": storage_usage,
            "created_at": ws.created_at,
            "last_activity": last_activity
        }
