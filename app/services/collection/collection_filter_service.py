# app/services/collection/collection_filter_service.py
# --------------------------------------------------
# Service layer to resolve document filter arrays based on collection selections and workspace scopes.

import logging
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.collection import Collection
from app.models.document import Document
from app.models.workspace import Workspace

logger = logging.getLogger(__name__)


class CollectionFilterService:
    """
    Service responsible for validating tenant boundaries and translating collection IDs into document ID filters.
    """
    def __init__(self, db: Session):
        self.db = db

    def validate_and_resolve_filters(
        self,
        user_id: int,
        workspace_id: int,
        collection_ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Validates collection permissions and resolves target document IDs inside the workspace.
        """
        # 1. Validate Workspace
        workspace = self.db.scalar(select(Workspace).where(Workspace.id == workspace_id))
        if not workspace:
            logger.warning(f"Workspace {workspace_id} not found during filter resolution.")
            raise ValueError("Workspace not found")
        if workspace.owner_id != user_id:
            logger.warning(f"User {user_id} unauthorized to access workspace {workspace_id}")
            raise PermissionError("Unauthorized workspace access")

        # 2. Case: Search all documents in the active workspace
        if not collection_ids:
            stmt = select(Document.id).where(
                Document.workspace_id == workspace_id,
                Document.owner_id == user_id
            )
            doc_ids = list(self.db.scalars(stmt).all())
            logger.info(f"Resolved all {len(doc_ids)} documents in workspace {workspace_id}.")
            return doc_ids if doc_ids else [-1]

        # 3. Case: Filter search scope to selected collections
        doc_ids_set = set()
        for col_id in collection_ids:
            col = self.db.get(Collection, col_id)
            if not col:
                logger.warning(f"Collection {col_id} not found.")
                raise KeyError(f"Collection {col_id} not found")
            if col.owner_id != user_id:
                logger.warning(f"User {user_id} unauthorized to access collection {col_id}")
                raise PermissionError(f"Unauthorized collection access: {col_id}")
            if col.workspace_id != workspace_id:
                logger.warning(f"Workspace mismatch: Collection {col_id} is in workspace {col.workspace_id}, not {workspace_id}")
                raise ValueError(f"Collection {col_id} does not belong to workspace {workspace_id}")

            for doc in col.documents:
                # Extra safety check: ensure linked document belongs to this user and workspace
                if doc.owner_id == user_id and doc.workspace_id == workspace_id:
                    doc_ids_set.add(doc.id)

        resolved_ids = list(doc_ids_set)
        logger.info(f"Resolved {len(resolved_ids)} document IDs from collections: {collection_ids}")
        return resolved_ids if resolved_ids else [-1]
