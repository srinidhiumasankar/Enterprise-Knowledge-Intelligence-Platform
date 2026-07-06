# app/services/collection/collection_service.py
# -------------------------------------------
# Core service orchestrating collections lifecycle, many-to-many document links, and ownership verification gates.

import logging
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.collection import Collection
from app.models.document import Document
from app.models.workspace import Workspace
from app.repositories.collection_repository import CollectionRepository

logger = logging.getLogger(__name__)


class CollectionService:
    """
    Service layer providing validation gates and logic for Document Collections organization.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = CollectionRepository(db)

    def _validate_workspace(self, user_id: int, workspace_id: Optional[int]) -> int:
        """
        Confirms workspace belongs to the user, or returns the current active workspace.
        """
        from app.services.workspace.workspace_context_service import WorkspaceContextService
        ctx_service = WorkspaceContextService(self.db)

        if workspace_id is not None:
            workspace = self.db.scalar(select(Workspace).where(Workspace.id == workspace_id))
            if not workspace:
                raise ValueError("Workspace not found")
            if workspace.owner_id != user_id:
                raise PermissionError("Unauthorized access to workspace")
            return workspace.id

        # Fallback to current active workspace
        active_ws = ctx_service.get_active_workspace(user_id)
        return active_ws.id

    def create_collection(
        self,
        owner_id: int,
        name: str,
        description: Optional[str] = None,
        workspace_id: Optional[int] = None
    ) -> Collection:
        """
        Enforces workspace access and name uniqueness, then creates collection.
        """
        validated_ws_id = self._validate_workspace(owner_id, workspace_id)

        # Check duplicate name within workspace boundary
        if self.repo.exists_by_name(validated_ws_id, name):
            raise ValueError(f"Collection with name '{name}' already exists in workspace.")

        col = self.repo.create(validated_ws_id, owner_id, name, description)
        logger.info(f"Collection created successfully: id={col.id}")
        return col

    def get_collection(self, collection_id: int, owner_id: int) -> Collection:
        """
        Loads collection after performing user ownership validation check.
        """
        col = self.repo.get(collection_id)
        if not col:
            raise KeyError("Collection not found")
        if col.owner_id != owner_id:
            raise PermissionError("Unauthorized access to collection")
        return col

    def update_collection(
        self,
        collection_id: int,
        owner_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Collection:
        """
        Modifies metadata properties of an active collection.
        """
        col = self.get_collection(collection_id, owner_id)

        if name and name != col.name:
            if self.repo.exists_by_name(col.workspace_id, name):
                raise ValueError(f"Collection with name '{name}' already exists in workspace.")

        updated = self.repo.update(collection_id, name, description)
        logger.info(f"Collection updated: id={collection_id}")
        return updated

    def delete_collection(self, collection_id: int, owner_id: int) -> bool:
        """
        Deletes a collection.
        """
        self.get_collection(collection_id, owner_id)
        res = self.repo.delete(collection_id)
        logger.info(f"Collection deleted: id={collection_id}")
        return res

    def list_collections(
        self,
        owner_id: int,
        page: int = 1,
        page_size: int = 20,
        workspace_id: Optional[int] = None
    ) -> Tuple[List[Collection], int]:
        """
        Returns paginated list of collection entries owned by user.
        """
        validated_ws_id = self._validate_workspace(owner_id, workspace_id)
        return self.repo.list(owner_id, page, page_size, workspace_id=validated_ws_id)

    def add_document(self, collection_id: int, owner_id: int, document_id: int) -> bool:
        """
        Associates a document to a collection after validating permissions and workspace bounds.
        """
        col = self.get_collection(collection_id, owner_id)

        doc = self.db.get(Document, document_id)
        if not doc:
            raise KeyError("Document not found")

        # Verify document user access
        if doc.owner_id != owner_id:
            raise PermissionError("Unauthorized access to document")
        
        # Verify workspace alignment matches
        if doc.workspace_id is None:
            doc.workspace_id = col.workspace_id
            self.db.add(doc)
            self.db.commit()
        elif doc.workspace_id != col.workspace_id:
            raise ValueError("Workspace mismatch: Document and Collection must be in same workspace.")

        # Check duplicate links
        if doc in col.documents:
            raise ValueError("Document already exists in collection.")

        res = self.repo.add_document(collection_id, document_id)
        logger.info(f"Linked document {document_id} to collection {collection_id}")
        return res

    def remove_document(self, collection_id: int, owner_id: int, document_id: int) -> bool:
        """
        Removes a document link relationship.
        """
        col = self.get_collection(collection_id, owner_id)

        doc = self.db.get(Document, document_id)
        if not doc:
            raise KeyError("Document not found")

        if doc.owner_id != owner_id:
            raise PermissionError("Unauthorized access to document")

        # Verify link is present
        if doc not in col.documents:
            raise ValueError("Document does not belong to collection.")

        res = self.repo.remove_document(collection_id, document_id)
        logger.info(f"Unlinked document {document_id} from collection {collection_id}")
        return res

    def get_documents(self, collection_id: int, owner_id: int) -> List[Document]:
        """
        Returns all document entities associated with the collection.
        """
        self.get_collection(collection_id, owner_id)
        return self.repo.list_documents(collection_id)

    def get_statistics(self, collection_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Generates stats reports for the collection.
        """
        self.get_collection(collection_id, owner_id)
        stats = self.repo.statistics(collection_id)
        logger.info(f"Generated stats report for collection {collection_id}")
        return stats
