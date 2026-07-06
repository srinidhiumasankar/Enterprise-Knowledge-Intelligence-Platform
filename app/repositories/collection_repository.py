# app/repositories/collection_repository.py
# ----------------------------------------
# Data access repository layer for managing Document Collections and many-to-many document links.

import logging
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select, func
from sqlalchemy.orm import Session, selectinload

from app.models.collection import Collection
from app.models.document import Document

logger = logging.getLogger(__name__)


class CollectionRepository:
    """
    Repository class encapsulating database operations for the Collection model.
    """
    def __init__(self, db: Session):
        self.db = db

    def create(self, workspace_id: int, owner_id: int, name: str, description: Optional[str] = None) -> Collection:
        """
        Creates and persists a new Collection entry.
        """
        col = Collection(
            workspace_id=workspace_id,
            owner_id=owner_id,
            name=name,
            description=description
        )
        self.db.add(col)
        self.db.commit()
        self.db.refresh(col)
        logger.info(f"Collection created in DB: id={col.id}, name={col.name}")
        return col

    def update(self, collection_id: int, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Collection]:
        """
        Updates metadata fields on a collection.
        """
        col = self.db.get(Collection, collection_id)
        if col:
            if name is not None:
                col.name = name
            if description is not None:
                col.description = description
            col.updated_at = func.now()
            self.db.add(col)
            self.db.commit()
            self.db.refresh(col)
            logger.info(f"Collection metadata updated: id={collection_id}")
            return col
        return None

    def delete(self, collection_id: int) -> bool:
        """
        Deletes a collection. This only drops association links from document_collections, 
        preserving the Document records themselves.
        """
        col = self.db.get(Collection, collection_id)
        if col:
            # Clear associated documents list before delete to explicitly unlink them
            col.documents.clear()
            self.db.delete(col)
            self.db.commit()
            logger.info(f"Collection deleted from DB: id={collection_id}")
            return True
        return False

    def exists(self, collection_id: int) -> bool:
        """
        Checks presence of a collection.
        """
        return self.db.scalar(select(Collection.id).where(Collection.id == collection_id)) is not None

    def exists_by_name(self, workspace_id: int, name: str) -> bool:
        """
        Checks name uniqueness within a workspace boundary.
        """
        return self.db.scalar(
            select(Collection.id).where(Collection.workspace_id == workspace_id, Collection.name == name)
        ) is not None

    def list(self, owner_id: int, page: int = 1, page_size: int = 20, workspace_id: Optional[int] = None) -> Tuple[List[Collection], int]:
        """
        Lists collections belonging to an owner.
        """
        query = select(Collection).where(Collection.owner_id == owner_id)
        if workspace_id is not None:
            query = query.where(Collection.workspace_id == workspace_id)
        
        # Count
        count_query = select(func.count()).select_from(query.subquery())
        total_records = self.db.scalar(count_query) or 0
        
        # Ordering & offsets
        query = query.order_by(Collection.updated_at.desc(), Collection.id.desc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        items = list(self.db.scalars(query).all())
        return items, total_records

    def get(self, collection_id: int) -> Optional[Collection]:
        """
        Retrieves a collection by its database integer ID.
        """
        return self.db.get(Collection, collection_id)

    def add_document(self, collection_id: int, document_id: int) -> bool:
        """
        Appends a document link to the collection documents list.
        """
        col = self.db.get(Collection, collection_id)
        doc = self.db.get(Document, document_id)
        if col and doc:
            if doc not in col.documents:
                col.documents.append(doc)
                col.updated_at = func.now()
                self.db.add(col)
                self.db.commit()
                logger.info(f"Linked document {document_id} to collection {collection_id}")
                return True
        return False

    def remove_document(self, collection_id: int, document_id: int) -> bool:
        """
        Removes a document link from the collection documents list.
        """
        col = self.db.get(Collection, collection_id)
        doc = self.db.get(Document, document_id)
        if col and doc:
            if doc in col.documents:
                col.documents.remove(doc)
                col.updated_at = func.now()
                self.db.add(col)
                self.db.commit()
                logger.info(f"Unlinked document {document_id} from collection {collection_id}")
                return True
        return False

    def list_documents(self, collection_id: int) -> List[Document]:
        """
        Returns all documents inside the collection.
        """
        query = select(Collection).where(Collection.id == collection_id).options(
            selectinload(Collection.documents)
        )
        col = self.db.scalar(query)
        return col.documents if col else []

    def statistics(self, collection_id: int) -> Optional[Dict[str, Any]]:
        """
        Aggregates metadata stats for the collection.
        """
        query = select(Collection).where(Collection.id == collection_id).options(
            selectinload(Collection.documents)
        )
        col = self.db.scalar(query)
        if not col:
            return None

        docs = col.documents
        doc_count = len(docs)
        total_size = sum(d.file_size for d in docs if d.file_size)

        created_times = [d.created_at for d in docs if d.created_at]
        first_uploaded = min(created_times) if created_times else None
        last_uploaded = max(created_times) if created_times else None

        owner_email = col.owner.email if col.owner else "N/A"
        workspace_name = col.workspace.name if col.workspace else "N/A"

        return {
            "document_count": doc_count,
            "total_size": total_size,
            "first_uploaded": first_uploaded,
            "last_uploaded": last_uploaded,
            "last_modified": col.updated_at or col.created_at,
            "owner_email": owner_email,
            "workspace_name": workspace_name
        }
