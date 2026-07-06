# app/repositories/dashboard_repository.py
# ----------------------------------------
# Repository layer executing optimized database queries for dashboard metrics.

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.collection import Collection
from app.models.document_collection import DocumentCollection
from app.models.conversation import Conversation
from app.models.chat_message import ChatMessage
from app.models.search_history import SearchHistory

logger = logging.getLogger(__name__)


class DashboardRepository:
    """
    Repository class encapsulating SQL queries for the workspace dashboard.
    """
    def __init__(self, db: Session):
        self.db = db

    def get_workspace_overview(self, workspace_id: int, owner_id: int) -> Dict[str, Any]:
        """
        Retrieves overview of the active workspace, including owner and last activity date.
        """
        # Fetch workspace details
        from app.models.workspace import Workspace
        ws = self.db.get(Workspace, workspace_id)
        if not ws:
            raise KeyError("Workspace not found")

        owner = self.db.get(User, owner_id)
        owner_name = (owner.full_name or (owner.email.split("@")[0] if owner.email else "User")) if owner else "Unknown"

        # Calculate last activity timestamp as the max of created_at/updated_at across tables
        last_doc = self.db.scalar(select(func.max(Document.created_at)).where(Document.workspace_id == workspace_id))
        last_conv = self.db.scalar(select(func.max(Conversation.updated_at)).where(Conversation.workspace_id == workspace_id))
        last_search = self.db.scalar(select(func.max(SearchHistory.created_at)).where(SearchHistory.workspace_id == workspace_id))

        dates = [d for d in [last_doc, last_conv, last_search] if d is not None]
        last_activity = max(dates) if dates else ws.created_at

        return {
            "workspace_name": ws.name,
            "owner_name": owner_name,
            "created_at": ws.created_at,
            "last_activity_at": last_activity
        }

    def get_conversation_metrics(self, workspace_id: int) -> Dict[str, Any]:
        """
        Aggregates conversation counts, archived states, messages, and lengths.
        """
        total_conv = self.db.scalar(select(func.count(Conversation.id)).where(Conversation.workspace_id == workspace_id)) or 0
        active_conv = self.db.scalar(select(func.count(Conversation.id)).where(Conversation.workspace_id == workspace_id, Conversation.is_archived == False)) or 0
        archived_conv = total_conv - active_conv

        total_messages = self.db.scalar(
            select(func.count(ChatMessage.id))
            .join(Conversation, Conversation.id == ChatMessage.conversation_id)
            .where(Conversation.workspace_id == workspace_id)
        ) or 0

        avg_len = (total_messages / total_conv) if total_conv > 0 else 0.0

        return {
            "total_conversations": total_conv,
            "active_conversations": active_conv,
            "archived_conversations": archived_conv,
            "total_messages": total_messages,
            "average_conversation_length": round(avg_len, 2)
        }

    def get_collection_metrics(self, workspace_id: int) -> Dict[str, Any]:
        """
        Aggregates collections counts, largest collection size, and documents per collection.
        """
        total_coll = self.db.scalar(select(func.count(Collection.id)).where(Collection.workspace_id == workspace_id)) or 0

        # Largest collection
        largest_name = None
        largest_size = 0

        largest_res = self.db.execute(
            select(Collection.name, func.count(DocumentCollection.document_id).label("doc_count"))
            .join(DocumentCollection, Collection.id == DocumentCollection.collection_id)
            .where(Collection.workspace_id == workspace_id)
            .group_by(Collection.id)
            .order_by(desc("doc_count"))
            .limit(1)
        ).first()

        if largest_res:
            largest_name = largest_res[0]
            largest_size = largest_res[1]

        # Total collection-document mappings
        total_mapped = self.db.scalar(
            select(func.count(DocumentCollection.document_id))
            .join(Collection, Collection.id == DocumentCollection.collection_id)
            .where(Collection.workspace_id == workspace_id)
        ) or 0

        avg_docs = (total_mapped / total_coll) if total_coll > 0 else 0.0

        return {
            "total_collections": total_coll,
            "largest_collection_name": largest_name,
            "largest_collection_size": largest_size,
            "average_documents_per_collection": round(avg_docs, 2)
        }

    def get_search_metrics(self, workspace_id: int) -> Dict[str, Any]:
        """
        Aggregates workspace search counts (today, this week), frequencies, and latency.
        """
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        week_start = now - timedelta(days=7)

        base_filter = (SearchHistory.workspace_id == workspace_id,)

        searches_today = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_filter, SearchHistory.created_at >= today_start)) or 0
        searches_week = self.db.scalar(select(func.count(SearchHistory.id)).where(*base_filter, SearchHistory.created_at >= week_start)) or 0

        avg_latency = self.db.scalar(select(func.avg(SearchHistory.execution_time_ms)).where(*base_filter)) or 0.0

        # Most frequent query
        freq_res = self.db.execute(
            select(SearchHistory.query, func.count(SearchHistory.id).label("search_count"))
            .where(*base_filter)
            .group_by(SearchHistory.query)
            .order_by(desc("search_count"))
            .limit(1)
        ).first()

        most_frequent = freq_res[0] if freq_res else None

        return {
            "searches_today": searches_today,
            "searches_this_week": searches_week,
            "most_frequent_query": most_frequent,
            "average_retrieval_time_ms": round(float(avg_latency), 2)
        }

    def get_document_metrics(self, workspace_id: int) -> Dict[str, Any]:
        """
        Aggregates document upload statistics (counts, uploads today/week, chunks, embeddings).
        """
        now = datetime.utcnow()
        today_start = datetime(now.year, now.month, now.day)
        week_start = now - timedelta(days=7)

        total_docs = self.db.scalar(select(func.count(Document.id)).where(Document.workspace_id == workspace_id)) or 0
        uploaded_today = self.db.scalar(select(func.count(Document.id)).where(Document.workspace_id == workspace_id, Document.created_at >= today_start)) or 0
        uploaded_week = self.db.scalar(select(func.count(Document.id)).where(Document.workspace_id == workspace_id, Document.created_at >= week_start)) or 0

        total_chunks = self.db.scalar(
            select(func.count(DocumentChunk.id))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.workspace_id == workspace_id)
        ) or 0

        return {
            "uploaded_today": uploaded_today,
            "uploaded_this_week": uploaded_week,
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_embeddings": total_chunks  # Embeddings map 1-to-1 with document chunks
        }

    def _get_directory_size(self, path: str) -> int:
        """
        Helper calculating directory disk usage recursively in bytes.
        """
        total = 0
        if os.path.exists(path):
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += self._get_directory_size(entry.path)
            except Exception as e:
                logger.warning(f"Error reading path size for {path}: {e}")
        return total

    def get_storage_metrics(self, workspace_id: int) -> Dict[str, Any]:
        """
        Computes disk and metadata storage size metrics.
        """
        total_size = self.db.scalar(select(func.sum(Document.file_size)).where(Document.workspace_id == workspace_id)) or 0
        avg_size = self.db.scalar(select(func.avg(Document.file_size)).where(Document.workspace_id == workspace_id)) or 0.0

        # Largest document
        largest_doc = self.db.execute(
            select(Document.filename, Document.file_size)
            .where(Document.workspace_id == workspace_id)
            .order_by(desc(Document.file_size))
            .limit(1)
        ).first()

        largest_name = largest_doc[0] if largest_doc else None
        largest_bytes = largest_doc[1] if largest_doc else 0

        # Chunks count (embeddings)
        total_embeddings = self.db.scalar(
            select(func.count(DocumentChunk.id))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.workspace_id == workspace_id)
        ) or 0

        # Vector store disk usage (estimate or scan vector_store/chroma_db folders)
        vdb_size = self._get_directory_size("./vector_store") or self._get_directory_size("./chroma_db")

        return {
            "total_storage_bytes": int(total_size),
            "average_document_size_bytes": round(float(avg_size), 2),
            "largest_document_name": largest_name,
            "largest_document_size_bytes": largest_bytes,
            "total_embeddings": total_embeddings,
            "vector_db_size_bytes": vdb_size
        }

    def get_recent_activities(self, workspace_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Merges most recent actions (uploads, searches, conversations, collection updates) chronologically.
        """
        # Fetch individual feeds up to limit
        uploads = self.db.scalars(
            select(Document).where(Document.workspace_id == workspace_id)
            .order_by(desc(Document.created_at)).limit(limit)
        ).all()

        searches = self.db.scalars(
            select(SearchHistory).where(SearchHistory.workspace_id == workspace_id)
            .order_by(desc(SearchHistory.created_at)).limit(limit)
        ).all()

        conversations = self.db.scalars(
            select(Conversation).where(Conversation.workspace_id == workspace_id)
            .order_by(desc(Conversation.created_at)).limit(limit)
        ).all()

        collections = self.db.scalars(
            select(Collection).where(Collection.workspace_id == workspace_id)
            .order_by(desc(Collection.updated_at)).limit(limit)
        ).all()

        # Build activity payload objects
        activities = []
        for d in uploads:
            activities.append({
                "type": "document_upload",
                "description": f"Uploaded document: {d.filename}",
                "timestamp": d.created_at,
                "metadata": {"id": d.id, "filename": d.filename, "size": d.file_size}
            })
        for s in searches:
            activities.append({
                "type": "search_query",
                "description": f"Executed search query: '{s.query}'",
                "timestamp": s.created_at,
                "metadata": {"id": s.id, "query": s.query, "latency_ms": s.execution_time_ms}
            })
        for c in conversations:
            activities.append({
                "type": "conversation_start",
                "description": f"Started conversation: '{c.title or 'Untitled'}'",
                "timestamp": c.created_at,
                "metadata": {"id": c.id, "title": c.title}
            })
        for col in collections:
            activities.append({
                "type": "collection_update",
                "description": f"Updated collection: '{col.name}'",
                "timestamp": col.updated_at or col.created_at,
                "metadata": {"id": col.id, "name": col.name}
            })

        # Sort and return top N
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities[:limit]
