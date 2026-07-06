# tests/test_dashboard.py
# -----------------------
# Unit tests for DashboardService and DashboardRepository.

import os
import sys
import unittest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Document, DocumentChunk, Collection, Conversation, ChatMessage, SearchHistory
from app.services.dashboard.dashboard_service import DashboardService
from app.services.workspace.workspace_service import WorkspaceService


class TestDashboard(unittest.TestCase):
    """
    Unit tests for dashboard metrics aggregation and workspace security validation.
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        self.db: Session = self.SessionLocal()
        self.ws_service = WorkspaceService(self.db)
        self.service = DashboardService(self.db)

        # Create test users
        self.user1 = User(full_name="Alice", email="alice@test.com", hashed_password="pw")
        self.user2 = User(full_name="Bob", email="bob@test.com", hashed_password="pw")
        self.db.add_all([self.user1, self.user2])
        self.db.commit()

        # Workspaces
        self.ws1 = self.ws_service.create_workspace(self.user1.id, "Workspace A")
        self.ws2 = self.ws_service.create_workspace(self.user1.id, "Workspace B")
        self.bob_ws = self.ws_service.create_workspace(self.user2.id, "Bob WS")

    def tearDown(self):
        self.db.close()

    def test_empty_workspace_dashboard(self):
        # Build dashboard for empty workspace - should return default metrics safely without errors
        res = self.service.build_dashboard_summary(self.ws1.id, self.user1.id)
        self.assertIsNotNone(res)
        self.assertEqual(res["overview"]["workspace_name"], "Workspace A")
        self.assertEqual(res["document_metrics"]["total_documents"], 0)
        self.assertEqual(res["conversation_metrics"]["total_conversations"], 0)
        self.assertEqual(res["collection_metrics"]["total_collections"], 0)
        self.assertEqual(res["storage_metrics"]["total_storage_bytes"], 0)

    def test_metrics_aggregation(self):
        # Create a document
        doc1 = Document(
            filename="report.pdf",
            stored_filename="report_stored.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=1024,
            file_path="/path/to/report.pdf",
            owner_id=self.user1.id,
            workspace_id=self.ws1.id
        )
        self.db.add(doc1)
        self.db.commit()

        # Add chunk
        chunk = DocumentChunk(
            document_id=doc1.id,
            chunk_number=0,
            chunk_text="Knowledge chunks content text here.",
            character_count=35
        )
        self.db.add(chunk)

        # Create collections
        coll = Collection(
            workspace_id=self.ws1.id,
            owner_id=self.user1.id,
            name="Research Docs",
            description="Research"
        )
        self.db.add(coll)
        self.db.commit()

        # Map document to collection
        from app.models.document_collection import DocumentCollection
        mapping = DocumentCollection(document_id=doc1.id, collection_id=coll.id)
        self.db.add(mapping)

        # Create conversation and message
        conv = Conversation(
            workspace_id=self.ws1.id,
            user_id=self.user1.id,
            title="Chat 1"
        )
        self.db.add(conv)
        self.db.commit()

        msg = ChatMessage(
            conversation_id=conv.id,
            role="USER",
            content="Hello AI",
            token_count=2
        )
        self.db.add(msg)

        # Create search history
        search = SearchHistory(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            query="test query",
            execution_time_ms=80,
            result_count=1
        )
        self.db.add(search)
        self.db.commit()

        # Test aggregates
        res = self.service.build_dashboard_summary(self.ws1.id, self.user1.id)
        self.assertEqual(res["document_metrics"]["total_documents"], 1)
        self.assertEqual(res["document_metrics"]["total_chunks"], 1)
        self.assertEqual(res["collection_metrics"]["total_collections"], 1)
        self.assertEqual(res["collection_metrics"]["largest_collection_size"], 1)
        self.assertEqual(res["conversation_metrics"]["total_conversations"], 1)
        self.assertEqual(res["conversation_metrics"]["total_messages"], 1)
        self.assertEqual(res["search_metrics"]["searches_today"], 1)
        self.assertEqual(res["storage_metrics"]["total_storage_bytes"], 1024)

    def test_workspace_security_boundaries(self):
        # Alice tries to view Bob's workspace statistics -> PermissionError
        with self.assertRaises(PermissionError):
            self.service.build_dashboard_summary(self.bob_ws.id, self.user1.id)

        # Bob tries to view Alice's workspace statistics -> PermissionError
        with self.assertRaises(PermissionError):
            self.service.build_dashboard_summary(self.ws1.id, self.user2.id)

    def test_activity_limits(self):
        # Record 25 search queries
        for i in range(25):
            search = SearchHistory(
                user_id=self.user1.id,
                workspace_id=self.ws1.id,
                query=f"query {i}",
                execution_time_ms=50,
                result_count=1
            )
            self.db.add(search)
        self.db.commit()

        # Build dashboard summary
        res = self.service.build_dashboard_summary(self.ws1.id, self.user1.id)
        # Should be capped at DASHBOARD_ACTIVITY_LIMIT which is 20 by default
        self.assertEqual(len(res["recent_activity"]), 20)


if __name__ == "__main__":
    unittest.main()
