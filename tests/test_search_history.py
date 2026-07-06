# tests/test_search_history.py
# -----------------------------
# Unit tests for SearchHistoryService and SearchHistoryRepository.

import os
import sys
import time
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, SearchHistory
from app.services.search_history.search_history_service import SearchHistoryService
from app.services.workspace.workspace_service import WorkspaceService


class TestSearchHistory(unittest.TestCase):
    """
    Unit test suite validating search history CRUD, stats, limits, and isolation rules.
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
        self.service = SearchHistoryService(self.db)

        # Create test users
        self.user1 = User(full_name="Alice", email="alice@work.com", hashed_password="pw")
        self.user2 = User(full_name="Bob", email="bob@work.com", hashed_password="pw")
        self.db.add_all([self.user1, self.user2])
        self.db.commit()

        # Workspaces
        self.ws1 = self.ws_service.create_workspace(self.user1.id, "Workspace A")
        self.ws2 = self.ws_service.create_workspace(self.user1.id, "Workspace B")
        self.bob_ws = self.ws_service.create_workspace(self.user2.id, "Bob WS")

    def tearDown(self):
        self.db.close()

    def test_search_recording_and_recent(self):
        # 1. Record search inside Workspace A
        entry = self.service.record_search(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            query="LangChain Agentic Workflows",
            filters_json={"collection_ids": [1]},
            execution_time_ms=120,
            result_count=5
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry.query, "LangChain Agentic Workflows")
        self.assertEqual(entry.result_count, 5)

        # 2. Get recent searches - should return 1 entry
        recent = self.service.get_recent(self.user1.id, self.ws1.id)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].query, "LangChain Agentic Workflows")

        # 3. Workspace B recent searches should be empty (workspace isolation)
        recent_b = self.service.get_recent(self.user1.id, self.ws2.id)
        self.assertEqual(len(recent_b), 0)

    def test_frequent_searches(self):
        # Record identical queries to check frequency rank counting
        self.service.record_search(self.user1.id, self.ws1.id, "vector databases")
        self.service.record_search(self.user1.id, self.ws1.id, "vector databases")
        self.service.record_search(self.user1.id, self.ws1.id, "chromadb vs pinecone")
        self.service.record_search(self.user1.id, self.ws1.id, "vector databases")

        freq = self.service.get_frequent(self.user1.id, self.ws1.id)
        self.assertEqual(freq[0][0], "vector databases")
        self.assertEqual(freq[0][1], 3)
        self.assertEqual(freq[1][0], "chromadb vs pinecone")
        self.assertEqual(freq[1][1], 1)

    def test_statistics(self):
        self.service.record_search(self.user1.id, self.ws1.id, "short", execution_time_ms=100)
        self.service.record_search(self.user1.id, self.ws1.id, "longer query", execution_time_ms=200)

        stats = self.service.get_statistics(self.user1.id, self.ws1.id)
        self.assertEqual(stats["total_searches"], 2)
        self.assertEqual(stats["average_query_length"], 8.5)  # (5 + 12) / 2
        self.assertEqual(stats["average_latency_ms"], 150.0)  # (100 + 200) / 2

    def test_pagination(self):
        # Record 25 entries
        for i in range(25):
            self.service.record_search(self.user1.id, self.ws1.id, f"query {i}")

        items, total = self.service.list_history(self.user1.id, self.ws1.id, page=2, page_size=10)
        self.assertEqual(total, 25)
        self.assertEqual(len(items), 10)
        # page 2 should have items index 10 to 19

    def test_delete_and_clear(self):
        entry = self.service.record_search(self.user1.id, self.ws1.id, "query to delete")
        
        # Bob tries to delete Alice's entry -> raises PermissionError
        with self.assertRaises(PermissionError):
            self.service.delete_history(entry.id, self.user2.id)

        # Alice deletes it -> success
        success = self.service.delete_history(entry.id, self.user1.id)
        self.assertTrue(success)
        self.assertIsNone(self.db.get(SearchHistory, entry.id))

        # Record another entry and clear all
        self.service.record_search(self.user1.id, self.ws1.id, "clear me")
        self.service.clear_history(self.user1.id, self.ws1.id)
        recent = self.service.get_recent(self.user1.id, self.ws1.id)
        self.assertEqual(len(recent), 0)


if __name__ == "__main__":
    unittest.main()
