# tests/test_collection_retrieval.py
# ---------------------------------
# Unit tests for CollectionFilterService and collection-aware retrieval pipeline filtering.

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Collection, Document
from app.services.collection.collection_filter_service import CollectionFilterService


class TestCollectionRetrieval(unittest.TestCase):
    """
    Unit test cases covering CollectionFilterService validations and document resolving properties.
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        # Pristine schema rebuild
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        self.db: Session = self.SessionLocal()
        self.filter_service = CollectionFilterService(self.db)

        # Users setup
        self.user1 = User(full_name="User 1", email="user1@retrieval.com", hashed_password="pw")
        self.user2 = User(full_name="User 2", email="user2@retrieval.com", hashed_password="pw")
        self.db.add_all([self.user1, self.user2])
        self.db.commit()

        # Workspaces setup
        self.ws1 = Workspace(owner_id=self.user1.id, name="WS 1")
        self.ws2 = Workspace(owner_id=self.user2.id, name="WS 2")
        self.db.add_all([self.ws1, self.ws2])
        self.db.commit()

        # Documents setup
        self.doc1 = Document(filename="d1.pdf", stored_filename="s1.pdf", file_extension="pdf", mime_type="pdf", file_size=10, file_path="/p1", owner_id=self.user1.id, workspace_id=self.ws1.id)
        self.doc2 = Document(filename="d2.pdf", stored_filename="s2.pdf", file_extension="pdf", mime_type="pdf", file_size=10, file_path="/p2", owner_id=self.user1.id, workspace_id=self.ws1.id)
        self.doc3 = Document(filename="d3.pdf", stored_filename="s3.pdf", file_extension="pdf", mime_type="pdf", file_size=10, file_path="/p3", owner_id=self.user2.id, workspace_id=self.ws2.id)
        self.db.add_all([self.doc1, self.doc2, self.doc3])
        self.db.commit()

        # Collections setup
        self.col1 = Collection(workspace_id=self.ws1.id, owner_id=self.user1.id, name="Col 1")
        self.col2 = Collection(workspace_id=self.ws1.id, owner_id=self.user1.id, name="Col 2")
        self.db.add_all([self.col1, self.col2])
        self.db.commit()

        # Add documents to collections
        self.col1.documents.append(self.doc1)
        self.col2.documents.append(self.doc2)
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_single_collection_retrieval(self):
        resolved = self.filter_service.validate_and_resolve_filters(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            collection_ids=[self.col1.id]
        )
        self.assertEqual(resolved, [self.doc1.id])

    def test_multiple_collection_retrieval(self):
        resolved = self.filter_service.validate_and_resolve_filters(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            collection_ids=[self.col1.id, self.col2.id]
        )
        self.assertCountEqual(resolved, [self.doc1.id, self.doc2.id])

    def test_empty_collection_returns_negative_one(self):
        empty_col = Collection(workspace_id=self.ws1.id, owner_id=self.user1.id, name="Empty Col")
        self.db.add(empty_col)
        self.db.commit()

        resolved = self.filter_service.validate_and_resolve_filters(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            collection_ids=[empty_col.id]
        )
        self.assertEqual(resolved, [-1])

    def test_search_all_documents_in_workspace(self):
        # Passing empty collection_ids resolves to all documents in the active workspace
        resolved = self.filter_service.validate_and_resolve_filters(
            user_id=self.user1.id,
            workspace_id=self.ws1.id,
            collection_ids=None
        )
        self.assertCountEqual(resolved, [self.doc1.id, self.doc2.id])

    def test_workspace_isolation_and_permission_failures(self):
        # User 1 attempts to search using collection belonging to User 2 -> raises PermissionError
        foreign_col = Collection(workspace_id=self.ws2.id, owner_id=self.user2.id, name="Foreign Col")
        self.db.add(foreign_col)
        self.db.commit()

        with self.assertRaises(PermissionError):
            self.filter_service.validate_and_resolve_filters(
                user_id=self.user1.id,
                workspace_id=self.ws1.id,
                collection_ids=[foreign_col.id]
            )

        # Workspace mismatch error (collection from WS 2 passed with WS 1 scope)
        col_on_ws2_owned_by_user1 = Collection(workspace_id=self.ws2.id, owner_id=self.user1.id, name="Hack Col")
        # Note: If workspace ownership checks catch it first or collection checks
        with self.assertRaises(PermissionError):
            self.filter_service.validate_and_resolve_filters(
                user_id=self.user1.id,
                workspace_id=self.ws2.id,  # User 1 is not owner of WS 2
                collection_ids=[col_on_ws2_owned_by_user1.id]
            )


if __name__ == "__main__":
    unittest.main()
