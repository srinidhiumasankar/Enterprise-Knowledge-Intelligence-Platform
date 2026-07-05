# tests/test_collection_service.py
# -------------------------------
# Unit tests for CollectionService and CollectionRepository.

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Collection, Document
from app.services.collection.collection_service import CollectionService
from app.repositories.collection_repository import CollectionRepository


class TestCollectionService(unittest.TestCase):
    """
    Unit test cases covering collection operations (create, update, delete, link/unlink, stats, and checks).
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db: Session = self.SessionLocal()
        self.service = CollectionService(self.db)
        self.repo = CollectionRepository(self.db)

        # Base user setups
        self.user1 = User(full_name="Alice", email="alice@collection.com", hashed_password="pw")
        self.user2 = User(full_name="Bob", email="bob@collection.com", hashed_password="pw")
        self.db.add(self.user1)
        self.db.add(self.user2)
        self.db.commit()

        # Workspace setups
        self.ws1 = Workspace(owner_id=self.user1.id, name="Workspace 1")
        self.ws2 = Workspace(owner_id=self.user2.id, name="Workspace 2")
        self.db.add(self.ws1)
        self.db.add(self.ws2)
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        self.db.query(Document).delete()
        self.db.query(Collection).delete()
        self.db.query(Workspace).delete()
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()

    def test_create_and_update_collection(self):
        # 1. Create collection
        col = self.service.create_collection(self.user1.id, "Reports", "Annual Reports", self.ws1.id)
        self.assertEqual(col.name, "Reports")
        self.assertEqual(col.workspace_id, self.ws1.id)

        # 2. Prevent duplicates in workspace
        with self.assertRaises(ValueError):
            self.service.create_collection(self.user1.id, "Reports", "Another", self.ws1.id)

        # 3. Update collection
        updated = self.service.update_collection(col.id, self.user1.id, name="New Reports")
        self.assertEqual(updated.name, "New Reports")

    def test_delete_collection_preserves_documents(self):
        # Create collection and document
        col = self.service.create_collection(self.user1.id, "Reports", None, self.ws1.id)
        doc = Document(
            filename="rep.pdf",
            stored_filename="stored.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=100,
            file_path="/path",
            owner_id=self.user1.id,
            workspace_id=self.ws1.id
        )
        self.db.add(doc)
        self.db.commit()

        # Link document to collection
        self.service.add_document(col.id, self.user1.id, doc.id)

        # Delete collection
        self.service.delete_collection(col.id, self.user1.id)

        # Verify collection is gone, but document remains!
        self.assertIsNone(self.repo.get(col.id))
        self.assertIsNotNone(self.db.get(Document, doc.id))

    def test_add_and_remove_document(self):
        col = self.service.create_collection(self.user1.id, "Reports", None, self.ws1.id)
        doc = Document(
            filename="doc1.pdf",
            stored_filename="s1.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=250,
            file_path="/path",
            owner_id=self.user1.id,
            workspace_id=self.ws1.id
        )
        self.db.add(doc)
        self.db.commit()

        # Add document to collection
        self.service.add_document(col.id, self.user1.id, doc.id)
        docs = self.service.get_documents(col.id, self.user1.id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, doc.id)

        # Re-adding should fail
        with self.assertRaises(ValueError):
            self.service.add_document(col.id, self.user1.id, doc.id)

        # Remove document
        self.service.remove_document(col.id, self.user1.id, doc.id)
        docs_after = self.service.get_documents(col.id, self.user1.id)
        self.assertEqual(len(docs_after), 0)

    def test_list_and_pagination(self):
        for i in range(5):
            self.service.create_collection(self.user1.id, f"Col {i}", None, self.ws1.id)

        items, total = self.service.list_collections(self.user1.id, page=1, page_size=3)
        self.assertEqual(len(items), 3)
        self.assertEqual(total, 5)

    def test_statistics(self):
        col = self.service.create_collection(self.user1.id, "Reports", None, self.ws1.id)
        doc1 = Document(filename="d1.pdf", stored_filename="s1.pdf", file_extension="pdf", mime_type="application/pdf", file_size=100, file_path="/path", owner_id=self.user1.id, workspace_id=self.ws1.id)
        doc2 = Document(filename="d2.pdf", stored_filename="s2.pdf", file_extension="pdf", mime_type="application/pdf", file_size=150, file_path="/path", owner_id=self.user1.id, workspace_id=self.ws1.id)
        self.db.add_all([doc1, doc2])
        self.db.commit()

        self.service.add_document(col.id, self.user1.id, doc1.id)
        self.service.add_document(col.id, self.user1.id, doc2.id)

        stats = self.service.get_statistics(col.id, self.user1.id)
        self.assertEqual(stats["document_count"], 2)
        self.assertEqual(stats["total_size"], 250)
        self.assertEqual(stats["owner_email"], "alice@collection.com")
        self.assertEqual(stats["workspace_name"], "Workspace 1")

    def test_ownership_and_workspace_boundary_validations(self):
        col = self.service.create_collection(self.user1.id, "Reports", None, self.ws1.id)
        
        # Document in different workspace/owner (ws2 / user2)
        doc = Document(
            filename="foreign.pdf",
            stored_filename="sf.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=500,
            file_path="/path",
            owner_id=self.user2.id,
            workspace_id=self.ws2.id
        )
        self.db.add(doc)
        self.db.commit()

        # User 1 attempts to add User 2's document -> raises PermissionError
        with self.assertRaises(PermissionError):
            self.service.add_document(col.id, self.user1.id, doc.id)

        # User 2 attempts to retrieve User 1's collection -> raises PermissionError
        with self.assertRaises(PermissionError):
            self.service.get_collection(col.id, self.user2.id)


if __name__ == "__main__":
    unittest.main()
