# tests/test_workspace_service.py
# -------------------------------
# Unit tests for WorkspaceService and WorkspaceRepository.

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Document, Collection, Conversation
from app.services.workspace.workspace_service import WorkspaceService


class TestWorkspaceService(unittest.TestCase):
    """
    Unit test cases for Workspace operations and validation rules.
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

        self.db: Session = self.SessionLocal()
        self.service = WorkspaceService(self.db)

        # Create test users
        self.user1 = User(full_name="Alice", email="alice@work.com", hashed_password="pw")
        self.user2 = User(full_name="Bob", email="bob@work.com", hashed_password="pw")
        self.db.add_all([self.user1, self.user2])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_workspace_creation_and_defaults(self):
        # 1. First workspace creation should auto-assign as default
        ws1 = self.service.create_workspace(self.user1.id, "First WS", "Alice's main workspace")
        self.assertEqual(ws1.name, "First WS")
        self.assertTrue(ws1.is_default)

        # 2. Second workspace creation should NOT be default
        ws2 = self.service.create_workspace(self.user1.id, "Second WS")
        self.assertEqual(ws2.name, "Second WS")
        self.assertFalse(ws2.is_default)

        # 3. Setting second workspace as default should swap default flags
        self.service.set_default_workspace(self.user1.id, ws2.id)
        
        self.db.refresh(ws1)
        self.db.refresh(ws2)
        self.assertFalse(ws1.is_default)
        self.assertTrue(ws2.is_default)

    def test_duplicate_workspace_names_prevented(self):
        self.service.create_workspace(self.user1.id, "Unique Name")
        
        # Duplicate name for same user should fail
        with self.assertRaises(ValueError):
            self.service.create_workspace(self.user1.id, "Unique Name")

        # Same name for different user should succeed
        ws_bob = self.service.create_workspace(self.user2.id, "Unique Name")
        self.assertEqual(ws_bob.name, "Unique Name")

    def test_update_workspace(self):
        ws = self.service.create_workspace(self.user1.id, "Main")
        updated = self.service.update_workspace(ws.id, self.user1.id, name="Main Updated", description="New Desc")
        self.assertEqual(updated.name, "Main Updated")
        self.assertEqual(updated.description, "New Desc")

    def test_delete_workspace_restrictions(self):
        ws1 = self.service.create_workspace(self.user1.id, "Default WS")
        
        # 1. Cannot delete when it is the user's only workspace
        with self.assertRaises(ValueError):
            self.service.delete_workspace(ws1.id, self.user1.id)

        # Create second workspace
        ws2 = self.service.create_workspace(self.user1.id, "Other WS")

        # 2. Cannot delete default workspace
        with self.assertRaises(ValueError):
            self.service.delete_workspace(ws1.id, self.user1.id)

        # 3. Deleting non-default workspace should succeed
        success = self.service.delete_workspace(ws2.id, self.user1.id)
        self.assertTrue(success)
        self.assertEqual(self.service.repo.count_for_user(self.user1.id), 1)

    def test_workspace_ownership_and_validation(self):
        ws_alice = self.service.create_workspace(self.user1.id, "Alice WS")
        
        # Bob tries to access Alice's workspace -> raises PermissionError
        with self.assertRaises(PermissionError):
            self.service.get_workspace(ws_alice.id, self.user2.id)

        # Bob tries to update Alice's workspace -> raises PermissionError
        with self.assertRaises(PermissionError):
            self.service.update_workspace(ws_alice.id, self.user2.id, name="Hacked")

    def test_workspace_statistics(self):
        ws = self.service.create_workspace(self.user1.id, "Analytics WS")
        
        doc = Document(filename="file.txt", stored_filename="s.txt", file_extension="txt", mime_type="text/plain", file_size=120, file_path="/p", owner_id=self.user1.id, workspace_id=ws.id)
        col = Collection(workspace_id=ws.id, owner_id=self.user1.id, name="My Collection")
        conv = Conversation(workspace_id=ws.id, user_id=self.user1.id, title="Test thread")
        self.db.add_all([doc, col, conv])
        self.db.commit()

        stats = self.service.get_workspace_statistics(ws.id, self.user1.id)
        self.assertEqual(stats["document_count"], 1)
        self.assertEqual(stats["collection_count"], 1)
        self.assertEqual(stats["conversation_count"], 1)
        self.assertEqual(stats["storage_usage"], 120)


if __name__ == "__main__":
    unittest.main()
