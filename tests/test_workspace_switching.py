# tests/test_workspace_switching.py
# ---------------------------------
# Unit tests for WorkspaceContextService and workspace switching functionality.

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Collection, Conversation, UserPreference
from app.services.workspace.workspace_service import WorkspaceService
from app.services.workspace.workspace_context_service import WorkspaceContextService
from app.services.conversation.conversation_service import ConversationService
from app.services.collection.collection_service import CollectionService


class TestWorkspaceSwitching(unittest.TestCase):
    """
    Unit tests validating Workspace Switching flow, request-scoped context caching, and isolation.
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
        self.ctx_service = WorkspaceContextService(self.db)
        self.conv_service = ConversationService(self.db)
        self.col_service = CollectionService(self.db)

        # Create test users
        self.user_alice = User(full_name="Alice", email="alice@work.com", hashed_password="pw")
        self.user_bob = User(full_name="Bob", email="bob@work.com", hashed_password="pw")
        self.db.add_all([self.user_alice, self.user_bob])
        self.db.commit()

        # Workspace provision for Alice
        self.alice_ws1 = self.ws_service.create_workspace(self.user_alice.id, "Workspace One")
        self.alice_ws2 = self.ws_service.create_workspace(self.user_alice.id, "Workspace Two")

        # Workspace provision for Bob
        self.bob_ws = self.ws_service.create_workspace(self.user_bob.id, "Bob Workspace")

    def tearDown(self):
        self.ctx_service.clear_context()
        self.db.close()

    def test_active_workspace_retrieval(self):
        # Alice's active workspace should default to her default workspace (alice_ws1)
        active = self.ctx_service.get_active_workspace(self.user_alice.id)
        self.assertEqual(active.id, self.alice_ws1.id)

    def test_workspace_switching(self):
        # Switch Alice to Workspace Two
        switched = self.ctx_service.set_active_workspace(self.user_alice.id, self.alice_ws2.id)
        self.assertEqual(switched.id, self.alice_ws2.id)

        # Verify active workspace retrieval reflects the switch
        active = self.ctx_service.get_active_workspace(self.user_alice.id)
        self.assertEqual(active.id, self.alice_ws2.id)

    def test_context_caching(self):
        # Clear context to start fresh
        self.ctx_service.clear_context()
        
        # Get active workspace (first lookup caches it)
        active1 = self.ctx_service.get_active_workspace(self.user_alice.id)
        
        # Modify cached state in contextVar directly
        dummy = Workspace(id=999, owner_id=self.user_alice.id, name="Dummy")
        from app.services.workspace.workspace_context_service import _active_workspace_context
        _active_workspace_context.set(dummy)

        # Next lookup should pull from cache (returning dummy)
        active2 = self.ctx_service.get_active_workspace(self.user_alice.id)
        self.assertEqual(active2.id, 999)

        # Clear context
        self.ctx_service.clear_context()
        active3 = self.ctx_service.get_active_workspace(self.user_alice.id)
        self.assertEqual(active3.id, self.alice_ws1.id)

    def test_security_validation(self):
        # Alice tries to switch to Bob's workspace -> PermissionError
        with self.assertRaises(PermissionError):
            self.ctx_service.set_active_workspace(self.user_alice.id, self.bob_ws.id)

        # Alice tries to switch to non-existent workspace -> KeyError
        with self.assertRaises(KeyError):
            self.ctx_service.set_active_workspace(self.user_alice.id, 99999)

    def test_conversation_compatibility(self):
        # Ensure active workspace context translates to conversation default context
        self.ctx_service.set_active_workspace(self.user_alice.id, self.alice_ws2.id)
        
        # Create conversation with None workspace_id -> should use active (alice_ws2)
        conv = self.conv_service.create_conversation(self.user_alice.id, workspace_id=None, title="Thread 1")
        self.assertEqual(conv.workspace_id, self.alice_ws2.id)

    def test_collection_compatibility(self):
        # Ensure active workspace context translates to collection default context
        self.ctx_service.set_active_workspace(self.user_alice.id, self.alice_ws2.id)

        # Create collection with None workspace_id -> should use active (alice_ws2)
        col = self.col_service.create_collection(self.user_alice.id, "Col 1", workspace_id=None)
        self.assertEqual(col.workspace_id, self.alice_ws2.id)


if __name__ == "__main__":
    unittest.main()
