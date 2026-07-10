# tests/test_conversation_service.py
# ----------------------------------
# Unit tests for the ConversationService and ConversationRepository.

import os
import sys
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Conversation, ChatMessage
from app.services.conversation.conversation_service import ConversationService
from app.repositories.conversation_repository import ConversationRepository
from app.models.chat_message import MessageRole


class TestConversationService(unittest.TestCase):
    """
    Unit test cases for Phase 9.2 backend conversation management services.
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db: Session = self.SessionLocal()
        self.service = ConversationService(self.db)
        self.repo = ConversationRepository(self.db)

        # Base user setups
        self.user1 = User(full_name="User One", email="user1@test.com", hashed_password="pw")
        self.user2 = User(full_name="User Two", email="user2@test.com", hashed_password="pw")
        self.db.add(self.user1)
        self.db.add(self.user2)
        self.db.commit()

        # Workspace setups
        self.ws1 = Workspace(owner_id=self.user1.id, name="WS One", is_default=True)
        self.ws2 = Workspace(owner_id=self.user2.id, name="WS Two", is_default=False)
        self.db.add(self.ws1)
        self.db.add(self.ws2)
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        # Drop all rows to isolate tests
        self.db.query(ChatMessage).delete()
        self.db.query(Conversation).delete()
        self.db.query(Workspace).delete()
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()

    def test_create_conversation_explicit_workspace(self):
        # Create conversation specifying ws1
        conv = self.service.create_conversation(user_id=self.user1.id, workspace_id=self.ws1.id, title="Test Chat")
        self.assertIsNotNone(conv.id)
        self.assertEqual(conv.title, "Test Chat")
        self.assertEqual(conv.workspace_id, self.ws1.id)
        self.assertEqual(conv.user_id, self.user1.id)

    def test_create_conversation_auto_workspace(self):
        # Create conversation without specifying workspace_id
        # Should pick the user's first workspace (ws1)
        conv = self.service.create_conversation(user_id=self.user1.id, title="Auto Chat")
        self.assertEqual(conv.workspace_id, self.ws1.id)

    def test_get_conversation_ownership_checks(self):
        conv = self.service.create_conversation(user_id=self.user1.id, workspace_id=self.ws1.id)
        
        # Owner fetches it - should work
        fetched = self.service.get_conversation(conv.id, self.user1.id)
        self.assertEqual(fetched.id, conv.id)

        # Other user fetches it - should raise PermissionError
        with self.assertRaises(PermissionError):
            self.service.get_conversation(conv.id, self.user2.id)

        # Non-existent conversation - should raise KeyError
        with self.assertRaises(KeyError):
            self.service.get_conversation(99999, self.user1.id)

    def test_append_messages(self):
        conv = self.service.create_conversation(user_id=self.user1.id, workspace_id=self.ws1.id)
        
        # Append message
        msg1 = self.service.append_message(
            conversation_id=conv.id,
            user_id=self.user1.id,
            role="user",
            content="Hello AI",
            token_count=10
        )
        self.assertEqual(msg1.content, "Hello AI")
        self.assertEqual(msg1.role, MessageRole.USER)
        self.assertEqual(msg1.token_count, 10)

        # Fetch conversation again to verify message in thread lists
        fetched = self.service.get_conversation(conv.id, self.user1.id)
        self.assertEqual(len(fetched.messages), 1)
        self.assertEqual(fetched.messages[0].content, "Hello AI")

    def test_pagination(self):
        # Create 5 conversations
        for i in range(5):
            self.service.create_conversation(user_id=self.user1.id, workspace_id=self.ws1.id, title=f"Chat {i}")

        # List first page (size 3)
        items1, total = self.service.list_conversations(user_id=self.user1.id, page=1, page_size=3)
        self.assertEqual(len(items1), 3)
        self.assertEqual(total, 5)

        # List second page (size 3)
        items2, total = self.service.list_conversations(user_id=self.user1.id, page=2, page_size=3)
        self.assertEqual(len(items2), 2)

    def test_soft_delete_restore_and_hard_delete(self):
        conv = self.service.create_conversation(user_id=self.user1.id, workspace_id=self.ws1.id)
        
        # 1. Soft delete
        self.service.delete_conversation(conv.id, self.user1.id)
        
        # Should not show in active lists
        items, total = self.service.list_conversations(user_id=self.user1.id, page=1, page_size=20)
        self.assertEqual(total, 0)
        
        # Should raise KeyError when fetching active conversation
        with self.assertRaises(KeyError):
            self.service.get_conversation(conv.id, self.user1.id)

        # Can be fetched if include_deleted=True
        fetched_del = self.service.get_conversation(conv.id, self.user1.id, include_deleted=True)
        self.assertIsNotNone(fetched_del.deleted_at)

        # 2. Restore
        self.service.restore_conversation(conv.id, self.user1.id)
        fetched_rest = self.service.get_conversation(conv.id, self.user1.id)
        self.assertIsNone(fetched_rest.deleted_at)

        # 3. Permanent delete
        self.service.permanently_delete(conv.id, self.user1.id)
        
        # Fetching even with include_deleted=True should raise KeyError
        with self.assertRaises(KeyError):
            self.service.get_conversation(conv.id, self.user1.id, include_deleted=True)


if __name__ == "__main__":
    unittest.main()
