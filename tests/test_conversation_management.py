# tests/test_conversation_management.py
# -------------------------------------
# Unit tests for the ConversationService and ConversationRepository management extensions (rename, pin, archive, search).

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


class TestConversationManagement(unittest.TestCase):
    """
    Unit test cases covering conversation organizations (rename, pin, archive, search, pagination).
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

        # User setups
        self.user = User(full_name="Tester", email="test@test.com", hashed_password="pw")
        self.other_user = User(full_name="Other", email="other@test.com", hashed_password="pw")
        self.db.add(self.user)
        self.db.add(self.other_user)
        self.db.commit()

        # Workspace setups
        self.ws = Workspace(owner_id=self.user.id, name="Test WS")
        self.db.add(self.ws)
        self.db.commit()

    def tearDown(self):
        self.db.rollback()
        self.db.query(ChatMessage).delete()
        self.db.query(Conversation).delete()
        self.db.query(Workspace).delete()
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()

    def test_rename_conversation(self):
        conv = self.service.create_conversation(user_id=self.user.id, workspace_id=self.ws.id, title="Old Title")
        self.assertEqual(conv.title, "Old Title")

        # Rename
        renamed = self.service.rename_conversation(conv.id, self.user.id, "New Title")
        self.assertEqual(renamed.title, "New Title")
        
        # Verify DB persist
        fetched = self.service.get_conversation(conv.id, self.user.id)
        self.assertEqual(fetched.title, "New Title")

        # Verify unauthorized user rename fails
        with self.assertRaises(PermissionError):
            self.service.rename_conversation(conv.id, self.other_user.id, "Hack Title")

    def test_pin_and_unpin_conversation(self):
        conv = self.service.create_conversation(user_id=self.user.id, workspace_id=self.ws.id, title="Thread 1")
        self.assertFalse(conv.is_pinned)

        # Pin
        self.service.pin_conversation(conv.id, self.user.id)
        pinned_list = self.service.get_pinned_conversations(self.user.id)
        self.assertEqual(len(pinned_list), 1)
        self.assertEqual(pinned_list[0].id, conv.id)

        # Unpin
        self.service.unpin_conversation(conv.id, self.user.id)
        pinned_list_after = self.service.get_pinned_conversations(self.user.id)
        self.assertEqual(len(pinned_list_after), 0)

    def test_archive_and_unarchive_conversation(self):
        conv = self.service.create_conversation(user_id=self.user.id, workspace_id=self.ws.id, title="Thread 2")
        self.assertFalse(conv.is_archived)

        # Archive
        self.service.archive_conversation(conv.id, self.user.id)
        
        # Normal list should exclude archived
        items, total = self.service.list_conversations(self.user.id)
        self.assertEqual(total, 0)

        # Archived list should contain it
        archived_list = self.service.get_archived_conversations(self.user.id)
        self.assertEqual(len(archived_list), 1)
        self.assertEqual(archived_list[0].id, conv.id)

        # Unarchive
        self.service.unarchive_conversation(conv.id, self.user.id)
        
        # Normal list should contain it again
        items_after, total_after = self.service.list_conversations(self.user.id)
        self.assertEqual(total_after, 1)

    def test_search_conversations(self):
        # Create conversations
        conv1 = self.service.create_conversation(user_id=self.user.id, workspace_id=self.ws.id, title="Pineapple FAQ")
        conv2 = self.service.create_conversation(user_id=self.user.id, workspace_id=self.ws.id, title="Banana FAQ")
        
        # Add messages
        self.service.append_message(conv1.id, self.user.id, "user", "I like sweet fruits like apples and pineapples.")
        self.service.append_message(conv2.id, self.user.id, "user", "Do you like yellow bananas?")

        # 1. Search title match
        items, total = self.service.search_conversations(self.user.id, "Pineapple")
        self.assertEqual(total, 1)
        self.assertEqual(items[0].id, conv1.id)

        # 2. Search message body match
        items_msg, total_msg = self.service.search_conversations(self.user.id, "yellow")
        self.assertEqual(total_msg, 1)
        self.assertEqual(items_msg[0].id, conv2.id)

        # 3. Search case-insensitive match
        items_case, total_case = self.service.search_conversations(self.user.id, "BANANA")
        self.assertEqual(total_case, 1)

        # 4. Search no match
        items_none, total_none = self.service.search_conversations(self.user.id, "xyz123")
        self.assertEqual(total_none, 0)


if __name__ == "__main__":
    unittest.main()
