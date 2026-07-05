# tests/test_conversation_memory.py
# ---------------------------------
# Unit tests for the ConversationMemoryService and HistoryFormatter.

import os
import sys
import unittest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import User, Workspace, Conversation, ChatMessage
from app.models.chat_message import MessageRole
from app.services.conversation.history_formatter import HistoryFormatter
from app.services.conversation.conversation_memory_service import ConversationMemoryService
from app.ai.prompt_builder import PromptBuilder


class TestConversationMemory(unittest.TestCase):
    """
    Unit test cases covering formatted outputs, trimming constraints, user/workspace isolation, and prompts building.
    """
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db: Session = self.SessionLocal()
        self.memory_service = ConversationMemoryService(self.db)

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

    def test_history_formatter_chronological(self):
        # Create list of messages with manually set times/IDs to check sorting
        m1 = ChatMessage(role=MessageRole.USER, content="Hello", created_at=datetime.utcnow() - timedelta(minutes=2))
        m2 = ChatMessage(role=MessageRole.ASSISTANT, content="Hi there", created_at=datetime.utcnow() - timedelta(minutes=1))
        
        formatted = HistoryFormatter.format_history([m2, m1], limit=10)
        expected = "User: Hello\nAssistant: Hi there"
        self.assertEqual(formatted, expected)

    def test_history_formatter_trim_limit(self):
        m1 = ChatMessage(role=MessageRole.USER, content="One", id=1, created_at=datetime.utcnow() - timedelta(minutes=3))
        m2 = ChatMessage(role=MessageRole.ASSISTANT, content="Two", id=2, created_at=datetime.utcnow() - timedelta(minutes=2))
        m3 = ChatMessage(role=MessageRole.USER, content="Three", id=3, created_at=datetime.utcnow() - timedelta(minutes=1))

        # Cap limit to 2
        formatted = HistoryFormatter.format_history([m1, m2, m3], limit=2)
        expected = "Assistant: Two\nUser: Three"
        self.assertEqual(formatted, expected)

    def test_memory_service_load_and_token_trimming(self):
        conv = Conversation(workspace_id=self.ws.id, user_id=self.user.id, title="Memory Thread")
        self.db.add(conv)
        self.db.commit()

        # Add 3 messages with token counts
        msg1 = ChatMessage(conversation_id=conv.id, role=MessageRole.USER, content="Query One", token_count=100, created_at=datetime.utcnow() - timedelta(minutes=3))
        msg2 = ChatMessage(conversation_id=conv.id, role=MessageRole.ASSISTANT, content="Response One", token_count=200, created_at=datetime.utcnow() - timedelta(minutes=2))
        msg3 = ChatMessage(conversation_id=conv.id, role=MessageRole.USER, content="Query Two", token_count=150, created_at=datetime.utcnow() - timedelta(minutes=1))
        self.db.add_all([msg1, msg2, msg3])
        self.db.commit()

        # Load with large token budget - should return all 3 messages
        text1, tokens1 = self.memory_service.load_conversation_history(conv.id, self.user.id, max_tokens=1000)
        self.assertIn("Query One", text1)
        self.assertIn("Response One", text1)
        self.assertIn("Query Two", text1)
        self.assertEqual(tokens1, 450)

        # Load with small token budget (max_tokens=300)
        # Should drop Query One (100) and Response One (200), keeping only Query Two (150)
        # Or wait, if budget is 300, Query Two (150) + Response One (200) = 350 (over limit). So it breaks and returns only Query Two.
        text2, tokens2 = self.memory_service.load_conversation_history(conv.id, self.user.id, max_tokens=300)
        self.assertNotIn("Query One", text2)
        self.assertNotIn("Response One", text2)
        self.assertIn("Query Two", text2)
        self.assertEqual(tokens2, 150)

    def test_user_and_workspace_isolation(self):
        conv = Conversation(workspace_id=self.ws.id, user_id=self.user.id, title="Private Thread")
        self.db.add(conv)
        self.db.commit()

        # Loading by other user should raise PermissionError
        with self.assertRaises(PermissionError):
            self.memory_service.load_conversation_history(conv.id, self.other_user.id)

    def test_prompt_builder_integration(self):
        history = "User: How are you?\nAssistant: I am fine."
        chunks = [{"text": "Retrieval Knowledge content.", "metadata": {"filename": "doc.txt"}}]
        
        prompt = PromptBuilder.build_prompt(
            question="Is that correct?",
            chunks=chunks,
            conversation_history=history
        )
        
        self.assertIn("=== Conversation History ===", prompt)
        self.assertIn("User: How are you?", prompt)
        self.assertIn("Assistant: I am fine.", prompt)
        self.assertIn("=== Retrieved Knowledge ===", prompt)
        self.assertIn("Retrieval Knowledge content.", prompt)
        self.assertIn("=== Current Question ===", prompt)
        self.assertIn("Is that correct?", prompt)


if __name__ == "__main__":
    unittest.main()
