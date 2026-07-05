# tests/test_phase_9_1_models.py
# ------------------------------
# Unit tests for Phase 9.1 database models, relationships, and cascades.

import os
import sys
import unittest
import uuid
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import Base
from app.models import (
    User, Document, Workspace, Conversation,
    ChatMessage, Collection, DocumentCollection, SearchHistory, UserPreference
)
from app.models.chat_message import MessageRole


class TestPhase91Models(unittest.TestCase):
    """
    Unit tests covering schema integrity, properties, and relationship cascades for new tables.
    """
    @classmethod
    def setUpClass(cls):
        # Create an in-memory SQLite database to test declarations
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db: Session = self.SessionLocal()

    def tearDown(self):
        self.db.rollback()
        self.db.close()

    def test_model_creation_and_fields(self):
        # 1. User
        user = User(full_name="Alice Smith", email="alice@test.com", hashed_password="hashed")
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        self.assertIsNotNone(user.id)
        self.assertEqual(len(user.uuid), 36)

        # 2. Workspace
        workspace = Workspace(owner_id=user.id, name="Alice's Workspace", description="Personal space")
        self.db.add(workspace)
        self.db.commit()
        self.db.refresh(workspace)

        self.assertIsNotNone(workspace.id)
        self.assertEqual(len(workspace.uuid), 36)
        self.assertEqual(workspace.is_default, False)
        self.assertEqual(workspace.is_active, True)
        self.assertIsNotNone(workspace.created_at)
        self.assertIsNotNone(workspace.updated_at)

        # 3. UserPreference
        pref = UserPreference(user_id=user.id, theme="dark", default_workspace=workspace.id, temperature=0.7)
        self.db.add(pref)
        self.db.commit()
        self.db.refresh(pref)

        self.assertEqual(pref.theme, "dark")
        self.assertEqual(pref.temperature, 0.7)

        # 4. Conversation
        conv = Conversation(workspace_id=workspace.id, user_id=user.id, title="QA Session")
        self.db.add(conv)
        self.db.commit()
        self.db.refresh(conv)

        self.assertEqual(conv.title, "QA Session")
        self.assertIsNone(conv.deleted_at)  # SoftDeleteMixin check

        # 5. ChatMessage
        msg = ChatMessage(conversation_id=conv.id, role=MessageRole.USER, content="Hello!")
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)

        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello!")

        # 6. Collection
        col = Collection(workspace_id=workspace.id, owner_id=user.id, name="AI Docs")
        self.db.add(col)
        self.db.commit()
        self.db.refresh(col)

        self.assertEqual(col.name, "AI Docs")

        # 7. Document & DocumentCollection mapping
        doc = Document(
            filename="ai.pdf",
            stored_filename="stored.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=1024,
            file_path="/path",
            owner_id=user.id,
            workspace_id=workspace.id
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        self.assertEqual(doc.filename, "ai.pdf")

        # Associate doc to collection
        col.documents.append(doc)
        self.db.commit()
        
        # Verify collection document retrieval works
        self.assertEqual(len(col.documents), 1)
        self.assertEqual(col.documents[0].id, doc.id)

        # 8. SearchHistory
        hist = SearchHistory(user_id=user.id, workspace_id=workspace.id, query="What is AI?")
        self.db.add(hist)
        self.db.commit()
        self.db.refresh(hist)

        self.assertEqual(hist.query, "What is AI?")

    def test_relationships_and_cascades(self):
        # Create records
        user = User(full_name="Bob Jones", email="bob@test.com", hashed_password="hashed")
        self.db.add(user)
        self.db.commit()

        workspace = Workspace(owner_id=user.id, name="Bob's Workspace")
        self.db.add(workspace)
        self.db.commit()

        pref = UserPreference(user_id=user.id, theme="light")
        self.db.add(pref)

        conv = Conversation(workspace_id=workspace.id, user_id=user.id, title="Test Chat")
        self.db.add(conv)
        self.db.commit()

        # Bidirectional relationships verification
        self.assertEqual(workspace.owner, user)
        self.assertIn(workspace, user.workspaces)

        self.assertEqual(conv.workspace, workspace)
        self.assertIn(conv, workspace.conversations)
        self.assertIn(conv, user.conversations)

        self.assertEqual(pref.user, user)
        self.assertEqual(user.preference, pref)

        # Cascade delete check
        self.db.delete(user)
        self.db.commit()

        # Bob's workspaces, conversations, and preferences should be cascade-deleted
        ws_check = self.db.query(Workspace).filter_by(id=workspace.id).first()
        self.assertIsNone(ws_check)

        pref_check = self.db.query(UserPreference).filter_by(id=pref.id).first()
        self.assertIsNone(pref_check)

        conv_check = self.db.query(Conversation).filter_by(id=conv.id).first()
        self.assertIsNone(conv_check)


if __name__ == "__main__":
    unittest.main()
