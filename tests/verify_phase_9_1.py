# tests/verify_phase_9_1.py
# --------------------------
# Standalone verification script for Phase 9.1 (Enterprise Database Architecture).
# Verifies that migrations ran, tables exist, and relationships traverse correctly.

import os
import sys
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.database import Base
from app.models import (
    User, Document, Workspace, Conversation,
    ChatMessage, Collection, SearchHistory, UserPreference
)
from app.models.chat_message import MessageRole

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_1")


def verify_phase_9_1():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.1 DATABASE ARCHITECTURE VERIFICATION")
    logger.info("==================================================")

    # 1. Connect to actual SQLite database file configured in Settings
    logger.info(f"Connecting to database: {settings.DATABASE_URL}")
    engine = create_engine(settings.DATABASE_URL)
    inspector = inspect(engine)

    # 2. Check table availability
    logger.info("\n--- Verifying Tables Presence ---")
    existing_tables = inspector.get_table_names()
    required_tables = [
        "users", "documents", "document_chunks", "workspaces",
        "conversations", "chat_messages", "collections",
        "document_collections", "search_histories", "user_preferences"
    ]
    for table in required_tables:
        if table not in existing_tables:
            raise AssertionError(f"Table '{table}' is missing from the database!")
        logger.info(f"✓ Table '{table}' verified.")

    # 3. Create Session to perform relationship traversals
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        logger.info("\n--- Verifying Database Operations & Relationships ---")
        
        # 3.1 Create User
        user = User(
            full_name="Verification Agent",
            email="verify91@agent.com",
            hashed_password="verification_secure_password"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"✓ User created (id={user.id}, uuid={user.uuid})")

        # 3.2 Create Workspace
        workspace = Workspace(
            owner_id=user.id,
            name="Verify Workspace",
            description="Diagnostic test environment"
        )
        db.add(workspace)
        db.commit()
        db.refresh(workspace)
        logger.info(f"✓ Workspace created (id={workspace.id}, uuid={workspace.uuid})")

        # 3.3 Create UserPreference
        pref = UserPreference(
            user_id=user.id,
            theme="dark",
            default_workspace=workspace.id,
            preferred_model="gemini-2.5-flash",
            temperature=0.3,
            top_k=5
        )
        db.add(pref)
        db.commit()
        db.refresh(pref)
        logger.info(f"✓ UserPreference created (id={pref.id}, theme={pref.theme})")

        # 3.4 Create Conversation
        conv = Conversation(
            workspace_id=workspace.id,
            user_id=user.id,
            title="Diagnostics Conversation"
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        logger.info(f"✓ Conversation created (id={conv.id}, uuid={conv.uuid})")

        # 3.5 Create ChatMessage
        msg = ChatMessage(
            conversation_id=conv.id,
            role=MessageRole.SYSTEM,
            content="Initializing verification test protocol..."
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        logger.info(f"✓ ChatMessage created (id={msg.id}, role={msg.role.value})")

        # 3.6 Create Collection
        col = Collection(
            workspace_id=workspace.id,
            owner_id=user.id,
            name="Diagnostic Docs"
        )
        db.add(col)
        db.commit()
        db.refresh(col)
        logger.info(f"✓ Collection created (id={col.id}, name={col.name})")

        # 3.7 Create Document and map to Collection (DocumentCollection)
        doc = Document(
            filename="diag.pdf",
            stored_filename="stored_diag.pdf",
            file_extension="pdf",
            mime_type="application/pdf",
            file_size=512,
            file_path="/path/diag.pdf",
            owner_id=user.id,
            workspace_id=workspace.id
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        logger.info(f"✓ Document created (id={doc.id}, filename={doc.filename})")

        # Many-to-many bridge mapping validation
        col.documents.append(doc)
        db.commit()
        logger.info("✓ DocumentCollection relationship mapped successfully.")

        # 3.8 Create SearchHistory
        sh = SearchHistory(
            user_id=user.id,
            workspace_id=workspace.id,
            query="test verification filter",
            result_count=10
        )
        db.add(sh)
        db.commit()
        db.refresh(sh)
        logger.info(f"✓ SearchHistory created (id={sh.id}, query='{sh.query}')")

        # 3.9 Bidirectional checks
        logger.info("\n--- Traversing Bidirectional Relations ---")
        assert user.preference == pref, "User -> UserPreference relationship failed!"
        assert workspace in user.workspaces, "User -> Workspaces list failed!"
        assert conv in workspace.conversations, "Workspace -> Conversations list failed!"
        assert msg in conv.messages, "Conversation -> ChatMessages list failed!"
        assert doc in col.documents, "Collection -> Documents association failed!"
        assert col in doc.collections, "Document -> Collections association failed!"
        assert doc in workspace.documents, "Workspace -> Documents association failed!"
        assert sh in workspace.search_history, "Workspace -> SearchHistory association failed!"
        logger.info("✓ All bidirectional ORM relation traversals succeeded.")

        # 3.10 Clean up test records
        logger.info("\n--- Cleaning Up Verification Records ---")
        db.delete(user) # Cascade deletes workspaces, conversations, preferences, search history, collections
        db.commit()
        
        # Verify cascades
        assert db.query(Workspace).filter_by(id=workspace.id).first() is None
        assert db.query(UserPreference).filter_by(id=pref.id).first() is None
        assert db.query(Conversation).filter_by(id=conv.id).first() is None
        assert db.query(ChatMessage).filter_by(id=msg.id).first() is None
        assert db.query(Collection).filter_by(id=col.id).first() is None
        assert db.query(SearchHistory).filter_by(id=sh.id).first() is None
        logger.info("✓ Cascade deletes and cleanup verify successful.")

    except Exception as e:
        db.rollback()
        logger.error(f"Verification encountered an exception: {e}", exc_info=True)
        raise e
    finally:
        db.close()

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.1 DATABASE ARCHITECTURE VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.1 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.1 DATABASE ARCHITECTURE VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.1 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_1()
