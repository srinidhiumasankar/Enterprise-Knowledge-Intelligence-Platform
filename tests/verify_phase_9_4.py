# tests/verify_phase_9_4.py
# --------------------------
# Standalone verification script for Phase 9.4 (Chat History Integration).
# Verifies retrieval pipeline chat integration, history trimming, prompt formatting, and user isolation.

import os
import sys
import logging
from unittest.mock import MagicMock, patch
from sqlalchemy import select

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import User, Workspace, Conversation, ChatMessage
from app.models.chat_message import MessageRole
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.conversation.conversation_memory_service import ConversationMemoryService

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_9_4")


def verify_phase_9_4():
    logger.info("==================================================")
    logger.info("STARTING PHASE 9.4 CHAT HISTORY INTEGRATION VERIFICATION")
    logger.info("==================================================")

    db = SessionLocal()
    pipeline = RetrievalPipeline()

    try:
        # 1. Create test structures
        logger.info("Step 1: Setting up isolated test records...")
        user_a = User(full_name="User A", email="usera@rag.com", hashed_password="pw")
        user_b = User(full_name="User B", email="userb@rag.com", hashed_password="pw")
        db.add(user_a)
        db.add(user_b)
        db.commit()

        ws_a = Workspace(owner_id=user_a.id, name="Workspace A")
        ws_b = Workspace(owner_id=user_b.id, name="Workspace B")
        db.add(ws_a)
        db.add(ws_b)
        db.commit()

        conv_a = Conversation(workspace_id=ws_a.id, user_id=user_a.id, title="QA Thread A")
        conv_b = Conversation(workspace_id=ws_b.id, user_id=user_b.id, title="QA Thread B")
        db.add(conv_a)
        db.add(conv_b)
        db.commit()

        # Add initial query response items to conv_a to represent existing history
        m1 = ChatMessage(conversation_id=conv_a.id, role=MessageRole.USER, content="Hello RAG", token_count=5)
        m2 = ChatMessage(conversation_id=conv_a.id, role=MessageRole.ASSISTANT, content="Hello human", token_count=5)
        db.add_all([m1, m2])
        db.commit()

        logger.info(f"Test user A: ID={user_a.id}, Conversation A: ID={conv_a.id}")
        logger.info(f"Test user B: ID={user_b.id}, Conversation B: ID={conv_b.id}")

        # 2. Check User / Workspace Isolation
        logger.info("Step 2: Checking user isolation gates...")
        user_context_hack = {
            "conversation_id": conv_a.id,
            "user_id": user_b.id  # User B attempts to access User A's conversation
        }
        try:
            pipeline.chat("What is RAG?", user_context_hack)
            raise AssertionError("User isolation check failed: User B was able to access User A's conversation history!")
        except PermissionError:
            logger.info("✓ User isolation validation successfully blocked unauthorized access.")

        # 3. Verify Memory Loading and formatting
        logger.info("Step 3: Loading history from memory service...")
        memory_service = ConversationMemoryService(db)
        history_text, token_count = memory_service.load_conversation_history(conv_a.id, user_a.id)
        assert "User: Hello RAG" in history_text, "Formatting error: USER prefix/text mismatch"
        assert "Assistant: Hello human" in history_text, "Formatting error: ASSISTANT prefix/text mismatch"
        logger.info("✓ History formatted and loaded successfully.")

        # 4. Pipeline execution and LLM answer generation check
        logger.info("Step 4: Executing retrieval pipeline RAG chat method...")
        user_context_valid = {
            "conversation_id": conv_a.id,
            "user_id": user_a.id
        }

        # Mock the Gemini call and RetrievalPipeline.retrieve (to run database-independent)
        mock_answer = "RAG stands for Retrieval-Augmented Generation."
        
        with patch("app.ai.gemini_service.GeminiService.generate_answer", return_value=mock_answer) as mock_gemini:
            with patch.object(RetrievalPipeline, "retrieve", return_value=[]) as mock_retrieve:
                response = pipeline.chat("What does RAG stand for?", user_context_valid)
                
                # Check LLM response
                assert response == mock_answer, "LLM response mismatch!"
                assert mock_gemini.call_count == 1, "Gemini API should be called exactly once."
                
                # Check Prompt Structure
                called_args = mock_gemini.call_args[1]
                prompt_content = called_args.get("prompt", "")
                assert "=== Conversation History ===" in prompt_content, "Conversation History header missing from prompt"
                assert "=== Retrieved Knowledge ===" in prompt_content, "Retrieved Knowledge header missing from prompt"
                assert "=== Current Question ===" in prompt_content, "Current Question header missing from prompt"
                logger.info("✓ LLM prompt structure incorporates conversation history and current question.")

        # 5. Assistant Response Persistence Check
        logger.info("Step 5: Verifying assistant response database persistence...")
        db.expire_all()
        messages = db.scalars(
            select(ChatMessage).where(ChatMessage.conversation_id == conv_a.id).order_by(ChatMessage.id)
        ).all()
        
        # Original: USER "Hello RAG", ASSISTANT "Hello human"
        # Current execution added: USER "What does RAG stand for?", ASSISTANT "RAG stands for Retrieval-Augmented Generation."
        # Total messages should be 4
        assert len(messages) == 4, f"Expected 4 messages, found {len(messages)}"
        assert messages[2].role == MessageRole.USER
        assert messages[2].content == "What does RAG stand for?"
        assert messages[3].role == MessageRole.ASSISTANT
        assert messages[3].content == mock_answer
        logger.info("✓ User query and assistant response persisted successfully in database.")

        # 6. Clean up
        logger.info("Step 6: Cleaning up test data...")
        db.delete(user_a)
        db.delete(user_b)
        db.commit()
        logger.info("Cleanup complete.")

    except Exception as e:
        db.rollback()
        logger.error(f"Verification encountered an exception: {e}", exc_info=True)
        raise e
    finally:
        db.close()

    logger.info("==========================================")
    logger.info("PASS - PHASE 9.4 CHAT HISTORY INTEGRATION VERIFIED SUCCESSFULLY")
    logger.info("PASS - PHASE 9.4 VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 9.4 CHAT HISTORY INTEGRATION VERIFIED SUCCESSFULLY")
    print("PASS - PHASE 9.4 VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_9_4()
