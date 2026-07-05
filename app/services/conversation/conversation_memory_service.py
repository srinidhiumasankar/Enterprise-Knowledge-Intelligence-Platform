# app/services/conversation/conversation_memory_service.py
# ------------------------------------------------------
# Conversation memory service managing conversational contexts, limits, and trimming.

import logging
from typing import Tuple, Optional
from sqlalchemy.orm import Session

from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation.history_formatter import HistoryFormatter
from app.config import settings

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """
    Service layer providing token-constrained conversation history retrieval and trimming.
    """
    def __init__(self, db: Session):
        self.db = db
        self.repo = ConversationRepository(db)

    def load_conversation_history(
        self,
        conversation_id: int,
        user_id: int,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Loads, validates ownership, trims to budget (message count/token), and formats chat history.
        """
        if max_messages is None:
            max_messages = settings.MAX_CONVERSATION_MESSAGES
        if max_tokens is None:
            max_tokens = settings.MAX_CONVERSATION_TOKENS

        # 1. Retrieve conversation record (validates presence)
        conv = self.repo.get_by_id(conversation_id, include_deleted=False)
        if not conv:
            logger.warning(f"Memory load failed: Conversation {conversation_id} not found.")
            raise KeyError("Conversation not found")
        if conv.user_id != user_id:
            logger.warning(f"Memory load failed: User {user_id} unauthorized for conversation {conversation_id}")
            raise PermissionError("Unauthorized access to conversation")

        messages = conv.messages
        logger.info(f"Conversation {conversation_id} memory loaded: {len(messages)} messages fetched.")

        # 2. Filter active user and assistant messages
        from app.models.chat_message import MessageRole
        valid_messages = [
            m for m in messages
            if m.content and m.content.strip() and m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        
        # Sort chronologically
        valid_messages.sort(key=lambda m: (m.created_at, m.id))

        # 3. Trim by max message count constraint (keep last N)
        if max_messages > 0 and len(valid_messages) > max_messages:
            valid_messages = valid_messages[-max_messages:]
            logger.info(f"History trimmed: message count capped at {max_messages}.")

        # 4. Limit by token budget (from most recent to oldest)
        formatted_lines = []
        accumulated_tokens = 0

        for msg in reversed(valid_messages):
            line = HistoryFormatter.format_message(msg.role.name, msg.content)
            # Reuse cached db token counts, or estimate characters fallback
            msg_tokens = msg.token_count if msg.token_count is not None else self.estimate_tokens(line)

            if max_tokens > 0 and (accumulated_tokens + msg_tokens) > max_tokens:
                logger.info(f"History trimmed: token budget limit of {max_tokens} reached.")
                break

            formatted_lines.append(line)
            accumulated_tokens += msg_tokens

        # Reverse back to keep chronological order
        formatted_lines.reverse()
        formatted_history_text = "\n".join(formatted_lines)

        logger.info(f"Context built: {len(formatted_lines)} messages, {accumulated_tokens} tokens estimated.")
        return formatted_history_text, accumulated_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimates token count using standard English character heuristic (4 chars per token).
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
