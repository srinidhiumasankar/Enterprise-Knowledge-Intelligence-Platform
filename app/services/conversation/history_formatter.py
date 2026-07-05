# app/services/conversation/history_formatter.py
# ---------------------------------------------
# Formatter converting raw chat message models into structured chronological text blocks.

import logging
from typing import List
from app.models.chat_message import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class HistoryFormatter:
    """
    Utility formatting database chat messages chronologically for prompt injection.
    """
    @staticmethod
    def format_message(role: str, content: str) -> str:
        """
        Standardizes format label prefixes for User vs Assistant content.
        """
        prefix = "User" if role.lower() in ("user", "message_role.user") else "Assistant"
        return f"{prefix}: {content}"

    @staticmethod
    def format_history(messages: List[ChatMessage], limit: int = 20) -> str:
        """
        Sanitizes empty records, orders chronologically, and trims by length limits.
        """
        # 1. Filter out empty message values and system instruction blocks
        filtered = []
        for msg in messages:
            if not msg.content or not msg.content.strip():
                continue
            # Keep USER and ASSISTANT messages, ignore SYSTEM role for chat text flow
            if msg.role not in (MessageRole.USER, MessageRole.ASSISTANT):
                continue
            filtered.append(msg)

        # 2. Sort chronologically (ascending)
        filtered.sort(key=lambda m: (m.created_at, m.id))

        # 3. Trim length based on limit (take last N messages)
        if limit > 0 and len(filtered) > limit:
            trimmed = filtered[-limit:]
            logger.info(f"HistoryFormatter trimmed messages length from {len(filtered)} to last {limit}")
        else:
            trimmed = filtered

        # 4. Formulate output text
        lines = []
        for msg in trimmed:
            lines.append(HistoryFormatter.format_message(msg.role.name, msg.content))

        return "\n".join(lines)
