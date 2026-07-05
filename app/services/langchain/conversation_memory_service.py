# app/services/langchain/conversation_memory_service.py
# ----------------------------------------------------
# Service layer wrapper managing session histories and ConversationMemory allocations.

import logging
from typing import Dict, Optional, List, Any
from app.config import settings
from app.services.langchain.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)


class ConversationMemoryService:
    """
    Service tracking active conversation histories across multiple sessions.
    Provides session context fetching and history updates.
    """
    def __init__(self):
        self.sessions: Dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        """
        Retrieves or allocates a ConversationMemory instance for the given session ID.
        """
        active_session_id = session_id or "default_session"
        
        if active_session_id not in self.sessions:
            max_history = getattr(settings, "MAX_HISTORY_MESSAGES", 5)
            logger.info(f"Allocating new conversation memory context for session: '{active_session_id}' (max_history={max_history})")
            self.sessions[active_session_id] = ConversationMemory(max_history=max_history)
            
        return self.sessions[active_session_id]

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Helper method returning the raw history list of a session.
        """
        memory = self.get_memory(session_id)
        return memory.history

    def add_message(self, session_id: str, user_query: str, rewritten_query: str):
        """
        Appends a query transaction to the session memory.
        """
        # Read enable config
        enable_memory = getattr(settings, "ENABLE_CONVERSATION_MEMORY", True)
        if not enable_memory:
            logger.debug("Conversation memory is disabled in settings. Skipping append.")
            return

        memory = self.get_memory(session_id)
        memory.add_message(user_query, rewritten_query)


# Global singleton instance for memory persistence across retrieval pipeline calls
_shared_memory_service = ConversationMemoryService()


def get_memory_service() -> ConversationMemoryService:
    """
    Retrieves the persistent shared ConversationMemoryService singleton instance.
    """
    return _shared_memory_service
