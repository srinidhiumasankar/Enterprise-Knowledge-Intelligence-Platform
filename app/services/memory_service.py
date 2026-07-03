# app/services/memory_service.py
# ------------------------------
# Core Memory Service for session-isolated conversation memories.

import logging
from typing import Dict, Any

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

from app.services.langchain.memory import create_conversation_buffer_memory

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Session-isolated conversation memory service managing chat history per session_id.
    """

    def __init__(self):
        logger.info("Initializing Memory Service...")
        self._sessions: Dict[str, ConversationBufferMemory] = {}

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        Retrieve the ConversationBufferMemory instance for a session.
        Creates one if it does not exist.

        Parameters:
            session_id (str): The unique session ID identifier.

        Returns:
            ConversationBufferMemory: The session's memory instance.
        """
        if not session_id:
            raise ValueError("session_id cannot be empty.")
            
        if session_id not in self._sessions:
            logger.info(f"Creating new conversation memory for session_id: {session_id}")
            self._sessions[session_id] = create_conversation_buffer_memory()
            
        return self._sessions[session_id]

    def clear_memory(self, session_id: str) -> None:
        """
        Clear conversation memory for a session.

        Parameters:
            session_id (str): The unique session ID identifier.
        """
        if session_id in self._sessions:
            logger.info(f"Clearing memory for session_id: {session_id}")
            self._sessions[session_id].clear()
        else:
            logger.info(f"No active memory session found to clear for session_id: {session_id}")

    def create_memory(self, session_id: str) -> ConversationBufferMemory:
        """
        Explicitly create a new conversation memory instance for a session.
        If one already exists, it is overwritten (cleared).

        Parameters:
            session_id (str): The unique session ID identifier.

        Returns:
            ConversationBufferMemory: The session's memory instance.
        """
        if not session_id:
            raise ValueError("session_id cannot be empty.")
            
        logger.info(f"Explicitly creating memory for session_id: {session_id}")
        self._sessions[session_id] = create_conversation_buffer_memory()
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session memory buffer from the active sessions map.

        Parameters:
            session_id (str): The unique session ID identifier.
        """
        if session_id in self._sessions:
            logger.info(f"Deleting session_id: {session_id} from MemoryService.")
            del self._sessions[session_id]
        else:
            logger.info(f"No session found to delete for session_id: {session_id}")

    def delete_expired_sessions(self) -> None:
        """
        Clears all active sessions. Can be integrated with TTL or cron utilities.
        """
        logger.info("Cleaning up all conversational memory buffers...")
        self._sessions.clear()


# Global singleton instance of MemoryService
_memory_service_instance = MemoryService()


def get_memory_service() -> MemoryService:
    """
    FastAPI dependency injection provider returning the singleton MemoryService instance.

    Returns:
        MemoryService: The MemoryService instance.
    """
    return _memory_service_instance
