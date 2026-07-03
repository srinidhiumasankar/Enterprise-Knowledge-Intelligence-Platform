# app/services/langchain/memory.py
# --------------------------------
# LangChain Conversation Memory wrapper and helper functions.

import logging

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

logger = logging.getLogger(__name__)


def create_conversation_buffer_memory() -> ConversationBufferMemory:
    """
    Constructs a pre-configured LangChain ConversationBufferMemory instance.

    Purpose:
        Creates a session memory buffer with defined memory keys to track context across turns.

    Returns:
        ConversationBufferMemory: The initialized ConversationBufferMemory object.
    """
    logger.info("Initializing new LangChain ConversationBufferMemory wrapper...")
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer"
    )
