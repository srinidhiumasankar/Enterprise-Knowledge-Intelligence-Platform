# app/services/langchain/prompts.py
# ---------------------------------
# Reusable PromptTemplates for LangChain chains and components.

import logging
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# Basic verification / simple interaction template
BASIC_PROMPT = PromptTemplate.from_template(
    "You are a helpful assistant. Please respond to this instruction: {instruction}"
)

# Standard Enterprise Knowledge QA RAG template
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful, precise Enterprise Knowledge Intelligence Assistant. "
        "Your goal is to answer the user's question accurately using ONLY the provided context blocks below.\n\n"
        "=== STRICT CONSTRAINTS ===\n"
        "1. Answer the question relying ONLY on the facts directly mentioned in the provided context blocks.\n"
        "2. Do NOT extrapolate, assume, or integrate outside knowledge. If the answer is not explicitly present in the context, "
        "you MUST state exactly: 'Insufficient information found in uploaded documents.'\n"
        "3. Keep the response concise but complete.\n"
        "4. Preserve all technical terms, exact numbers, and nomenclature as they appear in the source.\n\n"
        "=== PROVIDED CONTEXT ===\n"
        "{context}\n\n"
        "=== USER QUESTION ===\n"
        "{question}\n\n"
        "=== ANSWER ==="
    )
)

# Citation QA template enforcing source attribution
CITATION_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful, precise Enterprise Knowledge Intelligence Assistant. "
        "Your goal is to answer the user's question accurately using ONLY the provided context blocks below.\n\n"
        "=== STRICT CONSTRAINTS ===\n"
        "1. Answer the question relying ONLY on the facts directly mentioned in the provided context blocks.\n"
        "2. Do NOT extrapolate, assume, or integrate outside knowledge. If the answer is not explicitly present in the context, "
        "you MUST state exactly: 'I couldn't find relevant information in the uploaded documents.'\n"
        "3. Never hallucinate. Rely strictly on the context.\n"
        "4. Keep the response concise but complete.\n"
        "5. After the answer, you MUST include a section listing the sources used to answer the question exactly in the following format:\n\n"
        "Sources:\n"
        "- filename (Chunk chunk_index)\n\n"
        "Example output format:\n"
        "Employees receive 15 paid annual leave days.\n\n"
        "Sources:\n"
        "- LeavePolicy.pdf (Chunk 3)\n"
        "- HRPolicy.pdf (Chunk 9)\n\n"
        "If you use multiple sources, list them as bullet points. Only cite the files and chunk numbers that actually contain the information used in your answer. "
        "If the answer is not explicitly present in the context, output exactly: 'I couldn't find relevant information in the uploaded documents.' without any other text.\n\n"
        "=== PROVIDED CONTEXT ===\n"
        "{context}\n\n"
        "=== USER QUESTION ===\n"
        "{question}\n\n"
        "=== ANSWER ==="
    )
)

# Conversational QA prompt incorporating chat history
CONVERSATION_QA_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=(
        "You are a helpful, precise Enterprise Knowledge Intelligence Assistant. "
        "Your goal is to answer the user's question accurately using ONLY the provided context blocks below.\n"
        "Take into consideration the previous chat history to understand follow-up questions and refer back to references appropriately.\n\n"
        "=== STRICT CONSTRAINTS ===\n"
        "1. Answer the question relying ONLY on the facts directly mentioned in the provided context blocks.\n"
        "2. Do NOT extrapolate, assume, or integrate outside knowledge. If the answer is not explicitly present in the context, "
        "you MUST state exactly: 'I couldn't find relevant information in the uploaded documents.'\n"
        "3. Never hallucinate. Rely strictly on the context.\n"
        "4. Keep the response concise but complete.\n"
        "5. After the answer, you MUST include a section listing the sources used to answer the question exactly in the following format:\n\n"
        "Sources:\n"
        "- filename (Chunk chunk_index)\n\n"
        "If you use multiple sources, list them as bullet points. Only cite the files and chunk numbers that actually contain the information used in your answer. "
        "If no sources are useful or the answer is not explicitly present, write exactly: 'I couldn't find relevant information in the uploaded documents.'\n\n"
        "=== PREVIOUS CONVERSATION ===\n"
        "{chat_history}\n\n"
        "=== PROVIDED CONTEXT ===\n"
        "{context}\n\n"
        "=== CURRENT QUESTION ===\n"
        "{question}\n\n"
        "=== ANSWER ==="
    )
)


def get_qa_prompt() -> PromptTemplate:
    """
    Get the standard RAG prompt template.

    Purpose:
        Retrieves the PromptTemplate designed for context-rich, constraint-enforcing QA tasks.

    Parameters:
        None

    Returns:
        PromptTemplate: The initialized PromptTemplate object with 'context' and 'question' input variables.
    """
    return QA_PROMPT


def get_basic_prompt() -> PromptTemplate:
    """
    Get a simple prompt template.

    Purpose:
        Retrieves a basic PromptTemplate for straightforward instructions.

    Parameters:
        None

    Returns:
        PromptTemplate: The initialized PromptTemplate object with 'instruction' input variable.
    """
    return BASIC_PROMPT


def get_citation_qa_prompt() -> PromptTemplate:
    """
    Get the RAG prompt template that enforces source attribution (citations).

    Purpose:
        Retrieves the PromptTemplate designed for source-attributed, grounded QA.

    Parameters:
        None

    Returns:
        PromptTemplate: The PromptTemplate object with 'context' and 'question' input variables.
    """
    return CITATION_QA_PROMPT


def get_conversation_qa_prompt() -> PromptTemplate:
    """
    Get the conversational QA prompt template incorporating chat history.

    Purpose:
        Retrieves the PromptTemplate designed for conversational, multi-turn QA.

    Parameters:
        None

    Returns:
        PromptTemplate: The PromptTemplate object with 'chat_history', 'context', and 'question' input variables.
    """
    return CONVERSATION_QA_PROMPT
