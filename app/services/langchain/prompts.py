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
        "you MUST state exactly: 'I cannot determine the answer from the uploaded documents.'\n"
        "3. Keep the response concise but complete.\n"
        "4. Preserve all technical terms, exact numbers, and nomenclature as they appear in the source.\n\n"
        "=== PROVIDED CONTEXT ===\n"
        "{context}\n\n"
        "=== USER QUESTION ===\n"
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
