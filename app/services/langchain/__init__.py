# app/services/langchain/__init__.py
# ----------------------------------
# Marks the `langchain` directory as a Python package.
# Exposes the main dependency functions and builders.

from app.services.langchain.llm import get_llm
from app.services.langchain.embeddings import get_embeddings
from app.services.langchain.prompts import get_qa_prompt, get_basic_prompt
from app.services.langchain.chains import create_basic_chain, create_qa_chain
from app.services.langchain.retriever import get_retriever

__all__ = [
    "get_llm",
    "get_embeddings",
    "get_qa_prompt",
    "get_basic_prompt",
    "create_basic_chain",
    "create_qa_chain",
    "get_retriever",
]
