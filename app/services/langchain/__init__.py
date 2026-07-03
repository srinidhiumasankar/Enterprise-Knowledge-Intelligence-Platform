# app/services/langchain/__init__.py
# ----------------------------------
# Marks the `langchain` directory as a Python package.
# Exposes the main dependency functions and builders.

from app.services.langchain.llm import get_llm
from app.services.langchain.embeddings import get_embeddings
from app.services.langchain.prompts import get_qa_prompt, get_basic_prompt, get_citation_qa_prompt, get_conversation_qa_prompt
from app.services.langchain.chains import create_basic_chain, create_qa_chain, create_rag_chain
from app.services.langchain.citations import create_citation_rag_chain, create_conversational_rag_chain
from app.services.langchain.retriever import get_retriever
from app.services.langchain.hybrid_retriever import get_hybrid_retriever
from app.services.langchain.multi_query import get_multi_query_retriever
from app.services.langchain.compression import CompressionRetriever, LLMBulkDocumentCompressor

__all__ = [
    "get_llm",
    "get_embeddings",
    "get_qa_prompt",
    "get_basic_prompt",
    "get_citation_qa_prompt",
    "get_conversation_qa_prompt",
    "create_basic_chain",
    "create_qa_chain",
    "create_rag_chain",
    "create_citation_rag_chain",
    "create_conversational_rag_chain",
    "get_retriever",
    "get_hybrid_retriever",
    "get_multi_query_retriever",
    "CompressionRetriever",
    "LLMBulkDocumentCompressor",
]
