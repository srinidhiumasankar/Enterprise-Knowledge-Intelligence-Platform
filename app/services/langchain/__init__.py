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
from app.services.langchain.parent_retriever import CustomParentDocumentRetriever, ParentRetriever
from app.services.langchain.self_query import ChromaSelfQueryRetriever, get_self_query_retriever
from app.services.langchain.pipeline import RetrievalPipeline
from app.services.langchain.ensemble import EnsembleRetriever
from app.services.langchain.ensemble_service import EnsembleRetrieverService
from app.services.langchain.adaptive import AdaptiveRetriever
from app.services.langchain.adaptive_service import AdaptiveRetrieverService
from app.services.langchain.query_rewriter import QueryRewriter
from app.services.langchain.query_rewriter_service import QueryRewriterService
from app.services.langchain.conversation_memory import ConversationMemory
from app.services.langchain.conversation_memory_service import ConversationMemoryService
from app.services.langchain.metadata_ranker import MetadataRanker
from app.services.langchain.metadata_ranker_service import MetadataRankerService
from app.services.langchain.result_scorer import ResultScorer
from app.services.langchain.result_scorer_service import ResultScorerService
from app.services.langchain.answer_verifier import AnswerVerifier
from app.services.langchain.answer_verifier_service import AnswerVerifierService
from app.services.langchain.retrieval_analytics import RetrievalAnalytics
from app.services.langchain.retrieval_analytics_service import RetrievalAnalyticsService
from app.services.langchain.health_monitor import HealthMonitor
from app.services.langchain.health_service import HealthService

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
    "CustomParentDocumentRetriever",
    "ParentRetriever",
    "ChromaSelfQueryRetriever",
    "get_self_query_retriever",
    "RetrievalPipeline",
    "EnsembleRetriever",
    "EnsembleRetrieverService",
    "AdaptiveRetriever",
    "AdaptiveRetrieverService",
    "QueryRewriter",
    "QueryRewriterService",
    "ConversationMemory",
    "ConversationMemoryService",
    "MetadataRanker",
    "MetadataRankerService",
    "ResultScorer",
    "ResultScorerService",
    "AnswerVerifier",
    "AnswerVerifierService",
    "RetrievalAnalytics",
    "RetrievalAnalyticsService",
    "HealthMonitor",
    "HealthService",
]
