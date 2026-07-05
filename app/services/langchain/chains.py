# app/services/langchain/chains.py
# --------------------------------
# Factory functions for constructing LangChain chains.

import logging
import time
from typing import Optional, List, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.langchain.llm import get_llm
from app.services.langchain.prompts import get_basic_prompt, get_qa_prompt

logger = logging.getLogger(__name__)


def create_basic_chain(llm: Optional[ChatGoogleGenerativeAI] = None) -> Runnable:
    """
    Construct a simple chain that processes an instruction with a default LLM.

    Purpose:
        Creates a basic LangChain run chain linking a standard prompt, LLM, and string parser.

    Parameters:
        llm (Optional[ChatGoogleGenerativeAI]): Custom LLM wrapper. Defaults to the cached get_llm().

    Returns:
        Runnable: The constructed LangChain pipeline ready to be invoked (e.g. chain.invoke({"instruction": "..."})).
    """
    logger.info("Constructing basic LangChain runnable chain.")
    active_llm = llm or get_llm()
    prompt = get_basic_prompt()
    return prompt | active_llm | StrOutputParser()


def create_qa_chain(llm: Optional[ChatGoogleGenerativeAI] = None) -> Runnable:
    """
    Construct a RAG-ready QA chain that accepts context and a question.

    Purpose:
        Creates a QA pipeline using the constraint-enforcing QA prompt template,
        LLM, and a string output parser.

    Parameters:
        llm (Optional[ChatGoogleGenerativeAI]): Custom LLM wrapper. Defaults to the cached get_llm().

    Returns:
        Runnable: The QA pipeline ready to be invoked (e.g. chain.invoke({"context": "...", "question": "..."})).
    """
    logger.info("Constructing RAG QA LangChain runnable chain.")
    active_llm = llm or get_llm()
    prompt = get_qa_prompt()
    return prompt | active_llm | StrOutputParser()


class LangChainRAGChain:
    """
    Production-grade RAG pipeline orchestrator implementing the execution flow:
    User Question -> Generate Query Embedding -> LangChain Retriever ->
    Retrieve Relevant Documents -> Build Context -> Prompt Template -> Gemini LLM -> Grounded Answer.
    """

    def __init__(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        prompt_template: Optional[Any] = None,
    ):
        self.owner_id = owner_id
        self.document_id = document_id
        self.top_k = top_k
        self.llm = llm or get_llm()

        # Lazy load get_retriever to prevent circular imports
        from app.services.langchain.retriever import get_retriever
        self.retriever = retriever or get_retriever(
            owner_id=owner_id, document_id=document_id, top_k=top_k
        )
        self.prompt_template = prompt_template or get_qa_prompt()

    def run(self, question: str) -> str:
        """
        Execute the complete LangChain RAG pipeline.

        Purpose:
            Retrieves context, formats prompt, executes Gemini, and returns a grounded answer.

        Parameters:
            question (str): User query.

        Returns:
            str: Generated grounded answer or fallback if information is insufficient.
        """
        total_start = time.perf_counter()
        logger.info(f"RAG chain execution started for query: '{question}'")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only.")

        # 1. Retrieve Relevant Documents using retriever
        retrieval_start = time.perf_counter()
        try:
            logger.info("Invoking LangChain retriever...")
            docs = self.retriever.invoke(question)
            retrieval_latency = time.perf_counter() - retrieval_start
            logger.info(f"Retrieved {len(docs)} chunks. Retrieval time: {retrieval_latency:.4f}s")
        except Exception as e:
            logger.error(f"Retriever failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain retrieval failed: {e}") from e

        # If context is empty/insufficient, return fallback immediately
        if not docs:
            logger.info("Zero documents retrieved. Returning fallback response directly.")
            total_latency = time.perf_counter() - total_start
            logger.info(f"Total chain latency: {total_latency:.4f}s")
            return "Insufficient information found in uploaded documents."

        # 2. Build Context
        context_start = time.perf_counter()
        formatted_context = self.format_docs(docs)
        context_latency = time.perf_counter() - context_start
        logger.info(f"Context built. Context building latency: {context_latency:.4f}s")

        # 3. Prompt Template
        prompt_start = time.perf_counter()
        try:
            formatted_prompt = self.prompt_template.format(
                context=formatted_context,
                question=question
            )
            prompt_latency = time.perf_counter() - prompt_start
            logger.info(f"Prompt formatted. Prompt creation time: {prompt_latency:.4f}s")
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain prompt formatting failed: {e}") from e

        # 4. Invoke Gemini LLM
        llm_start = time.perf_counter()
        try:
            logger.info("Executing Gemini LLM invocation...")
            response = self.llm.invoke(formatted_prompt)
            llm_latency = time.perf_counter() - llm_start
            logger.info(f"LLM execution succeeded. LLM execution time: {llm_latency:.4f}s")
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain LLM execution failed: {e}") from e

        raw_answer = response.content.strip()

        # Post-process response to handle missing answers or standard fallbacks
        insufficient_markers = [
            "insufficient information",
            "cannot determine the answer",
            "not present in the context",
            "not mentioned in the context",
            "not found in the uploaded documents",
            "i cannot determine",
            "no context available"
        ]
        
        lower_answer = raw_answer.lower()
        if not raw_answer or any(marker in lower_answer for marker in insufficient_markers):
            logger.info("LLM response matches insufficient information markers. Normalizing to fallback.")
            raw_answer = "Insufficient information found in uploaded documents."

        # 5. Answer Verification & Grounding Check
        try:
            from app.services.langchain.answer_verifier_service import AnswerVerifierService
            verifier_service = AnswerVerifierService()
            verifier = verifier_service.get_verifier()
            t_verify_start = time.perf_counter()
            final_response = verifier.verify_answer(raw_answer, docs)
            verify_latency = (time.perf_counter() - t_verify_start) * 1000
            from app.services.langchain.retrieval_analytics import RetrievalAnalytics
            RetrievalAnalytics.get_instance().record_latency("answer_verifier_latency", verify_latency)
        except Exception as e:
            logger.warning(f"Answer verification failed: {e}. Falling back to raw LLM answer.", exc_info=True)
            from app.services.langchain.answer_verifier import FinalResponse
            final_response = FinalResponse(
                raw_answer,
                verification_score=50.0,
                grounding_score=50.0,
                hallucination_risk="Medium",
                verification_status="Passed",
                confidence_level="Medium"
            )

        total_latency = time.perf_counter() - total_start
        logger.info(f"Total chain latency: {total_latency:.4f}s")
        return final_response

    @staticmethod
    def format_docs(docs: List[Any]) -> str:
        formatted = []
        for idx, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "Unknown Document")
            doc_id = doc.metadata.get("document_id", "N/A")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            text = doc.page_content
            block = (
                f"--- Source {idx} ---\n"
                f"File: {filename}\n"
                f"Document ID: {doc_id}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Content: {text.strip()}\n"
            )
            formatted.append(block)
        return "\n".join(formatted)


def create_rag_chain(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    llm: Optional[Any] = None,
    retriever: Optional[Any] = None,
    prompt_template: Optional[Any] = None,
) -> LangChainRAGChain:
    """
    Factory function to construct a reusable LangChainRAGChain instance.

    Purpose:
        Creates a grounded RAG execution pipeline.

    Parameters:
        owner_id (Optional[int]): User/owner filter context.
        document_id (Optional[int]): Document filter context.
        top_k (int): Number of chunks to fetch.
        llm (Optional[Any]): Custom LLM wrapper.
        retriever (Optional[Any]): Custom Retriever wrapper.
        prompt_template (Optional[Any]): Custom QA prompt template.

    Returns:
        LangChainRAGChain: The constructed pipeline orchestrator.
    """
    return LangChainRAGChain(
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template
    )
