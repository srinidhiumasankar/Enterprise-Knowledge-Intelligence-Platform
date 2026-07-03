# app/services/langchain/citations.py
# -----------------------------------
# Upgraded LangChain RAG pipeline orchestrator with precision source attribution.

import logging
import time
from typing import Optional, List, Any, Dict

from app.services.langchain.llm import get_llm
from app.services.langchain.prompts import get_citation_qa_prompt, get_conversation_qa_prompt
from app.services.citation_service import CitationService
from app.services.memory_service import get_memory_service

logger = logging.getLogger(__name__)


class LangChainCitationRAGChain:
    """
    Upgraded RAG pipeline implementing execution flow:
    User Question -> LangChain Retriever -> Format context with source/chunk metadata -> Gemini LLM -> Grounded answer with trailing sources.
    """

    def __init__(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        prompt_template: Optional[Any] = None,
        citation_service: Optional[Any] = None,
    ):
        self.owner_id = owner_id
        self.document_id = document_id
        self.top_k = top_k
        self.llm = llm or get_llm()

        # Lazy imports to prevent circular dependency
        from app.services.langchain.retriever import get_retriever
        self.retriever = retriever or get_retriever(
            owner_id=owner_id, document_id=document_id, top_k=top_k
        )
        self.prompt_template = prompt_template or get_citation_qa_prompt()
        self.citation_service = citation_service or CitationService()

    def run(self, question: str) -> str:
        """
        Execute the citation-attributed RAG pipeline.

        Parameters:
            question (str): User question.

        Returns:
            str: Generated response with citations, or fallback error response.
        """
        total_start = time.perf_counter()
        logger.info(f"Citation RAG chain execution started for query: '{question}'")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only.")

        # 1. Retrieve documents
        retrieval_start = time.perf_counter()
        try:
            logger.info("Invoking LangChain retriever...")
            docs = self.retriever.invoke(question)
            retrieval_latency = time.perf_counter() - retrieval_start
            logger.info(f"Retrieved {len(docs)} chunks. Retrieval time: {retrieval_latency:.4f}s")
        except Exception as e:
            logger.error(f"Retriever failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain retrieval failed: {e}") from e

        # Requirement 6: If no documents are retrieved, return fallback
        if not docs:
            logger.info("Zero documents retrieved. Returning fallback response directly.")
            total_latency = time.perf_counter() - total_start
            logger.info(f"Total chain latency: {total_latency:.4f}s")
            return "I couldn't find relevant information in the uploaded documents."

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

        # Handle fallback check: If response matches missing indicators
        insufficient_markers = [
            "insufficient information",
            "cannot determine",
            "not present in the context",
            "not mentioned in the context",
            "not found in the uploaded documents",
            "i couldn't find relevant information",
            "no context available"
        ]
        
        lower_answer = raw_answer.lower()
        if not raw_answer or any(marker in lower_answer for marker in insufficient_markers):
            logger.info("LLM response matches insufficient information markers. Normalizing to fallback.")
            raw_answer = "I couldn't find relevant information in the uploaded documents."

        total_latency = time.perf_counter() - total_start
        logger.info(f"Total chain latency: {total_latency:.4f}s")
        return raw_answer

    @staticmethod
    def format_docs(docs: List[Any]) -> str:
        """
        Format context according to context attribution layout schema:
        Source: employee_handbook.pdf
        Chunk: 12

        <chunk text>
        """
        formatted = []
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown Document")
            chunk_index = doc.metadata.get("chunk_index", 0)
            text = doc.page_content
            block = (
                f"Source: {filename}\n"
                f"Chunk: {chunk_index}\n\n"
                f"{text.strip()}"
            )
            formatted.append(block)
        
        return "\n\n".join(formatted)


class LangChainConversationalRAGChain(LangChainCitationRAGChain):
    """
    Upgraded RAG pipeline that incorporates session-isolated multi-turn conversational memory.
    """

    def __init__(
        self,
        session_id: str,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        prompt_template: Optional[Any] = None,
        citation_service: Optional[Any] = None,
        memory_service: Optional[Any] = None,
    ):
        super().__init__(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k,
            llm=llm,
            retriever=retriever,
            prompt_template=prompt_template or get_conversation_qa_prompt(),
            citation_service=citation_service,
        )
        self.session_id = session_id
        self.memory_service = memory_service or get_memory_service()

    def run(self, question: str) -> str:
        """
        Execute the conversational RAG pipeline.

        Parameters:
            question (str): User question.

        Returns:
            str: Generated response with memory history and citations, or fallback.
        """
        total_start = time.perf_counter()
        logger.info(f"Conversational RAG chain execution started for session: '{self.session_id}', query: '{question}'")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only.")

        # 1. Retrieve session memory
        try:
            memory = self.memory_service.get_memory(self.session_id)
            memory_vars = memory.load_memory_variables({})
            chat_history = memory_vars.get("chat_history", "")
            logger.info(f"Retrieved memory history: '{chat_history}'")
        except Exception as e:
            logger.error(f"Failed to load memory history: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load conversational memory: {e}") from e

        # 2. Retrieve documents
        retrieval_start = time.perf_counter()
        try:
            logger.info("Invoking LangChain retriever...")
            docs = self.retriever.invoke(question)
            retrieval_latency = time.perf_counter() - retrieval_start
            logger.info(f"Retrieved {len(docs)} chunks. Retrieval time: {retrieval_latency:.4f}s")
        except Exception as e:
            logger.error(f"Retriever failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain retrieval failed: {e}") from e

        # If no documents are retrieved, return fallback
        if not docs:
            logger.info("Zero documents retrieved. Returning fallback response directly.")
            total_latency = time.perf_counter() - total_start
            logger.info(f"Total chain latency: {total_latency:.4f}s")
            return "I couldn't find relevant information in the uploaded documents."

        # 3. Build Context
        context_start = time.perf_counter()
        formatted_context = self.format_docs(docs)
        context_latency = time.perf_counter() - context_start
        logger.info(f"Context built. Context building latency: {context_latency:.4f}s")

        # 4. Prompt Template
        prompt_start = time.perf_counter()
        try:
            formatted_prompt = self.prompt_template.format(
                chat_history=chat_history,
                context=formatted_context,
                question=question
            )
            prompt_latency = time.perf_counter() - prompt_start
            logger.info(f"Prompt formatted. Prompt creation time: {prompt_latency:.4f}s")
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}", exc_info=True)
            raise RuntimeError(f"RAG chain prompt formatting failed: {e}") from e

        # 5. Invoke Gemini LLM
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

        # Handle fallback check: If response matches missing indicators
        insufficient_markers = [
            "insufficient information",
            "cannot determine",
            "not present in the context",
            "not mentioned in the context",
            "not found in the uploaded documents",
            "i couldn't find relevant information",
            "no context available"
        ]
        
        lower_answer = raw_answer.lower()
        if not raw_answer or any(marker in lower_answer for marker in insufficient_markers):
            logger.info("LLM response matches insufficient information markers. Normalizing to fallback.")
            raw_answer = "I couldn't find relevant information in the uploaded documents."
        else:
            # 6. Save current turn back to memory only if it was a valid, successful grounded answer!
            try:
                memory.save_context(
                    inputs={"question": question},
                    outputs={"answer": raw_answer}
                )
                logger.info("Successfully updated memory buffer with new conversation turn.")
            except Exception as e:
                logger.error(f"Failed to save conversation turn to memory: {e}", exc_info=True)

        total_latency = time.perf_counter() - total_start
        logger.info(f"Total chain latency: {total_latency:.4f}s")
        return raw_answer


def create_citation_rag_chain(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    llm: Optional[Any] = None,
    retriever: Optional[Any] = None,
    prompt_template: Optional[Any] = None,
    citation_service: Optional[Any] = None,
) -> LangChainCitationRAGChain:
    """
    Factory function to construct a LangChainCitationRAGChain instance.
    """
    return LangChainCitationRAGChain(
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template,
        citation_service=citation_service,
    )


def create_conversational_rag_chain(
    session_id: str,
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    llm: Optional[Any] = None,
    retriever: Optional[Any] = None,
    prompt_template: Optional[Any] = None,
    citation_service: Optional[Any] = None,
    memory_service: Optional[Any] = None,
) -> LangChainConversationalRAGChain:
    """
    Factory function to construct a LangChainConversationalRAGChain instance.

    Parameters:
        session_id (str): Chat session identifier.
        owner_id (Optional[int]): User/owner filter.
        document_id (Optional[int]): Document filter.
        top_k (int): Number of chunks to fetch.
        llm (Optional[Any]): Custom LLM wrapper.
        retriever (Optional[Any]): Custom Retriever wrapper.
        prompt_template (Optional[Any]): Custom QA prompt template.
        citation_service (Optional[Any]): Custom CitationService wrapper.
        memory_service (Optional[Any]): Custom MemoryService wrapper.

    Returns:
        LangChainConversationalRAGChain: The initialized conversational pipeline.
    """
    return LangChainConversationalRAGChain(
        session_id=session_id,
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        llm=llm,
        retriever=retriever,
        prompt_template=prompt_template,
        citation_service=citation_service,
        memory_service=memory_service,
    )
