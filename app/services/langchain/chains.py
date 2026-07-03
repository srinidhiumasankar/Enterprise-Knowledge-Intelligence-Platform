# app/services/langchain/chains.py
# --------------------------------
# Factory functions for constructing LangChain chains.

import logging
from typing import Optional
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
