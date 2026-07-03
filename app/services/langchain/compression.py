# app/services/langchain/compression.py
# -------------------------------------
# Custom LangChain Document Compressor and Contextual Compression Retriever implementation.

import logging
import time
import re
from typing import Any, List, Dict, Optional, Sequence
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
try:
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from app.services.langchain.llm import get_llm
from app.services.langchain.retriever import get_retriever

logger = logging.getLogger(__name__)

COMPRESSION_PROMPT = PromptTemplate(
    input_variables=["query", "documents"],
    template="""You are an AI assistant specializing in document compression and information extraction.
Your task is to compress the provided documents to keep ONLY information directly relevant to answering the query: "{query}".

For each document, analyze the text and output a condensed version.
Instructions:
1. Remove irrelevant sentences, filler words, duplicate sentences, or unrelated paragraphs.
2. Retain all key facts, numbers, dates, and exact answers relevant to the query.
3. If a document is completely irrelevant to the query, output "IRRELEVANT".
4. Format your output strictly using XML-style tags as follows:
<compressed_doc id="[ID]">
[Compressed text or "IRRELEVANT"]
</compressed_doc>

Here are the documents to compress:
{documents}

Strictly output only the <compressed_doc> tags. Do not add any extra explanations or introductory text.
"""
)


class LLMBulkDocumentCompressor(BaseDocumentCompressor):
    """
    Custom LangChain document compressor performing bulk document compression in a single LLM call.
    """
    llm: Any = Field(description="LLM instance to perform compression")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        start_time = time.time()
        logger.info(f"Compressing {len(documents)} retrieved documents...")

        try:
            # Format documents into XML blocks
            doc_blocks = []
            for idx, doc in enumerate(documents):
                doc_blocks.append(f'<doc id="{idx}">\n{doc.page_content}\n</doc>')
            formatted_docs = "\n\n".join(doc_blocks)

            # Call LLM
            prompt = COMPRESSION_PROMPT.format(query=query, documents=formatted_docs)
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse results
            compressed_docs = []
            for idx, doc in enumerate(documents):
                pattern = rf'<compressed_doc\s+id=["\']?{idx}["\']?\s*>(.*?)</compressed_doc>'
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    compressed_text = match.group(1).strip()
                    if compressed_text and compressed_text.upper() != "IRRELEVANT":
                        meta = doc.metadata.copy()
                        compressed_docs.append(Document(page_content=compressed_text, metadata=meta))
                    else:
                        logger.info(f"Document at index {idx} marked as IRRELEVANT and removed.")
                else:
                    logger.warning(f"Failed to parse compressed output for index {idx}. Retaining original.")
                    compressed_docs.append(doc)

            latency = (time.time() - start_time) * 1000
            ratio = len(compressed_docs) / len(documents) if documents else 0.0
            logger.info(
                f"Compression completed: original={len(documents)}, compressed={len(compressed_docs)} "
                f"(ratio={ratio:.2%}) in {latency:.2f}ms"
            )
            return compressed_docs

        except Exception as e:
            logger.error(f"LLM bulk compression failed, falling back to original documents: {e}", exc_info=True)
            return documents


class CompressionRetriever:
    """
    Service wrapping ContextualCompressionRetriever to manage the compression pipeline.
    """
    def __init__(
        self,
        owner_id: Optional[int] = None,
        document_id: Optional[int] = None,
        top_k: int = 5,
        llm: Optional[Any] = None,
        base_retriever: Optional[BaseRetriever] = None,
    ):
        self.owner_id = owner_id
        self.document_id = document_id
        self.top_k = top_k
        self.llm = llm or get_llm()
        
        # Resolve base retriever (defaults to hybrid search via get_retriever)
        self.base_retriever = base_retriever or get_retriever(
            owner_id=owner_id,
            document_id=document_id,
            top_k=top_k
        )
        self.compressor = LLMBulkDocumentCompressor(llm=self.llm)
        self.compression_retriever = None
        self.initialize()

    def initialize(self):
        """
        Initializes the LangChain ContextualCompressionRetriever.
        """
        logger.info("Initializing Contextual Compression Retriever...")
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Directly compress a list of documents for a query.
        """
        return list(self.compressor.compress_documents(documents, query))

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieves and compresses relevant documents for the given query.
        """
        start_time = time.time()
        logger.info(f"Retrieval initiated for query: '{query}'")
        try:
            results = self.compression_retriever.invoke(query)
            latency = (time.time() - start_time) * 1000
            logger.info(f"Retrieval & compression completed. Found {len(results)} compressed documents in {latency:.2f}ms")
            return results
        except Exception as e:
            logger.error(f"CompressionRetriever retrieval failed: {e}. Falling back to base retrieval.", exc_info=True)
            # Fallback to base retriever to prevent crashing the pipeline
            try:
                fallback_results = self.base_retriever.invoke(query)
                return fallback_results
            except Exception as ex:
                logger.error(f"Fallback base retriever also failed: {ex}", exc_info=True)
                return []
