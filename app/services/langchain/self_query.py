# app/services/langchain/self_query.py
# ------------------------------------
# Self Query Retriever service using LangChain structured query constructor to generate metadata filters.

import logging
import time
import re
from typing import Any, List, Dict, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.structured_query import Comparison, Operation, StructuredQuery
from pydantic import Field

from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.chains.query_constructor.base import load_query_constructor_chain

from app.services.langchain.llm import get_llm

logger = logging.getLogger(__name__)

# Schema metadata fields definitions
metadata_field_info = [
    AttributeInfo(name="owner_id", description="The ID of the owner/user who uploaded the document", type="integer"),
    AttributeInfo(name="document_id", description="The unique ID of the document", type="integer"),
    AttributeInfo(name="filename", description="The filename of the document", type="string"),
    AttributeInfo(name="department", description="The department the document belongs to (e.g. HR, Finance, Engineering)", type="string"),
    AttributeInfo(name="category", description="The category of the document (e.g. policy, report, manual)", type="string"),
    AttributeInfo(name="page_number", description="The page number of the chunk within the original document", type="integer"),
    AttributeInfo(name="document_type", description="The type of the document (e.g. PDF, Word, Text)", type="string"),
    AttributeInfo(name="year", description="The publication or update year of the document (e.g. 2022, 2023, 2024)", type="integer"),
    AttributeInfo(name="author", description="The author who created the document", type="string"),
    AttributeInfo(name="created_date", description="The creation date of the document", type="string"),
    AttributeInfo(name="updated_date", description="The update date of the document", type="string"),
    AttributeInfo(name="source", description="The source system of the document", type="string"),
    AttributeInfo(name="tags", description="A list of tags associated with the document", type="string"),
]


def translate_filter(filter_expr: Any) -> Optional[Dict[str, Any]]:
    """
    Recursively translates a LangChain AST filter expression into a ChromaDB where dictionary.
    """
    if filter_expr is None:
        return None

    if isinstance(filter_expr, Comparison):
        comparator = filter_expr.comparator.value if hasattr(filter_expr.comparator, "value") else str(filter_expr.comparator)
        attr = filter_expr.attribute
        val = filter_expr.value

        # Map operator strings to Chroma DB where clauses
        if comparator == "eq":
            return {attr: val}
        elif comparator == "ne":
            return {attr: {"$ne": val}}
        elif comparator == "gt":
            return {attr: {"$gt": val}}
        elif comparator == "gte":
            return {attr: {"$gte": val}}
        elif comparator == "lt":
            return {attr: {"$lt": val}}
        elif comparator == "lte":
            return {attr: {"$lte": val}}
        elif comparator == "contains":
            # If not natively supported by Chroma, fallback to simple equality filter
            return {attr: val}
        else:
            return {attr: val}

    elif isinstance(filter_expr, Operation):
        operator = filter_expr.operator.value if hasattr(filter_expr.operator, "value") else str(filter_expr.operator)
        translated_args = [translate_filter(arg) for arg in filter_expr.arguments]
        translated_args = [arg for arg in translated_args if arg is not None]

        if not translated_args:
            return None
        if len(translated_args) == 1:
            return translated_args[0]

        if operator == "and":
            return {"$and": translated_args}
        elif operator == "or":
            return {"$or": translated_args}
        else:
            return {"$and": translated_args}

    return None


def patch_llm_with_retries(llm: Any, max_retries: int = 6, initial_delay: float = 4.0) -> Any:
    """
    Dynamically patches the LLM's invoke and generate methods to retry on rate limits (429/RESOURCE_EXHAUSTED).
    """
    original_invoke = llm.invoke
    original_generate = llm.generate

    def retrying_invoke(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return original_invoke(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Exception hit in patched invoke ({e}). "
                        f"Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

    def retrying_generate(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return original_generate(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Exception hit in patched generate ({e}). "
                        f"Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

    object.__setattr__(llm, "invoke", retrying_invoke)
    object.__setattr__(llm, "generate", retrying_generate)
    return llm


class ChromaSelfQueryRetriever(BaseRetriever):
    """
    Custom Self-Querying Retriever utilizing an LLM to parse metadata constraint filters
    from queries, translating them to Chroma filters, and routing to the RAG pipeline.
    """
    llm: Any = Field(description="The LLM instance to generate filters")
    owner_id: Optional[int] = Field(default=None, description="Context owner ID filter")
    document_id: Optional[int] = Field(default=None, description="Context document ID filter")
    top_k: int = Field(default=5, description="Number of results to retrieve")
    child_chunk_size: int = Field(default=300, description="Child splitter chunk size")
    child_overlap: int = Field(default=30, description="Child splitter overlap")
    parent_chunk_size: int = Field(default=1024, description="Parent chunk size")
    parent_overlap: int = Field(default=100, description="Parent overlap size")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Runs query constructor to build filters, and runs multi-stage retrieval pipeline.
        """
        start_time = time.time()
        logger.info(f"SelfQueryRetriever execution started for query: '{query}'")

        document_content_description = "Policy, guidelines, rules, leave allowance, and hr handbook documents."
        translated_filter = None
        semantic_query = query

        # 1. Self Query LLM Parsing
        try:
            llm_start = time.time()
            logger.info("Initializing query constructor chain...")
            query_constructor = load_query_constructor_chain(
                llm=self.llm,
                document_contents=document_content_description,
                attribute_info=metadata_field_info,
                verbose=False
            )

            logger.info("Invoking LLM structured query parsing...")
            structured_query = query_constructor.invoke({"query": query})
            llm_latency = (time.time() - llm_start) * 1000

            logger.info(f"Original Query: '{query}'")
            logger.info(f"Generated Structured Query: {structured_query}")
            logger.info(f"Semantic Query: '{structured_query.query}'")
            logger.info(f"Metadata Filters AST: {structured_query.filter}")
            logger.info(f"LLM parsing completed in {llm_latency:.2f}ms")

            # Translate AST Filter to Chroma filter dictionary
            translated_filter = translate_filter(structured_query.filter)
            logger.info(f"Chroma Metadata Filters: {translated_filter}")

            # Fallback semantic query if parser returns blank
            clean_semantic = structured_query.query.strip()
            if clean_semantic:
                semantic_query = clean_semantic

        except Exception as e:
            logger.error(f"LLM self-query parsing failed, falling back to Multi Query: {e}", exc_info=True)
            semantic_query = query
            translated_filter = None

        # 2. Combine extracted filters with session filters (owner_id, document_id)
        instance_filters = []
        if self.owner_id is not None:
            instance_filters.append({"owner_id": self.owner_id})
        if self.document_id is not None:
            instance_filters.append({"document_id": self.document_id})

        if translated_filter:
            instance_filters.append(translated_filter)

        combined_where = None
        if len(instance_filters) == 1:
            combined_where = instance_filters[0]
        elif len(instance_filters) > 1:
            combined_where = {"$and": instance_filters}

        logger.info(f"Combined where filter constraints: {combined_where}")

        # 3. Build downstream retrieval pipeline dynamically injecting combined_where
        try:
            from app.services.langchain.hybrid_retriever import get_hybrid_retriever
            from app.services.langchain.parent_retriever import ParentRetriever
            from app.services.langchain.compression import CompressionRetriever
            from app.services.langchain.multi_query import get_multi_query_retriever

            # A. Base Hybrid retriever with injected where_override
            logger.info("Initializing Hybrid search retriever with override filters...")
            hybrid_retriever = get_hybrid_retriever(
                owner_id=None,
                document_id=None,
                top_k=self.top_k
            )
            hybrid_retriever.where_override = combined_where

            # B. Parent Document retriever wrapping hybrid search
            logger.info("Initializing Parent document retriever...")
            parent_retriever_wrapper = ParentRetriever(
                owner_id=self.owner_id,
                document_id=self.document_id,
                top_k=self.top_k,
                child_chunk_size=self.child_chunk_size,
                child_overlap=self.child_overlap,
                parent_chunk_size=self.parent_chunk_size,
                parent_overlap=self.parent_overlap,
                base_retriever=hybrid_retriever
            )

            # C. Contextual Compression retriever wrapping parent document retriever
            logger.info("Initializing Contextual compression retriever...")
            compression_retriever_wrapper = CompressionRetriever(
                owner_id=self.owner_id,
                document_id=self.document_id,
                top_k=self.top_k,
                base_retriever=parent_retriever_wrapper.parent_document_retriever
            )

            # D. Multi Query retriever wrapping compression retriever
            logger.info("Initializing Multi Query retriever...")
            multi_query_retriever = get_multi_query_retriever(
                owner_id=self.owner_id,
                document_id=self.document_id,
                top_k=self.top_k,
                llm=self.llm,
                retriever=compression_retriever_wrapper.compression_retriever
            )

            # Run retrieval
            logger.info("Executing downstream retrieval pipeline...")
            retrieved_docs = multi_query_retriever.invoke(semantic_query)
            total_latency = (time.time() - start_time) * 1000
            
            logger.info(
                f"Self Query Retrieval completed successfully. Found {len(retrieved_docs)} documents. "
                f"Total retrieval latency: {total_latency:.2f}ms"
            )
            return retrieved_docs

        except Exception as e:
            logger.error(f"Downstream retrieval pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Self-Querying retrieval pipeline failed: {e}") from e


def get_self_query_retriever(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    llm: Optional[Any] = None,
    child_chunk_size: int = 300,
    child_overlap: int = 30,
    parent_chunk_size: int = 1024,
    parent_overlap: int = 100,
) -> BaseRetriever:
    """
    Factory function to construct a ChromaSelfQueryRetriever.
    """
    active_llm = llm or get_llm()
    patched_llm = patch_llm_with_retries(active_llm)
    return ChromaSelfQueryRetriever(
        llm=patched_llm,
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k,
        child_chunk_size=child_chunk_size,
        child_overlap=child_overlap,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=parent_overlap
    )
