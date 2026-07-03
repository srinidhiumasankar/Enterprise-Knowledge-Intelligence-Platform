# app/services/langchain/multi_query.py
# ------------------------------------
# LangChain Multi Query Retriever that uses an LLM to generate diverse search queries
# and executes hybrid search across all generated terms.

import logging
import time
import re
from typing import Any, List, Dict, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from app.services.langchain.llm import get_llm
from app.services.langchain.retriever import get_retriever

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant tasked with generating five semantically different search queries for a vector database.
The original query is: "{question}"

Your goal is to generate 5 diverse queries covering synonyms, alternative wordings, technical phrasing, beginner phrasing, and alternate sentence structures.
Provide these queries one per line. Do not add any bullet points, numbers, explanations, introductory text, or concluding text. Just write the queries, one per line.
"""
)


class MultiQueryRetriever(BaseRetriever):
    """
    LangChain Retriever that generates multiple diverse queries using an LLM,
    runs the underlying Hybrid Retriever for each, and applies Reciprocal Rank Fusion
    to produce ranked, deduplicated search results.
    """
    llm: Any = Field(description="The LLM instance to generate queries")
    retriever: BaseRetriever = Field(description="The underlying hybrid retriever instance")
    top_k: int = Field(default=5, description="Number of top documents to return")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Generates alternate queries, retrieves documents for each query, and merges them.
        """
        start_time = time.time()
        logger.info("========================================================")
        logger.info(f"Starting Multi Query Retriever for: '{query}'")
        logger.info(f"Original Query: '{query}'")

        try:
            # 1. Generate alternate queries using Gemini
            logger.info("Generating multiple search queries...")
            prompt = MULTI_QUERY_PROMPT.format(question=query)
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse lines robustly
            generated_queries = []
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Strip leading numbers/bullets (e.g. "1.", "2)", "-", "*")
                clean_line = re.sub(r'^\s*[\d\-\*\+\•\)\.]+\s*', '', line).strip()
                if clean_line:
                    generated_queries.append(clean_line)

            # Keep unique queries to avoid redundant retrieval
            seen_queries = set()
            unique_generated = []
            for q in generated_queries:
                if q.lower() not in seen_queries:
                    seen_queries.add(q.lower())
                    unique_generated.append(q)

            # Ensure we have at least 3 queries, adding the original if needed
            if len(unique_generated) < 3:
                logger.warning(f"Generated only {len(unique_generated)} queries. Adding original query.")
                if query.lower() not in seen_queries:
                    unique_generated.insert(0, query)

            logger.info("Generated Queries:")
            for idx, gq in enumerate(unique_generated, 1):
                logger.info(f"{idx} {gq}")

            # 2. Retrieve documents for each query
            logger.info("Running Hybrid Retriever...")
            all_retrieved_docs: List[List[Document]] = []
            
            for idx, gq in enumerate(unique_generated, 1):
                q_start = time.time()
                docs = self.retriever.invoke(gq)
                all_retrieved_docs.append(docs)
                q_latency = (time.time() - q_start) * 1000
                logger.info(f"Query {idx}: '{gq}' -> Retrieved {len(docs)} documents in {q_latency:.2f}ms")

            # 3. Merge and RRF Rank Fusion
            logger.info("Merging retrieved documents...")
            rrf_constant = 60.0
            rrf_scores = {}
            doc_map = {}

            for query_idx, docs in enumerate(all_retrieved_docs):
                for rank, doc in enumerate(docs):
                    # Use chunk_id or document_id + chunk_index as deduplication key
                    chunk_id = doc.metadata.get("chunk_id")
                    if not chunk_id:
                        chunk_id = f"{doc.metadata.get('document_id')}_{doc.metadata.get('chunk_index')}"
                    
                    if chunk_id not in doc_map:
                        doc_map[chunk_id] = doc
                    else:
                        doc_map[chunk_id].metadata.update(doc.metadata)

                    # Accumulate RRF scores
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rank + rrf_constant))

            unique_chunks_count = len(doc_map)
            logger.info(f"Unique chunks after deduplication: {unique_chunks_count}")

            # 4. Rank by fused RRF scores
            logger.info("Ranking documents...")
            ranked_docs = []
            for chunk_id, score in rrf_scores.items():
                doc = doc_map[chunk_id]
                doc.metadata["multi_query_rrf_score"] = score
                ranked_docs.append(doc)

            # Sort descending by cumulative score
            ranked_docs.sort(key=lambda x: x.metadata["multi_query_rrf_score"], reverse=True)
            final_docs = ranked_docs[:self.top_k]

            total_latency = (time.time() - start_time) * 1000
            logger.info(f"Returning Top {len(final_docs)} documents")
            logger.info(f"Multi Query Retriever completed in {total_latency:.2f}ms")
            logger.info("========================================================")

            return final_docs

        except Exception as e:
            logger.error(f"MultiQueryRetriever failed: {e}", exc_info=True)
            raise RuntimeError(f"MultiQueryRetriever retrieval failed: {e}") from e


def get_multi_query_retriever(
    owner_id: Optional[int] = None,
    document_id: Optional[int] = None,
    top_k: int = 5,
    llm: Optional[Any] = None,
    retriever: Optional[BaseRetriever] = None,
) -> BaseRetriever:
    """
    Factory function to construct a MultiQueryRetriever.
    """
    logger.info("Initializing LangChain Multi Query retriever...")
    
    active_llm = llm or get_llm()
    active_retriever = retriever or get_retriever(
        owner_id=owner_id,
        document_id=document_id,
        top_k=top_k
    )

    return MultiQueryRetriever(
        llm=active_llm,
        retriever=active_retriever,
        top_k=top_k
    )
