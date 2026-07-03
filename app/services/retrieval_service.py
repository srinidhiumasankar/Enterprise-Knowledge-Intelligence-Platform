# app/services/retrieval_service.py
# ---------------------------------
# Service layer for semantic search and chunk retrieval from ChromaDB.

import logging
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.database.chunk_repository import ChunkRepository
from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.chroma_service import ChromaService

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service class responsible for orchestrating semantic search queries:
    1. Query embedding generation
    2. Vector database similarity search (with user isolation filter)
    3. Retrieval and ranking of matching chunks
    """

    def __init__(
        self,
        chunk_repository: ChunkRepository,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_service: Optional[ChromaService] = None,
        vector_service: Optional[Any] = None,
    ):
        self.chunk_repo = chunk_repository
        self.embedding_service = embedding_service or EmbeddingService()
        self.chroma_service = chroma_service or ChromaService()

    async def retrieve(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search and return a ranked list of text chunks.
        Only retrieves chunks belonging to the specified user_id.
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query string received for semantic search.")
            raise ValueError("Search query cannot be empty or whitespace-only.")

        start_time = time.time()
        logger.info(f"Initiating semantic search for user_id={user_id}, query='{query}', top_k={top_k}")

        if threshold is None:
            from app.embeddings import config
            threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.45)

        try:
            # Retrieve a larger list to account for threshold filtering and deduplication
            fetch_limit = max(top_k * 3, 20)

            # 2. Semantic Search via LangChain Retriever wrapping ChromaDB
            logger.info(f"Searching ChromaDB via LangChain retriever wrapper with owner_id filter={user_id}, limit={fetch_limit}...")
            from app.services.langchain.retriever import get_retriever
            retriever = get_retriever(
                owner_id=user_id,
                top_k=fetch_limit,
                chroma_service=self.chroma_service,
                embedding_service=self.embedding_service
            )
            retrieved_docs = retriever.invoke(query)

            # Convert LangChain Documents back to standard lists for backward compatibility
            ids = [doc.metadata.get("chunk_id") for doc in retrieved_docs]
            distances = [doc.metadata.get("distance", 0.0) for doc in retrieved_docs]
            documents = [doc.page_content for doc in retrieved_docs]
            metadatas = [doc.metadata for doc in retrieved_docs]

            total_retrieved = len(ids)
            logger.info(f"ChromaDB returned {total_retrieved} raw results.")

            if not ids:
                logger.info(f"No semantic search results found for user_id={user_id}")
                return []

            # 3. Retrieve chunks from database to obtain database primary key IDs
            db_chunks = self.chunk_repo.get_chunks_by_uuids(ids)
            uuid_to_chunk_id = {chunk.uuid: chunk.id for chunk in db_chunks}

            # 4. Rank, filter and deduplicate results
            ranked_results = []
            seen_texts = set()
            seen_chunk_ids = set()
            filtered_count = 0

            for idx, chunk_uuid in enumerate(ids):
                db_chunk_id = uuid_to_chunk_id.get(chunk_uuid)
                if db_chunk_id is None:
                    logger.warning(f"Chunk UUID '{chunk_uuid}' exists in ChromaDB but was not found in SQL database.")
                    continue

                distance = distances[idx]
                meta = metadatas[idx]
                doc_id = meta.get("document_id")
                chunk_text = documents[idx]

                # Convert L2 distance to similarity score
                coll_metadata = getattr(self.chroma_service.collection, "metadata", None) or {}
                hnsw_space = coll_metadata.get("hnsw:space", "l2")
                if hnsw_space == "cosine":
                    score = 1.0 - distance
                else:
                    score = 1.0 / (1.0 + distance)

                # Keep score within reasonable bounds [0.0, 1.0] and round to 4 decimal places
                score = max(0.0, min(1.0, score))
                score = round(score, 4)

                # Filter out below threshold
                if score < threshold:
                    filtered_count += 1
                    continue

                # Deduplicate based on text content and database ID
                cleaned_text_lower = chunk_text.strip().lower()
                if cleaned_text_lower in seen_texts or db_chunk_id in seen_chunk_ids:
                    continue

                seen_texts.add(cleaned_text_lower)
                seen_chunk_ids.add(db_chunk_id)

                ranked_results.append({
                    "document_id": doc_id,
                    "chunk_id": db_chunk_id,
                    "score": score,
                    "text": chunk_text,
                    "metadata": {
                        "document_id": doc_id,
                        "chunk_id": chunk_uuid,
                        "owner_id": meta.get("owner_id"),
                        "filename": meta.get("filename")
                    }
                })

                if len(ranked_results) >= top_k:
                    break

            # Sort by score in descending order
            ranked_results.sort(key=lambda x: x["score"], reverse=True)

            execution_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Search completed. Query: '{query}'. "
                f"Retrieved: {total_retrieved} raw chunks, Filtered: {filtered_count} under threshold, "
                f"Returned: {len(ranked_results)} unique ranked chunks. "
                f"Execution time: {execution_time_ms:.2f}ms."
            )
            return ranked_results

        except Exception as e:
            logger.error(f"Error during semantic retrieval: {e}", exc_info=True)
            raise RuntimeError(f"Semantic retrieval failed: {e}")
