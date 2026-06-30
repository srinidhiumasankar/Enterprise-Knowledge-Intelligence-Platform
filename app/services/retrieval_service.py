# app/services/retrieval_service.py
# ---------------------------------
# Service layer for semantic search and chunk retrieval from ChromaDB.

import logging
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.database.chunk_repository import ChunkRepository
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService

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
        vector_service: Optional[VectorService] = None,
    ):
        self.chunk_repo = chunk_repository
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or VectorService()

    async def retrieve(
        self,
        user_id: int,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search and return a ranked list of text chunks.
        Only retrieves chunks belonging to the specified user_id.
        """
        if not query or not query.strip():
            logger.warning("Empty query string received for semantic search.")
            return []

        start_time = time.time()
        logger.info(f"Initiating semantic search for user_id={user_id}, query='{query}', top_k={top_k}")

        try:
            # 1. Generate Query Embedding
            logger.info("Generating query embedding...")
            query_embedding = self.embedding_service.embed_text(query)

            # 2. Semantic Search in ChromaDB (user isolated)
            logger.info(f"Searching ChromaDB collection with user_id filter={user_id}...")
            search_results = self.vector_service.search_vectors(
                query_embedding=query_embedding,
                limit=top_k,
                user_id=user_id
            )

            # Extract IDs, distances, documents, and metadatas
            ids = search_results.get("ids", [[]])[0] if search_results.get("ids") else []
            distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []
            documents = search_results.get("documents", [[]])[0] if search_results.get("documents") else []
            metadatas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []

            if not ids:
                logger.info(f"No semantic search results found for user_id={user_id}")
                return []

            # 3. Retrieve chunks from database to obtain database primary key IDs
            db_chunks = self.chunk_repo.get_chunks_by_uuids(ids)
            uuid_to_chunk_id = {chunk.uuid: chunk.id for chunk in db_chunks}

            # 4. Rank and format results
            ranked_results = []
            for idx, chunk_uuid in enumerate(ids):
                # Retrieve db primary key
                db_chunk_id = uuid_to_chunk_id.get(chunk_uuid)
                if db_chunk_id is None:
                    logger.warning(f"Chunk UUID '{chunk_uuid}' exists in ChromaDB but was not found in SQL database.")
                    continue

                distance = distances[idx]
                doc_id = metadatas[idx].get("document_id")
                chunk_text = documents[idx]

                # Convert L2 distance to similarity score
                # ChromaDB defaults to L2 distance where similarity = 1.0 / (1.0 + distance)
                # Cosine distance similarity = 1.0 - distance
                coll_metadata = getattr(self.vector_service.collection, "metadata", None) or {}
                hnsw_space = coll_metadata.get("hnsw:space", "l2")
                if hnsw_space == "cosine":
                    score = 1.0 - distance
                else:
                    score = 1.0 / (1.0 + distance)

                # Keep score within reasonable bounds [0.0, 1.0] and round to 4 decimal places
                score = max(0.0, min(1.0, score))
                score = round(score, 4)

                ranked_results.append({
                    "document_id": doc_id,
                    "chunk_id": db_chunk_id,
                    "score": score,
                    "text": chunk_text
                })

            # Sort by score in descending order
            ranked_results.sort(key=lambda x: x["score"], reverse=True)

            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Semantic search completed in {retrieval_time:.2f}ms. Returned {len(ranked_results)} results.")
            return ranked_results

        except Exception as e:
            logger.error(f"Error during semantic retrieval: {e}", exc_info=True)
            raise RuntimeError(f"Semantic retrieval failed: {e}")
