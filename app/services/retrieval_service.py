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
        self.last_search_diagnostics = {}

    def _trigger_bg_record(
        self,
        user_id: int,
        workspace_id: int,
        query: str,
        collection_ids: Optional[List[int]],
        latency_ms: float,
        result_count: int
    ):
        try:
            from app.config import settings
            enable_history = getattr(settings, "ENABLE_SEARCH_HISTORY", True)
            if enable_history and user_id and workspace_id:
                import threading
                from app.database import SessionLocal
                from app.services.search_history.search_history_service import SearchHistoryService

                filters_json = {
                    "collection_ids": collection_ids
                }

                def bg_record():
                    db_bg = SessionLocal()
                    try:
                        sh_service = SearchHistoryService(db_bg)
                        sh_service.record_search(
                            user_id=user_id,
                            workspace_id=workspace_id,
                            query=query,
                            filters_json=filters_json,
                            execution_time_ms=int(latency_ms),
                            result_count=result_count
                        )
                    except Exception as bg_e:
                        logger.error(f"Failed to record search history in background: {bg_e}")
                    finally:
                        db_bg.close()

                thread = threading.Thread(target=bg_record)
                thread.daemon = True
                thread.start()
        except Exception as e:
            logger.error(f"Failed to trigger search history background recording: {e}")

    async def retrieve(
        self,
        user_id: int,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        collection_ids: Optional[List[int]] = None,
        workspace_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search and return a ranked list of text chunks.
        Only retrieves chunks belonging to the specified user_id.
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query string received for semantic search.")
            raise ValueError("Search query cannot be empty or whitespace-only.")

        start_time = time.time()
        logger.info(f"Initiating semantic search for user_id={user_id}, query='{query}', top_k={top_k}, collection_ids={collection_ids}, workspace_id={workspace_id}, document_ids={document_ids}")

        if threshold is None:
            from app.embeddings import config
            threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.45)

        try:
            # Retrieve a larger list to account for threshold filtering and deduplication
            fetch_limit = max(top_k * 3, 20)

            # Collection and Workspace filtering integration
            document_id = None
            from app.config import settings
            enable_filtering = getattr(settings, "ENABLE_COLLECTION_FILTERING", True)
            from app.database import SessionLocal
            from app.services.workspace.workspace_service import WorkspaceService
            from app.services.collection.collection_filter_service import CollectionFilterService

            db = SessionLocal()
            try:
                ws_service = WorkspaceService(db)
                if not workspace_id:
                    ws = ws_service.get_active_workspace(user_id)
                    workspace_id = ws.id
                else:
                    ws_service.validate_workspace_ownership(user_id, workspace_id)

                if workspace_id:
                    if document_ids is not None:
                        resolved_doc_ids = document_ids
                    else:
                        filter_service = CollectionFilterService(db)
                        resolved_doc_ids = filter_service.validate_and_resolve_filters(
                            user_id=user_id,
                            workspace_id=workspace_id,
                            collection_ids=collection_ids
                        )
                    if not resolved_doc_ids:
                        document_id = -1
                    elif len(resolved_doc_ids) == 1:
                        document_id = resolved_doc_ids[0]
                    else:
                        document_id = {"$in": resolved_doc_ids}
            finally:
                db.close()

            # 2. Semantic Search via LangChain Retriever wrapping ChromaDB
            logger.info(f"Searching ChromaDB via LangChain retriever wrapper with owner_id filter={user_id}, document_id={document_id}, limit={fetch_limit}...")
            from app.services.langchain.retriever import get_retriever
            retriever = get_retriever(
                owner_id=user_id,
                document_id=document_id,
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

            # Initialize diagnostics dictionary
            self.last_search_diagnostics = {
                "query": query,
                "raw_retrieved_count": total_retrieved,
                "filtered_count": 0,
                "db_mismatch_count": 0,
                "duplicate_count": 0,
                "threshold": threshold,
                "why_zero_chunks": "",
                "chroma_returned_zero": False,
                "all_below_threshold": False,
                "all_duplicates": False
            }

            if not ids:
                logger.info(f"No semantic search results found for user_id={user_id}")
                execution_time_ms = (time.time() - start_time) * 1000
                self.last_search_diagnostics["chroma_returned_zero"] = True
                self.last_search_diagnostics["why_zero_chunks"] = "ChromaDB vector search returned 0 results for the query. Either the vector store is empty, or no chunks match the query embedding."
                self._trigger_bg_record(user_id, workspace_id, query, collection_ids, execution_time_ms, 0)
                return []

            # 3. Retrieve chunks from database to obtain database primary key IDs
            db_chunks = self.chunk_repo.get_chunks_by_uuids(ids)
            uuid_to_chunk_id = {chunk.uuid: chunk.id for chunk in db_chunks}

            # 4. Rank, filter and deduplicate results
            ranked_results = []
            seen_texts = set()
            seen_chunk_ids = set()
            filtered_count = 0
            db_mismatch_count = 0
            duplicate_count = 0

            for idx, chunk_uuid in enumerate(ids):
                db_chunk_id = uuid_to_chunk_id.get(chunk_uuid)
                if db_chunk_id is None:
                    logger.warning(f"Chunk UUID '{chunk_uuid}' exists in ChromaDB but was not found in SQL database.")
                    db_mismatch_count += 1
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
                    duplicate_count += 1
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

            # Update diagnostics
            self.last_search_diagnostics["filtered_count"] = filtered_count
            self.last_search_diagnostics["db_mismatch_count"] = db_mismatch_count
            self.last_search_diagnostics["duplicate_count"] = duplicate_count

            if len(ranked_results) == 0:
                if total_retrieved > 0 and filtered_count == total_retrieved:
                    self.last_search_diagnostics["all_below_threshold"] = True
                    self.last_search_diagnostics["why_zero_chunks"] = f"All {total_retrieved} retrieved chunks were filtered out because their similarity scores were below the threshold of {threshold}."
                elif total_retrieved > 0 and db_mismatch_count == total_retrieved:
                    self.last_search_diagnostics["why_zero_chunks"] = f"All {total_retrieved} retrieved chunks were missing from the SQL database (UUID mismatch)."
                elif total_retrieved > 0 and duplicate_count == total_retrieved:
                    self.last_search_diagnostics["all_duplicates"] = True
                    self.last_search_diagnostics["why_zero_chunks"] = "All retrieved chunks were filtered out as duplicates."
                else:
                    self.last_search_diagnostics["why_zero_chunks"] = f"No chunks remained after filtering (mismatch: {db_mismatch_count}, below threshold: {filtered_count}, duplicate: {duplicate_count})."

            logger.info(
                f"Search completed. Query: '{query}'. "
                f"Retrieved: {total_retrieved} raw chunks, Filtered: {filtered_count} under threshold, "
                f"Returned: {len(ranked_results)} unique ranked chunks. "
                f"Execution time: {execution_time_ms:.2f}ms."
            )

            # --- TEMPORARY DIAGNOSTIC LOGS ---
            logger.info(f"Retrieved chunk count: {len(ranked_results)}")
            logger.info(f"Similarity threshold: {threshold}")
            for idx, r in enumerate(ranked_results, 1):
                preview = r["text"][:200]
                logger.info(f"Chunk {idx} - Score: {r['score']}, Preview: {preview}")
            # ----------------------------------

            # self._trigger_bg_record(user_id, workspace_id, query, collection_ids, execution_time_ms, len(ranked_results))

            return ranked_results

        except (KeyError, ValueError, PermissionError) as e:
            raise e
        except Exception as e:
            logger.error(f"Error during semantic retrieval: {e}", exc_info=True)
            raise RuntimeError(f"Semantic retrieval failed: {e}")
