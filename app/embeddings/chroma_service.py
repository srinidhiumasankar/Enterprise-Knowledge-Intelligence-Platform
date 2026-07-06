# app/embeddings/chroma_service.py
# ---------------------------------
# Infrastructure layer managing vector storage operations inside ChromaDB.

import logging
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.embeddings import config

logger = logging.getLogger(__name__)


class ChromaService:
    """
    Service class managing the lifecycle and vector operations of ChromaDB.
    Follows a singleton pattern for the persistent client to prevent connection locks.
    """

    _client: Optional[chromadb.PersistentClient] = None
    _collection: Any = None

    def __init__(self, collection_name: Optional[str] = None) -> None:
        """
        Initialize the ChromaService instance.
        """
        self.custom_collection_name = collection_name
        self.initialize()

    def initialize(self) -> None:
        """
        Setup the persistent ChromaDB client using configured path settings.
        Validates collection dimension on startup and automatically heals dimension mismatches.
        """
        from app.config import settings
        import shutil

        # 1. Initialize persistent client
        if ChromaService._client is None:
            logger.info(f"Initializing persistent ChromaDB client at path: '{config.CHROMA_DB_PATH}'")
            try:
                os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
                ChromaService._client = chromadb.PersistentClient(
                    path=config.CHROMA_DB_PATH,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB PersistentClient: {e}")
                raise RuntimeError(f"ChromaDB persistent client initialization failed: {e}")

        self.client = ChromaService._client

        # Determine collection name
        target_name = self.custom_collection_name or config.CHROMA_COLLECTION_NAME

        # If using standard name and cached collection exists, reuse it
        if not self.custom_collection_name and ChromaService._collection is not None:
            self.collection = ChromaService._collection
            return

        # 2. Get or create collection with validation
        try:
            expected_dimension = settings.EMBEDDING_DIMENSION
            logger.info(f"Targeting collection '{target_name}' with expected dimension: {expected_dimension}")

            # Check if collection exists
            collection_names = [col.name for col in self.client.list_collections()]
            if target_name in collection_names:
                logger.info(f"Loaded collection '{target_name}'. Performing dimension check...")
                collection = self.client.get_collection(name=target_name)
                
                # Try to inspect dimension
                existing_dim = None
                if collection.metadata and "embedding_dimension" in collection.metadata:
                    existing_dim = collection.metadata["embedding_dimension"]
                    logger.info(f"Chroma collection metadata indicates dimension: {existing_dim}")
                else:
                    try:
                        peeked = collection.get(limit=1, include=["embeddings"])
                        if peeked and peeked.get("embeddings") and len(peeked["embeddings"]) > 0:
                            existing_dim = len(peeked["embeddings"][0])
                            logger.info(f"Peeked collection item indicates dimension: {existing_dim}")
                    except Exception as e:
                        logger.warning(f"Failed to inspect existing collection vector dimension: {e}")

                # Re-create collection on dimension mismatch or missing info
                if existing_dim is None or existing_dim != expected_dimension:
                    logger.warning(
                        f"Dimension mismatch or missing info for '{target_name}'. Recreating collection..."
                    )
                    self.client.delete_collection(name=target_name)
                    collection = self.client.create_collection(
                        name=target_name,
                        metadata={
                            "embedding_dimension": expected_dimension,
                            "model_name": config.EMBEDDING_MODEL_NAME,
                        },
                    )
                    logger.info(f"Collection '{target_name}' recreated successfully.")
                else:
                    # Update missing metadata if needed
                    if not collection.metadata or "embedding_dimension" not in collection.metadata:
                        collection.modify(
                            metadata={
                                "embedding_dimension": expected_dimension,
                                "model_name": config.EMBEDDING_MODEL_NAME
                            }
                        )
                self.collection = collection
            else:
                # Create new collection
                logger.info(f"Creating new collection '{target_name}' with dimension {expected_dimension}...")
                self.collection = self.client.create_collection(
                    name=target_name,
                    metadata={
                        "embedding_dimension": expected_dimension,
                        "model_name": config.EMBEDDING_MODEL_NAME
                    }
                )
                logger.info(f"Collection '{target_name}' created successfully.")

            # Cache the standard collection as singleton
            if not self.custom_collection_name:
                ChromaService._collection = self.collection

        except Exception as outer_e:
            logger.warning(f"ChromaDB validation error: {outer_e}. Safely resetting storage directory...")
            if not self.custom_collection_name:
                try:
                    ChromaService._client = None
                    ChromaService._collection = None
                    if os.path.exists(config.CHROMA_DB_PATH):
                        shutil.rmtree(config.CHROMA_DB_PATH)
                    os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
                    ChromaService._client = chromadb.PersistentClient(
                        path=config.CHROMA_DB_PATH,
                        settings=ChromaSettings(anonymized_telemetry=False)
                    )
                    self.client = ChromaService._client
                    self.collection = self.client.create_collection(
                        name=config.CHROMA_COLLECTION_NAME,
                        metadata={
                            "embedding_dimension": settings.EMBEDDING_DIMENSION,
                            "model_name": config.EMBEDDING_MODEL_NAME
                        }
                    )
                    ChromaService._collection = self.collection
                except Exception as inner_e:
                    logger.error(f"Fatal error resetting persistent client storage: {inner_e}")
                    raise RuntimeError(f"ChromaDB storage reset failed: {inner_e}")
            else:
                # For custom collection name, just delete and recreate it
                try:
                    self.client.delete_collection(name=target_name)
                    self.collection = self.client.create_collection(
                        name=target_name,
                        metadata={
                            "embedding_dimension": expected_dimension,
                            "model_name": config.EMBEDDING_MODEL_NAME
                        }
                    )
                except Exception as inner_e:
                    logger.error(f"Failed to reset custom collection: {inner_e}")
                    raise RuntimeError(f"ChromaDB custom collection reset failed: {inner_e}")

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        """
        Add a batch of document chunks and their embeddings to the vector database.

        Args:
            ids: List of unique chunk IDs (UUIDs).
            embeddings: List of matching float vectors.
            metadatas: List of dictionaries holding document meta attributes (owner_id, document_id, filename, chunk_id).
            documents: List of raw text chunks.
        """
        if not ids:
            logger.warning("Empty list of IDs passed to add_documents. Skipping.")
            return

        try:
            logger.info(f"Adding {len(ids)} documents to collection '{config.CHROMA_COLLECTION_NAME}'...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info("Successfully added documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB insertion failed: {e}")

    def similarity_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query ChromaDB to retrieve similar vectors based on distance metrics.

        Args:
            query_embedding: Vector embedding of the search query.
            n_results: Top K results to return.
            where: Metadata filtering parameters (e.g. {"owner_id": owner_id}).

        Returns:
            Dict[str, Any]: Query results dict containing matched document IDs, text content, and distances.
        """
        try:
            logger.info(f"Querying ChromaDB collection for top {n_results} matches. Filters: {where}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB query failed: {e}")

    def delete_document(self, document_id: int) -> None:
        """
        Delete all vector entries associated with a specific document ID.

        Args:
            document_id: The identifier of the document to purge.
        """
        try:
            logger.info(f"Deleting embeddings for document_id={document_id} from collection '{config.CHROMA_COLLECTION_NAME}'")
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Successfully deleted embeddings for document_id={document_id}")
        except Exception as e:
            logger.error(f"Error deleting embeddings for document_id={document_id}: {e}")
            raise RuntimeError(f"ChromaDB deletion failed: {e}")

    def update_document(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        """
        Update existing vectors, embeddings, or metadata in the collection.

        Args:
            ids: List of unique chunk IDs to update.
            embeddings: List of new embedding vectors.
            metadatas: List of updated metadata dictionaries.
            documents: List of new chunk texts.
        """
        if not ids:
            logger.warning("Empty list of IDs passed to update_document. Skipping.")
            return

        try:
            logger.info(f"Updating {len(ids)} documents in collection '{config.CHROMA_COLLECTION_NAME}'...")
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info("Successfully updated documents in ChromaDB.")
        except Exception as e:
            logger.error(f"Error updating documents in ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB update failed: {e}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic information and current statistics of the collection.

        Returns:
            Dict[str, Any]: Statistics containing collection name, metadata settings, and item count.
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": config.CHROMA_COLLECTION_NAME,
                "total_items": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error fetching collection info: {e}")
            raise RuntimeError(f"ChromaDB collection info query failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a connection/heartbeat test on ChromaDB.

        Returns:
            Dict[str, Any]: Status reporting dict indicating health status of connection.
        """
        try:
            heartbeat = self.client.heartbeat()
            status_str = "healthy" if heartbeat is not None else "unhealthy"
            return {
                "status": status_str,
                "heartbeat": heartbeat,
                "db_path": config.CHROMA_DB_PATH
            }
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    # -------------------------------------------------------------------------
    # Backward compatibility wrappers for Phase 6.1 schema
    # -------------------------------------------------------------------------

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Backward compatibility helper for create_collection."""
        return self.client.get_or_create_collection(name=name, metadata=metadata)

    def get_collection(self, name: str) -> Any:
        """Backward compatibility helper for get_collection."""
        return self.client.get_collection(name=name)

    def store_embeddings(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str]
    ) -> None:
        """Backward compatibility helper mapping store_embeddings to add_documents."""
        self.add_documents(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Backward compatibility helper mapping query to similarity_search."""
        # query_embeddings in Phase 6.1 was list-of-lists
        emb = query_embeddings[0] if query_embeddings else []
        return self.similarity_search(query_embedding=emb, n_results=n_results, where=where)

    def delete_document_embeddings(self, collection_name: str, document_id: int) -> None:
        """Backward compatibility helper mapping delete_document_embeddings to delete_document."""
        self.delete_document(document_id=document_id)

