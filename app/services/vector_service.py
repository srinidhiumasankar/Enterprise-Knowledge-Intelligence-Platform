# app/services/vector_service.py
# -----------------------------
# Service layer managing persistent ChromaDB operations.

import logging
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)


class VectorService:
    """
    Service class managing the lifetime and interactions of ChromaDB collection
    including client setup, vector insertion, deletion, and query operations.
    """

    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize persistent ChromaDB client and retrieve or create the target collection.
        
        Args:
            persist_directory: Path to database storage. If None, falls back to settings.
            collection_name: Name of ChromaDB collection. If None, falls back to settings.
        """
        self.persist_directory = persist_directory or settings.CHROMA_DB_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        # Ensure persist directory exists
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create persistent directory '{self.persist_directory}': {e}")
            raise RuntimeError(f"ChromaDB directory creation failed: {e}")

        logger.info(f"Initializing persistent ChromaDB client at: '{self.persist_directory}'")
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client/collection: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}")

    def insert_vectors(
        self,
        document_id: int,
        chunk_ids: List[str],
        chunk_indices: List[int],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Insert chunk text vectors and associated metadata for a document into ChromaDB.
        Ensures that existing vectors for this document are deleted before insertion to prevent duplicates.

        Args:
            document_id: The ID of the document being processed.
            chunk_ids: List of unique chunk string identifiers (usually chunk UUIDs).
            chunk_indices: List of integer indices corresponding to the chunk numbers.
            texts: List of the text chunks.
            embeddings: List of the floating-point vector representations.
            metadatas: Optional list of additional dictionaries with metadata context.
        """
        if not chunk_ids:
            logger.warning(f"Empty chunk list provided for insertion on document_id: {document_id}")
            return

        n = len(chunk_ids)
        if len(texts) != n or len(embeddings) != n or len(chunk_indices) != n:
            logger.error("List length mismatch in insert_vectors parameters.")
            raise ValueError(
                f"List lengths do not match: chunk_ids({len(chunk_ids)}), chunk_indices({len(chunk_indices)}), "
                f"texts({len(texts)}), embeddings({len(embeddings)})."
            )

        if metadatas is not None and len(metadatas) != n:
            logger.error("Metadata list length does not match chunk list length.")
            raise ValueError(f"metadatas list length ({len(metadatas)}) must match chunk_ids length ({n}).")

        # Prepare enriched metadata
        enriched_metadatas = []
        for i in range(n):
            meta = metadatas[i].copy() if metadatas else {}
            # Standardized metadata required for retrieval
            meta["document_id"] = document_id
            meta["chunk_id"] = chunk_ids[i]
            meta["chunk_index"] = chunk_indices[i]
            enriched_metadatas.append(meta)

        try:
            # 1. Clean up existing vectors for this document to avoid duplicates
            logger.info(f"Removing pre-existing vectors for document_id: {document_id}")
            self.delete_vectors_by_document_id(document_id)

            # 2. Add vectors to the collection
            logger.info(f"Inserting {n} vectors for document_id: {document_id} into ChromaDB collection '{self.collection_name}'")
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=enriched_metadatas,
                documents=texts
            )
            logger.info(f"Successfully stored vectors for document_id: {document_id}")
        except Exception as e:
            logger.error(f"Failed to insert vectors into ChromaDB for document_id {document_id}: {e}")
            raise RuntimeError(f"ChromaDB insert failed: {e}")

    def delete_vectors_by_document_id(self, document_id: int) -> None:
        """
        Delete all vectors associated with a specific document ID.

        Args:
            document_id: The ID of the document whose vectors should be removed.
        """
        try:
            logger.info(f"Deleting vectors associated with document_id: {document_id} from collection '{self.collection_name}'")
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Successfully deleted vectors for document_id: {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete vectors for document_id {document_id} from ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB delete failed: {e}")

    def fetch_vectors_by_document_id(self, document_id: int) -> Dict[str, Any]:
        """
        Retrieve all stored vector entries matching a given document ID.

        Args:
            document_id: The ID of the target document.

        Returns:
            Dict: Dictionary containing retrieved IDs, documents, metadatas, and embeddings.
        """
        try:
            logger.info(f"Fetching vectors for document_id: {document_id} from collection '{self.collection_name}'")
            results = self.collection.get(
                where={"document_id": document_id},
                include=["embeddings", "metadatas", "documents"]
            )
            return results
        except Exception as e:
            logger.error(f"Failed to fetch vectors for document_id {document_id} from ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB fetch failed: {e}")

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get basic diagnostic stats on the initialized collection.

        Returns:
            Dict: Collection statistics including collection name and total number of records.
        """
        try:
            count = self.collection.count()
            stats = {
                "collection_name": self.collection_name,
                "total_vectors": count,
            }
            logger.info(f"Collection stats for '{self.collection_name}': {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to retrieve collection statistics: {e}")
            raise RuntimeError(f"ChromaDB count failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a connection check on the persistent ChromaDB client.

        Returns:
            Dict: Connection status report.
        """
        try:
            heartbeat = self.client.heartbeat()
            # If client isn't responding, heartbeat returns an integer or throws an exception
            if heartbeat is not None:
                status_str = "healthy"
            else:
                status_str = "unhealthy"
            
            return {
                "status": status_str,
                "heartbeat": heartbeat,
                "db_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
