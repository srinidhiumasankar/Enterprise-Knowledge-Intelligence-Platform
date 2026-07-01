# app/embeddings/vector_store_service.py
# --------------------------------------
# Business logic service layer orchestrating text embedding and vector database management.

import logging
from typing import Any, Dict, List, Optional

from app.embeddings.embedding_service import EmbeddingService
from app.embeddings.chroma_service import ChromaService

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    OOP service coordinating text embedding extraction and persistent vector storage.
    Acts as a bridge between EmbeddingService and ChromaService.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_service: Optional[ChromaService] = None
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.chroma_service = chroma_service or ChromaService()

    def add_documents(
        self,
        document_id: int,
        chunks: List[Any],
        filename: str,
        owner_id: int
    ) -> None:
        """
        Extract embeddings for a list of text chunks and store them in ChromaDB.

        Args:
            document_id: Database ID of the document.
            chunks: List of chunk objects (having uuid and chunk_text attributes).
            filename: Original file name.
            owner_id: ID of the user owning the document.
        """
        if not chunks:
            logger.warning(f"No chunks provided for document_id={document_id}.")
            return

        try:
            chunk_texts = [c.chunk_text for c in chunks]
            chunk_ids = [c.uuid for c in chunks]

            logger.info(f"Extracting batch embeddings for {len(chunks)} chunks of document_id={document_id}...")
            embeddings = self.embedding_service.generate_batch_embeddings(chunk_texts)

            # Build standardized metadata
            metadatas = [
                {
                    "document_id": document_id,
                    "chunk_id": c.uuid,
                    "owner_id": owner_id,
                    "filename": filename
                }
                for c in chunks
            ]

            logger.info(f"Storing vectors inside ChromaDB collection for document_id={document_id}...")
            self.chroma_service.add_documents(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=chunk_texts
            )
            logger.info(f"Vector storage completed successfully for document_id={document_id}.")
        except Exception as e:
            logger.error(f"Failed to add document vectors to ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"VectorStore add_documents failed: {e}")

    def delete_document(self, document_id: int) -> None:
        """
        Remove all vectors associated with a document ID from ChromaDB.

        Args:
            document_id: The ID of the document to purge.
        """
        try:
            logger.info(f"Purging vector index entries for document_id={document_id}...")
            self.chroma_service.delete_document(document_id)
            logger.info(f"Vectors purged successfully for document_id={document_id}.")
        except Exception as e:
            logger.error(f"Failed to delete document vectors from ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"VectorStore delete_document failed: {e}")

    def update_document(
        self,
        document_id: int,
        chunks: List[Any],
        filename: str,
        owner_id: int
    ) -> None:
        """
        Update the vector representations of a document by replacing old indexes with new ones.

        Args:
            document_id: Database ID of the document.
            chunks: List of updated chunk objects.
            filename: Original file name.
            owner_id: User owner ID.
        """
        try:
            logger.info(f"Updating vector index entries for document_id={document_id}...")
            # Deletion followed by addition guarantees clean, idempotent replacement
            self.delete_document(document_id)
            self.add_documents(
                document_id=document_id,
                chunks=chunks,
                filename=filename,
                owner_id=owner_id
            )
            logger.info(f"Vector index entries updated successfully for document_id={document_id}.")
        except Exception as e:
            logger.error(f"Failed to update document vectors in ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"VectorStore update_document failed: {e}")

    def similarity_search(
        self,
        query: str,
        owner_id: int,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic query search on ChromaDB, isolated by owner_id.

        Args:
            query: User's search query string.
            owner_id: Authenticated user ID (scope isolation).
            top_k: Limit of matching chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Ranked results containing text, document_id, and metadata attributes.
        """
        if not query or not query.strip():
            logger.warning("Empty query string received for similarity_search.")
            return []

        try:
            logger.info(f"Generating query embedding representation for query='{query}'...")
            query_embedding = self.embedding_service.generate_query_embedding(query)

            # Query ChromaDB with owner filter
            logger.info(f"Executing ChromaDB similarity search (owner_id={owner_id}, top_k={top_k})...")
            search_results = self.chroma_service.similarity_search(
                query_embedding=query_embedding,
                n_results=top_k,
                where={"owner_id": owner_id}
            )

            # Format the output into structured results list
            formatted_results = []
            ids = search_results.get("ids", [[]])[0] if search_results.get("ids") else []
            metadatas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
            documents = search_results.get("documents", [[]])[0] if search_results.get("documents") else []
            distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []

            for idx in range(len(ids)):
                formatted_results.append({
                    "id": ids[idx],
                    "document_id": metadatas[idx].get("document_id"),
                    "chunk_id": metadatas[idx].get("chunk_id"),
                    "owner_id": metadatas[idx].get("owner_id"),
                    "filename": metadatas[idx].get("filename"),
                    "text": documents[idx],
                    "score": round(float(distances[idx]), 4) if idx < len(distances) else 0.0
                })

            logger.info(f"Query returned {len(formatted_results)} matching vector results.")
            return formatted_results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            raise RuntimeError(f"VectorStore similarity_search failed: {e}")
