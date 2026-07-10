# app/utils/chroma_cleaner.py
# ---------------------------
# Utility script to clean up vector entries for deleted or foreign documents in ChromaDB.

import logging
from sqlalchemy import select
from app.database.connection import SessionLocal
from app.models.document import Document
from app.embeddings.chroma_service import ChromaService

logger = logging.getLogger(__name__)


def cleanup_stale_vectors() -> None:
    """
    Scans ChromaDB and purges any vector embeddings associated with documents
    that are not present in the main SQL database.
    """
    logger.info("Starting ChromaDB stale vectors cleanup...")
    try:
        chroma_service = ChromaService()
        # Retrieve stored items to extract metadata
        results = chroma_service.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        if not metadatas:
            logger.info("No vectors found in ChromaDB collection.")
            return

        # Locate unique document IDs present in ChromaDB
        chroma_doc_ids = set()
        for meta in metadatas:
            if meta and "document_id" in meta:
                try:
                    chroma_doc_ids.add(int(meta["document_id"]))
                except (ValueError, TypeError):
                    continue

        if not chroma_doc_ids:
            logger.info("No document metadata keys resolved in ChromaDB.")
            return

        logger.info(f"Resolved document IDs present in ChromaDB: {list(chroma_doc_ids)}")

        # Fetch valid document IDs from database
        db = SessionLocal()
        try:
            db_ids = set(db.scalars(select(Document.id)).all())
        finally:
            db.close()

        logger.info(f"Resolved document IDs present in SQL DB: {list(db_ids)}")

        # Determine which vector entries are stale
        stale_ids = chroma_doc_ids - db_ids
        if stale_ids:
            logger.warning(f"Found stale document vector collections to purge: {list(stale_ids)}")
            for doc_id in stale_ids:
                try:
                    chroma_service.delete_document(doc_id)
                    logger.info(f"Purged stale vector embeddings for document ID: {doc_id}")
                except Exception as ex:
                    logger.error(f"Failed to delete stale vector embeddings for document ID {doc_id}: {ex}")
        else:
            logger.info("Vector database is fully synchronized with SQLite database. No stale embeddings found.")

    except Exception as e:
        logger.error(f"ChromaDB stale vector sync failed: {e}", exc_info=True)
