# tests/verify_phase_5_2.py
# -------------------------
# Automated verification script for Vector Database Layer (Phase 5.2).

import logging
import os
import shutil
import sys
from typing import List

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.vector_service import VectorService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_5_2")


def clean_test_dir(path: str):
    """
    Remove the test directory if it exists.
    """
    if os.path.exists(path):
        logger.info(f"Cleaning up directory: {path}")
        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Could not clean directory {path}: {e}")


def run_tests():
    logger.info("Starting Phase 5.2 Vector Database Verification...")

    # Test settings
    test_db_dir = "test_chroma_db"
    test_collection = "test_collection_chunks"
    dim = settings.EMBEDDING_DIMENSION

    # Ensure clean start
    clean_test_dir(test_db_dir)

    try:
        # 1. Initialize VectorService
        logger.info("Step 1: Initializing VectorService...")
        vector_service = VectorService(persist_directory=test_db_dir, collection_name=test_collection)
        
        # 2. Check Health
        logger.info("Step 2: Performing Health Check...")
        health = vector_service.health_check()
        logger.info(f"Health Check Response: {health}")
        assert health["status"] == "healthy", "Health check failed."
        assert health["heartbeat"] is not None, "Heartbeat is missing."

        # 3. Check Statistics on empty collection
        logger.info("Step 3: Checking initial statistics...")
        stats = vector_service.get_collection_statistics()
        logger.info(f"Initial Collection Stats: {stats}")
        assert stats["total_vectors"] == 0, f"Expected 0 vectors, got {stats['total_vectors']}"

        # 4. Insert mock vectors
        logger.info("Step 4: Inserting mock vectors...")
        doc_id = 999
        chunk_ids = ["chunk_uuid_1", "chunk_uuid_2", "chunk_uuid_3"]
        chunk_indices = [1, 2, 3]
        texts = [
            "FastAPI is a modern web framework for building APIs with Python.",
            "SQLAlchemy is an open-source SQL toolkit and object-relational mapper.",
            "ChromaDB is the AI-native open-source vector database."
        ]
        # Generate mock embeddings of correct dimension
        embeddings = [
            [0.1] * dim,
            [0.2] * dim,
            [0.3] * dim
        ]
        metadatas = [
            {"source": "fastapi_doc"},
            {"source": "sqlalchemy_doc"},
            {"source": "chromadb_doc"}
        ]

        vector_service.insert_vectors(
            document_id=doc_id,
            chunk_ids=chunk_ids,
            chunk_indices=chunk_indices,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # 5. Fetch and verify stored content
        logger.info("Step 5: Fetching and verifying stored vectors...")
        fetched = vector_service.fetch_vectors_by_document_id(doc_id)
        assert fetched["ids"] is not None, "Fetched ids are missing."
        assert len(fetched["ids"]) == 3, f"Expected 3 records, got {len(fetched['ids'])}"
        
        # Verify text and metadata are stored correctly
        for i, cid in enumerate(chunk_ids):
            idx = fetched["ids"].index(cid)
            assert fetched["documents"][idx] == texts[i], f"Text mismatch at index {i}"
            assert fetched["metadatas"][idx]["document_id"] == doc_id, f"Doc ID metadata mismatch"
            assert fetched["metadatas"][idx]["chunk_id"] == cid, f"Chunk ID metadata mismatch"
            assert fetched["metadatas"][idx]["chunk_index"] == chunk_indices[i], f"Chunk Index metadata mismatch"
            assert fetched["metadatas"][idx]["source"] == metadatas[i]["source"], f"Extra metadata mismatch"
            # Check embedding distance is zero or close to original
            original_emb = embeddings[i]
            fetched_emb = fetched["embeddings"][idx]
            assert len(fetched_emb) == dim, "Fetched embedding dimension mismatch."
            for v1, v2 in zip(original_emb, fetched_emb):
                assert abs(v1 - v2) < 1e-5, f"Embedding float mismatch: {v1} vs {v2}"

        logger.info("Stored text, embeddings, and metadata verified successfully!")

        # 6. Verify Persistence (Client Restart)
        logger.info("Step 6: Testing client persistence across restart...")
        # Release the current client references
        del vector_service
        
        # Re-initialize service with same directory
        vector_service = VectorService(persist_directory=test_db_dir, collection_name=test_collection)
        fetched_after_restart = vector_service.fetch_vectors_by_document_id(doc_id)
        assert len(fetched_after_restart["ids"]) == 3, "Vectors lost after client restart!"
        logger.info("Persistence check passed! Vectors survived client recreation.")

        # 7. Verify duplicate prevention (repeated processing)
        logger.info("Step 7: Testing duplicate prevention (re-processing)...")
        # Re-process document 999 with 2 new chunks
        new_chunk_ids = ["chunk_uuid_new1", "chunk_uuid_new2"]
        new_chunk_indices = [1, 2]
        new_texts = [
            "FastAPI is extremely fast compared to NodeJS and Go.",
            "ChromaDB makes it easy to build developer-centric RAG applications."
        ]
        new_embeddings = [
            [0.5] * dim,
            [0.6] * dim
        ]
        new_metadatas = [
            {"source": "fastapi_new"},
            {"source": "chromadb_new"}
        ]

        vector_service.insert_vectors(
            document_id=doc_id,
            chunk_ids=new_chunk_ids,
            chunk_indices=new_chunk_indices,
            texts=new_texts,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )

        # Retrieve vectors for document 999 again
        fetched_after_update = vector_service.fetch_vectors_by_document_id(doc_id)
        logger.info(f"Fetched IDs after update: {fetched_after_update['ids']}")
        assert len(fetched_after_update["ids"]) == 2, f"Expected 2 vectors, found {len(fetched_after_update['ids'])}. Old vectors were not deleted."
        assert "chunk_uuid_1" not in fetched_after_update["ids"], "Old vector chunk_uuid_1 still present."
        assert "chunk_uuid_new1" in fetched_after_update["ids"], "New vector chunk_uuid_new1 missing."
        logger.info("Duplicate prevention verified successfully! Old vectors were replaced by new ones.")

        # 8. Check statistics
        logger.info("Step 8: Checking collection statistics...")
        stats = vector_service.get_collection_statistics()
        logger.info(f"Current Collection Stats: {stats}")
        assert stats["total_vectors"] == 2, f"Expected 2 vectors total, got {stats['total_vectors']}"

        # 9. Delete vectors by document_id
        logger.info("Step 9: Testing vector deletion by document ID...")
        vector_service.delete_vectors_by_document_id(doc_id)
        fetched_after_delete = vector_service.fetch_vectors_by_document_id(doc_id)
        assert len(fetched_after_delete["ids"]) == 0, f"Expected 0 vectors after delete, got {len(fetched_after_delete['ids'])}"
        
        stats_final = vector_service.get_collection_statistics()
        assert stats_final["total_vectors"] == 0, f"Collection count should be 0, got {stats_final['total_vectors']}"
        logger.info("Vector deletion by document ID verified successfully!")

        logger.info("ALL TESTS PASSED SUCCESSFULLY!")

    except AssertionError as ae:
        logger.error(f"Verification Failed (AssertionError): {ae}")
        clean_test_dir(test_db_dir)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Verification Failed (Unexpected Error): {e}", exc_info=True)
        clean_test_dir(test_db_dir)
        sys.exit(1)
    finally:
        # Cleanup
        clean_test_dir(test_db_dir)


if __name__ == "__main__":
    run_tests()
