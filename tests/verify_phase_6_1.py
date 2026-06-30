# tests/verify_phase_6_1.py
# -------------------------
# Verification script for Phase 6.1: Embedding Infrastructure Setup.

import logging
import sys
import os

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import EmbeddingService, ChromaService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_6_1")


def verify_infrastructure():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 6.1 INFRASTRUCTURE INTERFACE VERIFICATION")
    logger.info("==========================================================")

    # 1. Verify config settings
    logger.info("Step 1: Checking configuration parameters in settings.py...")
    assert hasattr(settings, "EMBEDDING_MODEL_NAME"), "EMBEDDING_MODEL_NAME setting is missing!"
    assert hasattr(settings, "CHROMA_DB_PATH"), "CHROMA_DB_PATH setting is missing!"
    assert hasattr(settings, "TOP_K_RESULTS"), "TOP_K_RESULTS setting is missing!"
    
    logger.info(f"✓ Configuration loaded. Model Name: '{settings.EMBEDDING_MODEL_NAME}'")
    logger.info(f"✓ Configuration loaded. DB Path: '{settings.CHROMA_DB_PATH}'")
    logger.info(f"✓ Configuration loaded. Top K Results: {settings.TOP_K_RESULTS}")

    # 2. Instantiate and verify EmbeddingService
    logger.info("\nStep 2: Instantiating and verifying EmbeddingService...")
    embedding_service = EmbeddingService()
    assert hasattr(embedding_service, "load_model"), "EmbeddingService lacks 'load_model' method!"
    assert hasattr(embedding_service, "generate_embedding"), "EmbeddingService lacks 'generate_embedding' method!"
    assert hasattr(embedding_service, "generate_batch_embeddings"), "EmbeddingService lacks 'generate_batch_embeddings' method!"
    assert hasattr(embedding_service, "health_check"), "EmbeddingService lacks 'health_check' method!"
    
    # Check that health check returns uninitialized state skeleton
    health_emb = embedding_service.health_check()
    assert isinstance(health_emb, dict), "EmbeddingService health_check must return a dictionary!"
    assert health_emb.get("status") == "uninitialized", "Expected default 'uninitialized' status!"
    logger.info("✓ EmbeddingService structural verification passed.")

    # 3. Instantiate and verify ChromaService
    logger.info("\nStep 3: Instantiating and verifying ChromaService...")
    chroma_service = ChromaService()
    assert hasattr(chroma_service, "initialize"), "ChromaService lacks 'initialize' method!"
    assert hasattr(chroma_service, "create_collection"), "ChromaService lacks 'create_collection' method!"
    assert hasattr(chroma_service, "get_collection"), "ChromaService lacks 'get_collection' method!"
    assert hasattr(chroma_service, "store_embeddings"), "ChromaService lacks 'store_embeddings' method!"
    assert hasattr(chroma_service, "query"), "ChromaService lacks 'query' method!"
    assert hasattr(chroma_service, "delete_document_embeddings"), "ChromaService lacks 'delete_document_embeddings' method!"
    assert hasattr(chroma_service, "health_check"), "ChromaService lacks 'health_check' method!"

    # Check that health check returns uninitialized state skeleton
    health_chroma = chroma_service.health_check()
    assert isinstance(health_chroma, dict), "ChromaService health_check must return a dictionary!"
    assert health_chroma.get("status") == "uninitialized", "Expected default 'uninitialized' status!"
    logger.info("✓ ChromaService structural verification passed.")

    logger.info("\n==========================================================")
    logger.info("ALL PHASE 6.1 ARCHITECTURAL CHECKS COMPLETED SUCCESSFULLY!")
    logger.info("==========================================================")


if __name__ == "__main__":
    try:
        verify_infrastructure()
    except AssertionError as ae:
        logger.error(f"Structural verification failed: {ae}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
