# tests/verify_phase_6_2.py
# -------------------------
# Automated verification script for Phase 6.2: Embedding Generation.

import logging
import sys
import os

# Add parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings.embedding_service import EmbeddingService
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_6_2")


def run_verification():
    logger.info("=================================================================")
    logger.info("STARTING PHASE 6.2 EMBEDDING SERVICE FUNCTIONAL VERIFICATION")
    logger.info("=================================================================")

    embedding_service = EmbeddingService()

    # 1. Verify health check on uninitialized state
    logger.info("Step 1: Checking initial health status...")
    initial_health = embedding_service.health_check()
    logger.info(f"Initial Health: {initial_health}")
    assert initial_health["status"] == "uninitialized", "Expected status to be uninitialized initially"
    assert not initial_health["model_loaded"], "Expected model_loaded to be False initially"

    # 2. Lazy loading and Singleton Check
    logger.info("\nStep 2: Testing lazy loading & model singleton...")
    model_1 = EmbeddingService.get_model()
    model_2 = EmbeddingService.get_model()
    assert model_1 is model_2, "Singleton violation: loaded model instances are different!"
    logger.info("✓ Lazy loading and singleton pattern verified successfully.")

    # 3. Check health status after loading
    logger.info("\nStep 3: Checking health status after model initialization...")
    loaded_health = embedding_service.health_check()
    logger.info(f"Loaded Health: {loaded_health}")
    assert loaded_health["status"] == "healthy", "Expected status to be healthy after model loading"
    assert loaded_health["model_loaded"], "Expected model_loaded to be True after initialization"
    logger.info("✓ Health check report verified.")

    # 4. Generate Single Embedding & Validation
    logger.info("\nStep 4: Testing single text embedding and inputs validation...")
    
    # Validation checks
    try:
        embedding_service.generate_embedding("")
        assert False, "Empty string failed to trigger ValueError!"
    except ValueError as ve:
        logger.info(f"✓ Correctly rejected empty string: {ve}")

    try:
        embedding_service.generate_embedding("   ")
        assert False, "Whitespace-only string failed to trigger ValueError!"
    except ValueError as ve:
        logger.info(f"✓ Correctly rejected whitespace-only string: {ve}")

    # Functional check
    text = "SentenceTransformers makes generating embeddings easy and fast."
    vector = embedding_service.generate_embedding(text)
    assert isinstance(vector, list), "Embedding output must be a Python list!"
    assert all(isinstance(val, float) for val in vector), "Embedding values must be floats!"
    assert len(vector) == settings.EMBEDDING_DIMENSION, f"Expected dimension size {settings.EMBEDDING_DIMENSION}, got {len(vector)}"
    logger.info(f"✓ Single embedding generated successfully. Vector length: {len(vector)}")

    # 5. Generate Batch Embeddings & Validation
    logger.info("\nStep 5: Testing batch embedding generation...")
    texts = [
        "First document text chunk for testing.",
        "Second text chunk that provides context.",
        "Third distinct statement."
    ]
    batch_vectors = embedding_service.generate_batch_embeddings(texts)
    assert len(batch_vectors) == len(texts), f"Batch length mismatch! Expected {len(texts)}, got {len(batch_vectors)}"
    for idx, vec in enumerate(batch_vectors):
        assert len(vec) == settings.EMBEDDING_DIMENSION, f"Vector at index {idx} size mismatch!"
    logger.info(f"✓ Batch of {len(texts)} embeddings generated successfully.")

    # 6. Generate Query Embedding
    logger.info("\nStep 6: Testing query embedding generation...")
    query = "How to search using embeddings?"
    query_vector = embedding_service.generate_query_embedding(query)
    assert len(query_vector) == settings.EMBEDDING_DIMENSION, "Query embedding dimension mismatch!"
    logger.info(f"✓ Query embedding generated successfully. Vector length: {len(query_vector)}")

    logger.info("\n=================================================================")
    logger.info("ALL PHASE 6.2 EMBEDDING SERVICE FUNCTIONAL TESTS PASSED!")
    logger.info("=================================================================")


if __name__ == "__main__":
    try:
        run_verification()
    except AssertionError as ae:
        logger.error(f"Verification Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
