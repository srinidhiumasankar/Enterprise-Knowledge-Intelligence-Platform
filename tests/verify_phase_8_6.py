# tests/verify_phase_8_6.py
# -------------------------
# Verification script for Phase 8.6 (Conversation Memory).

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.langchain import get_embeddings, create_conversational_rag_chain, get_conversation_qa_prompt
from app.services.memory_service import MemoryService
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_6")


def verify_phase_8_6():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.6 CONVERSATION MEMORY VERIFICATION")
    logger.info("==========================================================")

    # 1. Test Memory Creation & Session Isolation
    logger.info("Step 1: Testing MemoryService creation & session isolation...")
    try:
        memory_service = MemoryService()
        session_a = "session_alice_86"
        session_b = "session_bob_86"

        mem_a = memory_service.get_memory(session_a)
        mem_b = memory_service.get_memory(session_b)

        assert mem_a is not mem_b, "Session memories are shared, not isolated!"
        
        # Save context to Session A
        mem_a.save_context({"question": "Hello from Alice"}, {"answer": "Hi Alice!"})
        
        # Verify Session B is empty
        history_b = mem_b.load_memory_variables({}).get("chat_history", "")
        assert history_b == "", f"Session B contains Alice's history: {history_b}"
        
        history_a = mem_a.load_memory_variables({}).get("chat_history", "")
        assert "Alice" in history_a, "Alice's memory failed to record history context."
        
        logger.info("✓ Memory creation and session isolation verified successfully.")
    except Exception as e:
        logger.error(f"❌ Memory Service baseline checks failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Test Prompt Includes Memory Structure
    logger.info("Step 2: Checking prompt template memory structures...")
    try:
        prompt = get_conversation_qa_prompt()
        assert "chat_history" in prompt.input_variables, "chat_history input variable is missing"
        assert "=== PREVIOUS CONVERSATION ===" in prompt.template, "Prompt lacks PREVIOUS CONVERSATION boundary"
        logger.info("✓ Prompt template memory boundaries verified successfully.")
    except Exception as e:
        logger.error(f"❌ Prompt template verification failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Setup Test Data in ChromaDB
    logger.info("Step 3: Setup test data in ChromaDB...")
    test_doc_id = 860001
    test_owner_id = 999
    chunks = [
        "Company leave policy: Permanent employees receive 15 annual sick leaves.",
        "Sick leaves require a doctor note if they extend beyond 3 consecutive days."
    ]

    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id})

        # Insert Doc
        vectors = embeddings.embed_documents(chunks)
        chroma_service.add_documents(
            ids=[f"chunk-86-{test_doc_id}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id, "filename": "sick_leaves.pdf", "chunk_index": i}
                for i in range(len(chunks))
            ],
            documents=chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    # 4. Run Multi-turn conversational session
    logger.info("Step 4: Executing multi-turn conversation session...")
    try:
        session_id = "multi_turn_test_86"
        rag_chain = create_conversational_rag_chain(
            session_id=session_id,
            owner_id=test_owner_id,
            top_k=2,
            memory_service=memory_service
        )

        # Question 1: What is the leave policy?
        q1 = "What is the sick leave policy allowance?"
        ans1 = rag_chain.run(q1)
        logger.info(f"Q1: {q1}")
        logger.info(f"Ans1: {ans1}")
        assert "15" in ans1, "Answer did not retrieve the sick leaves allowance"

        # Verify Q1 and Ans1 are stored in history
        mem = memory_service.get_memory(session_id)
        history = mem.load_memory_variables({}).get("chat_history", "")
        logger.info(f"Memory after turn 1:\n{history}")
        assert "sick leave" in history.lower() or "15" in history, "History did not record turn 1 context."

        # Question 2: Do they require doctor notes? (Context dependent follow up)
        q2 = "Do they require any doctor notes?"
        ans2 = rag_chain.run(q2)
        logger.info(f"Q2: {q2}")
        logger.info(f"Ans2: {ans2}")
        assert "3" in ans2 or "consecutive" in ans2 or "doctor" in ans2, "Follow-up question failed to leverage turn history context"
        
        logger.info("✓ Multi-turn conversational flow succeeded and remembered previous turns.")
    except Exception as e:
        logger.error(f"❌ Multi-turn conversation execution failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Test clearing memory
    logger.info("Step 5: Testing clearing memory buffers...")
    try:
        memory_service.clear_memory(session_id)
        history_cleared = mem.load_memory_variables({}).get("chat_history", "")
        assert history_cleared == "", f"Memory failed to clear: {history_cleared}"
        logger.info("✓ Memory clearing verified successfully.")
    except Exception as e:
        logger.error(f"❌ Clearing memory verification failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Cleanup
    logger.info("Step 6: Cleaning up test data from ChromaDB...")
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.6 CONVERSATION MEMORY VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS - PHASE 8.6 CONVERSATION MEMORY VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_6()
