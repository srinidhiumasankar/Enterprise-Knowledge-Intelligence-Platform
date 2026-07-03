# tests/verify_phase_8_5.py
# -------------------------
# Verification script for Phase 8.5 (Citation Service & RAG Attribution).

import os
import sys
import logging

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.langchain import get_embeddings, create_citation_rag_chain, get_citation_qa_prompt
from app.services.citation_service import CitationService
from app.embeddings.chroma_service import ChromaService

# Setup lightweight logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_5")


def verify_phase_8_5():
    logger.info("==========================================================")
    logger.info("STARTING PHASE 8.5 CITATION SERVICE & RAG ATTRIBUTION VERIFICATION")
    logger.info("==========================================================")

    # 1. Test CitationService Formatting
    logger.info("Step 1: Testing CitationService formatting...")
    try:
        citation_service = CitationService()
        test_sources = [
            {"filename": "leave_policy.pdf", "chunk_index": 3},
            {"filename": "leave_policy.pdf", "chunk_index": 3},  # Duplicate check
            {"filename": "hr_rules.docx", "chunk_index": 12}
        ]
        formatted = citation_service.format_citations(test_sources)
        logger.info(f"Formatted Citations:\n{formatted}")
        
        assert "Sources:" in formatted, "Header 'Sources:' missing"
        assert "- leave_policy.pdf (Chunk 3)" in formatted, "leave_policy.pdf chunk 3 missing"
        assert "- hr_rules.docx (Chunk 12)" in formatted, "hr_rules.docx chunk 12 missing"
        # Deduplication check: count of leave_policy should be 1
        assert formatted.count("leave_policy.pdf") == 1, "Duplicate citations not deduplicated!"
        logger.info("✓ CitationService formatted output verified successfully.")
    except Exception as e:
        logger.error(f"❌ CitationService test failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Test No Hallucination Prompt Constraints
    logger.info("Step 2: Testing prompt template constraint existence...")
    try:
        prompt_template = get_citation_qa_prompt()
        prompt_content = prompt_template.template
        
        # Check constraints
        assert "Never hallucinate" in prompt_content, "Hallucination constraints missing"
        assert "I couldn't find relevant information in the uploaded documents" in prompt_content, "Missing fallback instruction in prompt template"
        logger.info("✓ Prompt template constraints verified successfully.")
    except Exception as e:
        logger.error(f"❌ Prompt template constraints test failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Setup Test Data in ChromaDB
    logger.info("Step 3: Setup test data in ChromaDB...")
    test_doc_id_1 = 850001
    test_doc_id_2 = 850002
    test_owner_id = 999
    
    doc_1_chunks = ["An annual holiday allowance of 25 leave days is granted to all permanent workers."]
    doc_2_chunks = ["Employees are eligible for standard medical insurance from day one of employment."]
    
    try:
        chroma_service = ChromaService()
        embeddings = get_embeddings()

        # Delete any pre-existing test vectors
        chroma_service.collection.delete(where={"document_id": test_doc_id_1})
        chroma_service.collection.delete(where={"document_id": test_doc_id_2})

        # Insert Doc 1
        vectors_1 = embeddings.embed_documents(doc_1_chunks)
        chroma_service.add_documents(
            ids=[f"chunk-85-{test_doc_id_1}-{i}" for i in range(len(doc_1_chunks))],
            embeddings=vectors_1,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id_1, "filename": "leave_allowance.pdf", "chunk_index": 7}
                for i in range(len(doc_1_chunks))
            ],
            documents=doc_1_chunks
        )

        # Insert Doc 2
        vectors_2 = embeddings.embed_documents(doc_2_chunks)
        chroma_service.add_documents(
            ids=[f"chunk-85-{test_doc_id_2}-{i}" for i in range(len(doc_2_chunks))],
            embeddings=vectors_2,
            metadatas=[
                {"owner_id": test_owner_id, "document_id": test_doc_id_2, "filename": "medical_rules.docx", "chunk_index": 14}
                for i in range(len(doc_2_chunks))
            ],
            documents=doc_2_chunks
        )
        logger.info("✓ Test data successfully inserted into ChromaDB.")
    except Exception as e:
        logger.error(f"❌ Failed to set up ChromaDB test data: {e}", exc_info=True)
        sys.exit(1)

    # 4. Test Upgraded Citation RAG Chain
    logger.info("Step 4: Verifying Upgraded Citation RAG Chain...")
    try:
        rag_chain = create_citation_rag_chain(owner_id=test_owner_id, top_k=2)
        question = "How many holiday allowance leave days do workers get?"
        answer = rag_chain.run(question)
        
        logger.info(f"Query: '{question}'")
        logger.info(f"Answer:\n{answer}")
        
        # Verify filenames and chunk indices are preserved in sources list
        assert "25" in answer, "Answer is missing holiday info"
        assert "leave_allowance.pdf" in answer, "Source filename missing in response"
        assert "Chunk 7" in answer or "7" in answer, "Source chunk index missing or wrong"
        logger.info("✓ Answer attribution and metadata preservation verified successfully.")
    except Exception as e:
        logger.error(f"❌ Citation RAG Chain execution failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Test Fallback Execution
    logger.info("Step 5: Verifying fallback execution when no relevant data is retrieved...")
    try:
        non_existent_query = "What is the policy for company cars?"
        answer_fallback = rag_chain.run(non_existent_query)
        logger.info(f"Query: '{non_existent_query}'")
        logger.info(f"Answer: '{answer_fallback}'")
        
        assert answer_fallback == "I couldn't find relevant information in the uploaded documents.", (
            f"Expected fallback message, got: '{answer_fallback}'"
        )
        logger.info("✓ Fallback behavior verified successfully.")
    except Exception as e:
        logger.error(f"❌ Fallback behavior verification failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Cleanup
    logger.info("Step 6: Cleaning up test data from ChromaDB...")
    try:
        chroma_service.collection.delete(where={"document_id": test_doc_id_1})
        chroma_service.collection.delete(where={"document_id": test_doc_id_2})
        logger.info("✓ Test data cleaned up successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean up test data: {e}")

    logger.info("==========================================================")
    logger.info("PASS - PHASE 8.5 LANGCHAIN RAG CHAIN WITH CITATIONS VERIFIED SUCCESSFULLY")
    logger.info("==========================================================")
    print("PASS - PHASE 8.5 LANGCHAIN RAG CHAIN WITH CITATIONS VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_5()
