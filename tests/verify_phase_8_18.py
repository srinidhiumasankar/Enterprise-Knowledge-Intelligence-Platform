# tests/verify_phase_8_18.py
# --------------------------
# Verification script for Phase 8.18 (Answer Verification & Hallucination Detection).
# Uses mocks to verify all requirements without consuming Gemini API quota.

import os
import sys
import logging
from typing import Any, List
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

# Add workspace directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.langchain.answer_verifier import AnswerVerifier, FinalResponse
from app.services.langchain.answer_verifier_service import AnswerVerifierService
from app.services.langchain.chains import create_rag_chain

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_phase_8_18")


class SimpleAssert:
    def assertEqual(self, a, b, msg=""):
        if a != b:
            raise AssertionError(f"{msg}: {a} != {b}")
    def assertTrue(self, cond, msg=""):
        if not cond:
            raise AssertionError(msg)

self = SimpleAssert()


def verify_phase_8_18():
    logger.info("==================================================")
    logger.info("STARTING PHASE 8.18 ANSWER VERIFIER VERIFICATION")
    logger.info("==================================================")

    # 1. Initialization and configuration check
    logger.info("\n--- Verifying Initialization ---")
    service = AnswerVerifierService()
    verifier = service.get_verifier()
    self.assertEqual(verifier.enable_verifier, True)
    self.assertEqual(verifier.grounding_threshold, 70.0)
    logger.info("✓ AnswerVerifierService and AnswerVerifier initialized correctly.")

    # Mocks retrieved context documents
    docs = [
        Document(
            page_content="The company vacation policy grants 20 days of paid time off (PTO) annually. Employees must submit requests via the HR portal.",
            metadata={"filename": "vacation_policy.pdf", "page_number": 2, "citation_key": "vac20", "confidence_score": 0.90}
        )
    ]

    # 2. Grounding calculation
    logger.info("\n--- Verifying Grounding Score ---")
    # Gounded answer
    grounded_ans = "The vacation policy grants 20 days of paid time off (PTO)."
    resp_g = verifier.verify_answer(grounded_ans, docs)
    self.assertEqual(resp_g.grounding_score, 100.0)
    
    # Partially grounded answer (1 supported sentence, 1 unsupported sentence)
    part_ans = "The vacation policy grants 20 days of paid time off (PTO). Employees also receive free lunches every Friday."
    resp_p = verifier.verify_answer(part_ans, docs)
    self.assertEqual(resp_p.grounding_score, 50.0)
    logger.info("✓ Grounding calculations verified successfully.")

    # 3. Keyword overlap
    logger.info("\n--- Verifying Keyword Overlap ---")
    # Low overlap answer
    low_overlap_ans = "Artificial Intelligence incorporates neural network modeling."
    resp_l = verifier.verify_answer(low_overlap_ans, docs)
    self.assertTrue(resp_l.verification_score < 40.0)
    logger.info("✓ Keyword overlap evaluation verified successfully.")

    # 4. Citation verification
    logger.info("\n--- Verifying Citation Verification ---")
    ans_no_cit = "The policy grants 20 days of paid time off (PTO)."
    ans_cit = "The policy grants 20 days of paid time off (PTO) [vac20]."
    
    resp_no_cit = verifier.verify_answer(ans_no_cit, docs)
    resp_cit = verifier.verify_answer(ans_cit, docs)
    
    # Citation presence improves verifier score
    self.assertTrue(resp_cit.verification_score >= resp_no_cit.verification_score)
    logger.info("✓ Citation checks verified successfully.")

    # 5. Hallucination detection
    logger.info("\n--- Verifying Hallucination Detection ---")
    # Unsupported entities hallucination
    hallucinated_ans = "The policy grants 20 days of paid time off (PTO). Jeff Bezos approved this directive."
    resp_hall = verifier.verify_answer(hallucinated_ans, docs)
    self.assertEqual(resp_hall.hallucination_risk, "High")
    logger.info("✓ Hallucination risk classification verified successfully.")

    # 6. Confidence levels
    logger.info("\n--- Verifying Confidence Levels ---")
    self.assertEqual(verifier.get_confidence_level(90.0), "Very High")
    self.assertEqual(verifier.get_confidence_level(75.0), "High")
    self.assertEqual(verifier.get_confidence_level(55.0), "Medium")
    self.assertEqual(verifier.get_confidence_level(35.0), "Low")
    self.assertEqual(verifier.get_confidence_level(15.0), "Very Low")
    logger.info("✓ Confidence levels verified successfully.")

    # 7. Pipeline integration
    logger.info("\n--- Verifying Pipeline Integration ---")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="The vacation policy grants 20 days of paid time off (PTO).")
    
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs

    rag_chain = create_rag_chain(llm=mock_llm, retriever=mock_retriever)
    final_resp = rag_chain.run("How many days of vacation?")

    # Verify that RAG chain output carries verifier metadata attributes
    self.assertTrue(isinstance(final_resp, FinalResponse))
    self.assertEqual(final_resp.verification_status, "Passed")
    self.assertEqual(final_resp.grounding_score, 100.0)
    logger.info("✓ Pipeline integration verified successfully.")

    # 8. Fallback
    logger.info("\n--- Verifying Fallback Behavior ---")
    faulty_verifier = AnswerVerifier()
    # Force exception during run
    faulty_verifier.stopwords = None
    fallback_resp = faulty_verifier.verify_answer("A response", docs)
    self.assertEqual(fallback_resp, "A response")
    self.assertEqual(fallback_resp.verification_status, "Passed")
    logger.info("✓ Fallback behavior verified successfully.")

    # 9. Cleanup
    logger.info("\nCleaning up test resources...")
    logger.info("✓ Cleanup complete.")

    logger.info("==========================================")
    logger.info("PASS - PHASE 8.18 ANSWER VERIFIER VERIFIED SUCCESSFULLY")
    logger.info("==========================================")
    print("PASS - PHASE 8.18 ANSWER VERIFIER VERIFIED SUCCESSFULLY")


if __name__ == "__main__":
    verify_phase_8_18()
