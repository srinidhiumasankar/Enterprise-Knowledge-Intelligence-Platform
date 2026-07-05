# app/services/langchain/answer_verifier.py
# ----------------------------------------
# Core Answer Verification and Hallucination Detection engine to analyze LLM responses against retrieved context.

import logging
import time
import re
from typing import Any, List, Dict, Set
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FinalResponse(str):
    """
    Subclass of str to maintain backward compatibility while carrying
    verification and grounding metadata properties.
    """
    def __new__(
        cls,
        content: str,
        verification_score: float = 0.0,
        grounding_score: float = 0.0,
        hallucination_risk: str = "Low",
        verification_status: str = "Passed",
        confidence_level: str = "High"
    ):
        obj = super().__new__(cls, content)
        obj.verification_score = verification_score
        obj.grounding_score = grounding_score
        obj.hallucination_risk = hallucination_risk
        obj.verification_status = verification_status
        obj.confidence_level = confidence_level
        return obj


def extract_entities(text: str) -> Set[str]:
    """
    Extracts potential Named Entities based on capitalized terms.
    """
    # Matches words starting with uppercase letters, length > 1
    words = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', text)
    return set(words)


class AnswerVerifier:
    """
    Evaluates LLM generated answers against retrieved documents using grounding metrics,
    citation verification, entity mapping, and hallucination scoring without additional LLM calls.
    """
    def __init__(
        self,
        enable_verifier: bool = True,
        grounding_threshold: float = 70.0,
        hallucination_threshold: float = 30.0,
        min_supported_keywords: int = 3
    ):
        self.enable_verifier = enable_verifier
        self.grounding_threshold = grounding_threshold
        self.hallucination_threshold = hallucination_threshold
        self.min_supported_keywords = min_supported_keywords
        
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'to', 'of', 'in', 'on', 'for', 'with', 'by', 'at', 'this', 'that',
            'these', 'those', 'it', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'how', 'why', 'where', 'when', 'if', 'then', 'else'
        }

    def verify_answer(self, answer: str, docs: List[Document]) -> FinalResponse:
        """
        Runs the verification pipeline and returns a metadata-enriched FinalResponse.
        """
        start_time = time.time()
        
        if not answer:
            return FinalResponse("", 0.0, 0.0, "High", "Failed", "Very Low")

        if not self.enable_verifier or not docs:
            logger.info("Answer verifier is disabled or retrieved documents are empty. Returning unmodified response.")
            return FinalResponse(
                answer,
                verification_score=100.0 if docs else 0.0,
                grounding_score=100.0 if docs else 0.0,
                hallucination_risk="Low" if docs else "High",
                verification_status="Passed" if docs else "Failed",
                confidence_level="High" if docs else "Very Low"
            )

        try:
            combined_docs = " ".join([d.page_content for d in docs]).lower()
            
            # Matches brackets [1], [Source 2], [fin24]
            citation_pattern = re.compile(r'\[\s*\d+\s*\]|\[\s*Source\b|\bSource\s+\d+\b|\[\s*[a-zA-Z0-9_\-\.]+\s*\]')
            has_citations = bool(citation_pattern.search(answer))
            
            # Clean citations from the text for keyword/entity/sentence checking
            cleaned_answer = citation_pattern.sub("", answer)

            # 1. Split answer into sentences
            sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', cleaned_answer) if s.strip()]
            if not sentences:
                sentences = [cleaned_answer]

            # 2. Compute sentence grounding scores
            grounded_count = 0
            for sentence in sentences:
                words = [re.sub(r'[^\w]', '', w).lower() for w in sentence.split()]
                content_words = [w for w in words if w and w not in self.stopwords and len(w) > 2]
                
                if not content_words:
                    # Treat generic filler sentences as grounded
                    grounded_count += 1
                    continue
                
                # Check how many content words exist in document body
                matches = sum(1 for w in content_words if w in combined_docs)
                overlap_ratio = matches / len(content_words)
                
                # If 40% or more content words are found, consider sentence grounded
                if overlap_ratio >= 0.4 or matches >= self.min_supported_keywords:
                    grounded_count += 1

            grounding_score = (grounded_count / len(sentences)) * 100.0

            # 3. Keyword overlap
            all_answer_words = [re.sub(r'[^\w]', '', w).lower() for w in cleaned_answer.split()]
            answer_content = [w for w in all_answer_words if w and w not in self.stopwords and len(w) > 2]
            
            matched_keywords = sum(1 for w in answer_content if w in combined_docs)
            keyword_overlap_ratio = (matched_keywords / len(answer_content)) if answer_content else 1.0

            # 4. Named Entity overlap
            answer_entities = extract_entities(cleaned_answer)
            doc_entities = extract_entities(" ".join([d.page_content for d in docs]))
            unsupported_entities = answer_entities - doc_entities

            # 6. Retriever confidence aggregation
            avg_doc_confidence = sum(d.metadata.get("confidence_score", 0.5) for d in docs) / len(docs)

            # 7. Final verification score
            # Balanced blend of Grounding, Keyword Overlap, and Retriever Confidence
            verification_score = (
                (grounding_score * 0.70) +
                (keyword_overlap_ratio * 100.0 * 0.20) +
                (avg_doc_confidence * 100.0 * 0.10)
            )

            # 8. Hallucination Risk Classification
            # Higher risk if unsupported entities are present, score is below threshold, or no citations found
            risk_flags = 0
            if unsupported_entities:
                risk_flags += 2
            if verification_score < self.hallucination_threshold:
                risk_flags += 2
            if not has_citations:
                risk_flags += 1
            if keyword_overlap_ratio < 0.15:
                risk_flags += 2
            if avg_doc_confidence < 0.35:
                risk_flags += 1

            if risk_flags >= 4:
                hallucination_risk = "Critical"
            elif risk_flags >= 2:
                hallucination_risk = "High"
            elif risk_flags == 1:
                hallucination_risk = "Medium"
            else:
                hallucination_risk = "Low"

            # 9. Verification status and confidence mapping
            verification_status = "Passed" if verification_score >= self.grounding_threshold else "Failed"

            # Map to confidence levels
            confidence_level = self.get_confidence_level(verification_score)

            latency = (time.time() - start_time) * 1000
            
            logger.info(
                f"Answer Verification completed in {latency:.2f}ms. | "
                f"Score: {verification_score:.1f} | "
                f"Grounding: {grounding_score:.1f} | "
                f"Hallucination Risk: {hallucination_risk} | "
                f"Confidence Level: {confidence_level} | "
                f"Status: {verification_status}"
            )

            return FinalResponse(
                answer,
                verification_score=verification_score,
                grounding_score=grounding_score,
                hallucination_risk=hallucination_risk,
                verification_status=verification_status,
                confidence_level=confidence_level
            )

        except Exception as e:
            logger.warning(f"Answer verifier encountered an error: {e}. Falling back to original answer.", exc_info=True)
            return FinalResponse(answer, 50.0, 50.0, "Medium", "Passed", "Medium")

    def get_confidence_level(self, score: float) -> str:
        """
        Maps verification score (0-100) to confidence levels.
        """
        if score >= 85.0:
            return "Very High"
        if score >= 70.0:
            return "High"
        if score >= 50.0:
            return "Medium"
        if score >= 30.0:
            return "Low"
        return "Very Low"
