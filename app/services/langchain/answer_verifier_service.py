# app/services/langchain/answer_verifier_service.py
# ------------------------------------------------
# Service layer wrapper managing configuration retrieval and instantiation of the AnswerVerifier.

import logging
from typing import Optional
from app.config import settings
from app.services.langchain.answer_verifier import AnswerVerifier

logger = logging.getLogger(__name__)


class AnswerVerifierService:
    """
    Service layer managing configuration and lifetime of the AnswerVerifier.
    """
    def get_verifier(
        self,
        enable_verifier: Optional[bool] = None,
        grounding_threshold: Optional[float] = None,
        hallucination_threshold: Optional[float] = None,
        min_supported_keywords: Optional[int] = None
    ) -> AnswerVerifier:
        """
        Builds and returns a configured AnswerVerifier instance.
        """
        active_enable = (
            enable_verifier
            if enable_verifier is not None
            else getattr(settings, "ENABLE_ANSWER_VERIFIER", True)
        )
        active_grounding = (
            grounding_threshold
            if grounding_threshold is not None
            else getattr(settings, "GROUNDING_THRESHOLD", 70.0)
        )
        active_hallucination = (
            hallucination_threshold
            if hallucination_threshold is not None
            else getattr(settings, "HALLUCINATION_THRESHOLD", 30.0)
        )
        active_min_keywords = (
            min_supported_keywords
            if min_supported_keywords is not None
            else getattr(settings, "MIN_SUPPORTED_KEYWORDS", 3)
        )

        return AnswerVerifier(
            enable_verifier=active_enable,
            grounding_threshold=active_grounding,
            hallucination_threshold=active_hallucination,
            min_supported_keywords=active_min_keywords
        )
