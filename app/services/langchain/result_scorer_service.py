# app/services/langchain/result_scorer_service.py
# -----------------------------------------------
# Service layer wrapper managing configuration retrieval and instantiation of the ResultScorer.

import logging
from typing import Dict, Optional
from app.config import settings
from app.services.langchain.result_scorer import ResultScorer

logger = logging.getLogger(__name__)


class ResultScorerService:
    """
    Service layer managing configuration and lifetime of the ResultScorer.
    """
    def get_result_scorer(
        self,
        enable_scorer: Optional[bool] = None,
        weights: Optional[Dict[str, float]] = None,
        min_confidence_score: Optional[float] = None
    ) -> ResultScorer:
        """
        Builds and returns a configured ResultScorer instance.
        """
        active_enable = (
            enable_scorer
            if enable_scorer is not None
            else getattr(settings, "ENABLE_RESULT_SCORER", True)
        )
        active_weights = (
            weights
            if weights is not None
            else getattr(settings, "RESULT_SCORING_WEIGHTS", {})
        )
        active_min_confidence = (
            min_confidence_score
            if min_confidence_score is not None
            else getattr(settings, "MIN_CONFIDENCE_SCORE", 0.0)
        )

        return ResultScorer(
            enable_scorer=active_enable,
            weights=active_weights,
            min_confidence_score=active_min_confidence
        )
