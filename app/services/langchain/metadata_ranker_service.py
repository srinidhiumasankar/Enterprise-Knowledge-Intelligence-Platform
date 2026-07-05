# app/services/langchain/metadata_ranker_service.py
# ------------------------------------------------
# Service layer wrapper managing configuration retrieval and instantiation of the MetadataRanker.

import logging
from typing import Dict, Optional
from app.config import settings
from app.services.langchain.metadata_ranker import MetadataRanker

logger = logging.getLogger(__name__)


class MetadataRankerService:
    """
    Service layer managing configuration and lifetime of the MetadataRanker.
    """
    def get_metadata_ranker(
        self,
        enable_ranker: Optional[bool] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> MetadataRanker:
        """
        Builds and returns a configured MetadataRanker instance.
        """
        active_enable = (
            enable_ranker
            if enable_ranker is not None
            else getattr(settings, "ENABLE_METADATA_RANKER", True)
        )
        
        # Build active weights by resolving settings configurations
        if weights is not None:
            active_weights = weights
        else:
            base_weights = getattr(settings, "METADATA_RANKING_WEIGHTS", {})
            active_weights = {
                "semantic": base_weights.get("semantic", 0.45),
                "rrf": base_weights.get("rrf", 0.20),
                "freshness": base_weights.get("freshness", getattr(settings, "FRESHNESS_WEIGHT", 0.10)),
                "importance": base_weights.get("importance", getattr(settings, "IMPORTANCE_WEIGHT", 0.10)),
                "type": base_weights.get("type", getattr(settings, "TYPE_WEIGHT", 0.05)),
                "citation": base_weights.get("citation", getattr(settings, "CITATION_WEIGHT", 0.05)),
                "completeness": base_weights.get("completeness", 0.05),
            }

        return MetadataRanker(
            enable_ranker=active_enable,
            weights=active_weights
        )
