# app/services/langchain/query_rewriter_service.py
# ------------------------------------------------
# Service layer wrapper managing configuration retrieval and instantiation of the QueryRewriter.

import logging
from typing import Dict, Optional
from app.config import settings
from app.services.langchain.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class QueryRewriterService:
    """
    Service layer managing configuration and lifetime of the QueryRewriter.
    """
    def get_query_rewriter(
        self,
        enable_rewriter: Optional[bool] = None,
        rewrite_rules: Optional[Dict[str, str]] = None,
        synonym_map: Optional[Dict[str, str]] = None,
        abbreviation_map: Optional[Dict[str, str]] = None
    ) -> QueryRewriter:
        """
        Builds and returns a configured QueryRewriter instance.
        """
        active_enable = (
            enable_rewriter
            if enable_rewriter is not None
            else getattr(settings, "ENABLE_QUERY_REWRITER", True)
        )
        active_rewrite = (
            rewrite_rules
            if rewrite_rules is not None
            else getattr(settings, "QUERY_REWRITE_RULES", {})
        )
        active_synonyms = (
            synonym_map
            if synonym_map is not None
            else getattr(settings, "SYNONYM_MAP", {})
        )
        active_abbreviations = (
            abbreviation_map
            if abbreviation_map is not None
            else getattr(settings, "ABBREVIATION_MAP", {})
        )

        return QueryRewriter(
            enable_rewriter=active_enable,
            rewrite_rules=active_rewrite,
            synonym_map=active_synonyms,
            abbreviation_map=active_abbreviations
        )
