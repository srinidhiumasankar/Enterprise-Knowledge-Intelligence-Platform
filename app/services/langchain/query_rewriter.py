# app/services/langchain/query_rewriter.py
# ----------------------------------------
# Core Query Rewriter and Expansion module to normalize and enrich user search queries before retrieval.

import logging
import time
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def expand_tokens(text: str, synonym_map: Dict[str, str], abbreviation_map: Dict[str, str]) -> str:
    """
    Splits text into tokens, strips attached punctuation, matches against case-insensitive
    synonym and abbreviation dictionaries, prepends 'year ' to numeric years, and reconstructs the text.
    """
    words = text.split()
    expanded_words = []
    
    for word in words:
        # Match prefix punctuation, core word, and suffix punctuation
        match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', word)
        if match:
            prefix, core, suffix = match.groups()
            core_lower = core.lower()
            
            # Check abbreviation first (case-insensitive key lookup)
            if core_lower in abbreviation_map:
                core = abbreviation_map[core_lower]
            # Check synonyms next (case-insensitive key lookup)
            elif core_lower in synonym_map:
                core = synonym_map[core_lower]
            # Prepend 'year ' to 4-digit years (e.g. 2022 -> year 2022)
            elif core.isdigit() and len(core) == 4 and (core.startswith("19") or core.startswith("20")):
                core = f"year {core}"
            
            expanded_words.append(f"{prefix}{core}{suffix}")
        else:
            expanded_words.append(word)
            
    return " ".join(expanded_words)


class QueryRewriter:
    """
    Provides rule-based query normalization, abbreviation expansion, synonym mapping,
    vague question rewriting, and failure fallback behaviors.
    """
    def __init__(
        self,
        enable_rewriter: bool = True,
        rewrite_rules: Dict[str, str] = None,
        synonym_map: Dict[str, str] = None,
        abbreviation_map: Dict[str, str] = None
    ):
        self.enable_rewriter = enable_rewriter
        
        # Standardize keys to lowercase for dictionary lookups
        self.rewrite_rules = {k.lower().strip(): v for k, v in (rewrite_rules or {}).items()}
        self.synonym_map = {k.lower().strip(): v for k, v in (synonym_map or {}).items()}
        self.abbreviation_map = {k.lower().strip(): v for k, v in (abbreviation_map or {}).items()}

    def rewrite(self, query: str) -> str:
        """
        Applies whitespace normalization, abbreviation/synonym expansion, direct rewrite rules,
        logs stats (latency & queries), and falls back to original query under failure.
        """
        start_time = time.time()
        
        # Normalize whitespace (fallback safety first)
        try:
            normalized = " ".join((query or "").strip().split())
        except Exception as e:
            logger.error(f"Query normalization failed: {e}", exc_info=True)
            return query

        if not self.enable_rewriter:
            logger.info(f"Query rewriting is disabled. Using normalized query: '{normalized}'")
            return normalized

        try:
            norm_lower = normalized.lower()
            
            # 1. Direct rewrite rules check (exact match on normalized query)
            if norm_lower in self.rewrite_rules:
                rewritten = self.rewrite_rules[norm_lower]
                latency = (time.time() - start_time) * 1000
                
                # Log rewrite operations
                logger.info(f"Original query: '{query}'")
                logger.info(f"Normalized query: '{normalized}'")
                logger.info(f"Expanded query: '{rewritten}'")
                logger.info(f"Rewritten query: '{rewritten}'")
                logger.info(f"Rewrite latency: {latency:.2f}ms")
                return rewritten

            # 2. Token-level synonym & abbreviation expansion
            rewritten = expand_tokens(normalized, self.synonym_map, self.abbreviation_map)
            latency = (time.time() - start_time) * 1000
            
            # Log rewrite operations
            logger.info(f"Original query: '{query}'")
            logger.info(f"Normalized query: '{normalized}'")
            logger.info(f"Expanded query: '{rewritten}'")
            logger.info(f"Rewritten query: '{rewritten}'")
            logger.info(f"Rewrite latency: {latency:.2f}ms")
            return rewritten

        except Exception as e:
            logger.warning(
                f"Query rewriter encountered an error: {e}. Falling back to original query.",
                exc_info=True
            )
            return query
