# app/services/citation_service.py
# --------------------------------
# Citation Service for managing document metadata and source attribution.

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CitationService:
    """
    Service for parsing, validating, and formatting document citations and source attribution.
    """

    def __init__(self):
        logger.info("Initializing Citation Service...")

    def format_citations(self, sources: List[Dict[str, Any]]) -> str:
        """
        Format a list of metadata source dicts into standard citation text.

        Parameters:
            sources (List[Dict[str, Any]]): Source dictionaries containing 'filename' and 'chunk_index'.

        Returns:
            str: Standard formatted citation string.
        """
        if not sources:
            return ""

        lines = ["Sources:"]
        seen = set()
        for src in sources:
            filename = src.get("filename", "Unknown Document")
            chunk_index = src.get("chunk_index")
            if chunk_index is None:
                chunk_index = src.get("chunk_id", "N/A")
            citation_key = (filename, chunk_index)
            if citation_key not in seen:
                seen.add(citation_key)
                lines.append(f"- {filename} (Chunk {chunk_index})")
        
        return "\n".join(lines)


def get_citation_service() -> CitationService:
    """
    FastAPI dependency provider to retrieve a CitationService instance.

    Returns:
        CitationService: The CitationService instance.
    """
    return CitationService()
