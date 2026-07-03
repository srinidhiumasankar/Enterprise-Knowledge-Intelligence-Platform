# app/ai/prompt_builder.py
# ------------------------
# Component responsible for building context-rich, constraint-enforcing prompts for LLM.

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Constructs production-grade RAG prompts by formatting retrieved context
    and enforcing strict constraints to prevent hallucinations.
    """

    @staticmethod
    def build_prompt(
        question: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a context prompt combining the user question with the retrieved chunks.

        Args:
            question: The user's query string.
            chunks: A list of retrieval result dictionaries, each containing 'text' and 'metadata'.

        Returns:
            str: The constructed prompt for the LLM.
        """
        logger.info(f"Building prompt for question: '{question}' with {len(chunks) if chunks else 0} context chunks.")

        if not question or not question.strip():
            logger.warning("Empty question passed to PromptBuilder.")
            question = ""

        # Format retrieved context
        context_blocks = []
        if chunks:
            for idx, chunk in enumerate(chunks, 1):
                # Handle dictionary access safely
                text = ""
                metadata = {}
                
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content") or ""
                    metadata = chunk.get("metadata") or {}
                    doc_id = chunk.get("document_id")
                    chunk_id = chunk.get("chunk_id")
                else:
                    # Handle object structure
                    text = getattr(chunk, "text", "") or getattr(chunk, "content", "") or ""
                    metadata = getattr(chunk, "metadata", {}) or {}
                    doc_id = getattr(chunk, "document_id", None)
                    chunk_id = getattr(chunk, "chunk_id", None)

                # Extract metadata info
                filename = metadata.get("filename") or "Unknown Document"
                doc_id = doc_id or metadata.get("document_id") or "N/A"
                chunk_id = chunk_id or metadata.get("chunk_id") or "N/A"

                block = (
                    f"--- Source {idx} ---\n"
                    f"File: {filename}\n"
                    f"Document ID: {doc_id}\n"
                    f"Chunk ID: {chunk_id}\n"
                    f"Content: {text.strip()}\n"
                )
                context_blocks.append(block)

        # Assemble prompt template
        context_str = "\n".join(context_blocks) if context_blocks else "No relevant context available."

        prompt = (
            "You are a helpful, precise Enterprise Knowledge Intelligence Assistant. "
            "Your goal is to answer the user's question accurately using ONLY the provided context blocks below.\n\n"
            "=== STRICT CONSTRAINTS ===\n"
            "1. Answer the question relying ONLY on the facts directly mentioned in the provided context blocks.\n"
            "2. Do NOT extrapolate, assume, or integrate outside knowledge. If the answer is not explicitly present in the context, "
            "you MUST state exactly: 'I cannot determine the answer from the uploaded documents.'\n"
            "3. Keep the response concise but complete.\n"
            "4. Preserve all technical terms, exact numbers, and nomenclature as they appear in the source.\n\n"
            "=== PROVIDED CONTEXT ===\n"
            f"{context_str}\n\n"
            "=== USER QUESTION ===\n"
            f"{question}\n\n"
            "=== ANSWER ==="
        )

        logger.debug(f"Generated prompt length: {len(prompt)} characters.")
        return prompt
