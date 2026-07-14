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
        conversation_history: Optional[str] = None
    ) -> str:
        """
        Build a context prompt combining the user question, conversation history, and retrieved chunks.

        Args:
            question: The user's query string.
            chunks: A list of retrieval result dictionaries, each containing 'text' and 'metadata'.
            conversation_history: A text block representing formatted chat messages.

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

        # Compile segments
        prompt_segments = [
            "You are a helpful, precise Enterprise Knowledge Intelligence Assistant. "
            "Your goal is to answer the user's question accurately using the provided context blocks.\n\n"
            "=== STRICT CONSTRAINTS ===\n"
            "1. Answer the question relying ONLY on the facts directly mentioned in the provided context blocks. Every single sentence in your generated answer MUST originate from the retrieved document chunks. Do NOT use or supplement with any outside knowledge, general background knowledge, or external facts. Rely strictly on the context.\n"
            "2. Explicitly forbid and avoid any form of hallucination, invented facts, assumptions, extrapolation, or speculation. If the answer is not explicitly stated in the context, you must state exactly: 'I cannot determine the answer from the uploaded documents.'\n"
            "3. If relevant document context exists in the context blocks, you MUST answer the question using that context. Do NOT state that the documents are insufficient or refuse to answer when valid context exists.\n"
            "4. Only say that the uploaded documents contain insufficient information when the retrieved context genuinely contains no relevant facts to answer the question.\n"
            "5. If multiple context chunks are retrieved, merge them naturally and coherently into a single answer without repeating information or duplicating content.\n"
            "6. Add clear document citations to your answer whenever possible. Cite the source file name for every claim or fact using the format '(Source: <filename>)' (for example: (Source: ml_notes.txt) or (Source: architecture_specs.pdf)).\n"
            "7. Keep the response concise but complete.\n"
            "8. Preserve all technical terms, exact numbers, and nomenclature as they appear in the source.\n"
        ]

        if conversation_history:
            prompt_segments.append(
                "=== Conversation History ===\n"
                "------------------\n"
                f"{conversation_history}\n"
            )

        prompt_segments.append(
            "=== Retrieved Knowledge ===\n"
            "-------------------\n"
            f"{context_str}\n"
        )

        prompt_segments.append(
            "=== Current Question ===\n"
            "----------------\n"
            f"{question}\n"
        )

        prompt_segments.append("=== ANSWER ===")

        prompt = "\n".join(prompt_segments)
        logger.info("=== FINAL PROMPT SENT TO LLM ===")
        logger.info(prompt)
        logger.info("=================================")
        logger.info(f"Final prompt contains retrieved context: {bool(chunks)}")
        logger.debug(f"Generated prompt length: {len(prompt)} characters.")
        return prompt
