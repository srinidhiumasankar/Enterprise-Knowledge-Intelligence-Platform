# app/services/langchain/conversation_memory.py
# ---------------------------------------------
# Conversation Memory module storing user queries and expanding conversational follow-up questions.

import logging
import time
import re
from typing import Any, List, Dict, Tuple

logger = logging.getLogger(__name__)


def extract_subject(prev_query: str) -> str:
    """
    Extracts the core search subject from a previous query by stripping common instruction
    or question helper prefixes.
    """
    prev_clean = prev_query.strip().rstrip("?").rstrip(".")
    prefixes = [
        "what is the", "what is", "tell me about the", "tell me about",
        "explain the", "explain", "what are the", "what are", "how about",
        "show me the", "show me", "give me the", "give me"
    ]
    prev_lower = prev_clean.lower()
    for prefix in prefixes:
        if prev_lower.startswith(prefix):
            return prev_clean[len(prefix):].strip()
    return prev_clean


class ConversationMemory:
    """
    Represents conversation memory for a single session, tracking past queries and generating
    context-aware queries for conversational follow-ups.
    """
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def add_message(self, user_query: str, rewritten_query: str):
        """
        Saves a query transaction in history and limits history to the last configured turns.
        """
        # Store query transaction with timestamp
        self.history.append({
            "user_query": user_query,
            "rewritten_query": rewritten_query,
            "timestamp": time.time()
        })
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        logger.info(f"Added query to memory. Current history size: {len(self.history)}")

    def build_context_aware_query(self, current_query: str) -> str:
        """
        Analyzes the current query and history to determine if it is a follow-up.
        If it is, builds and returns a context-aware query combining current query and history.
        """
        if not self.history:
            return current_query

        # Normalize and tokenize
        normalized = " ".join(current_query.strip().split())
        query_lower = normalized.lower()
        words = query_lower.split()

        # Follow-up triggers check
        trigger_phrases = ["how many", "after that"]
        trigger_words = {
            "it", "that", "those", "they", "this", "its", "more", "again",
            "continue", "next", "then"
        }
        
        # Recognize short prepositional fragments as follow-ups (e.g. "After 2022?")
        prepositions = {"after", "before", "since", "until", "in", "during"}
        first_word = words[0].rstrip("?").rstrip(".") if words else ""
        is_preposition_fragment = first_word in prepositions and len(words) <= 3

        has_trigger = (
            any(phrase in query_lower for phrase in trigger_phrases) or
            any(w in trigger_words for w in words) or
            is_preposition_fragment
        )

        if not has_trigger:
            # Query is complete, return it unmodified
            logger.info("Current query is classified as complete. No history resolution applied.")
            return current_query

        # Extract context subject from last message in history
        # We prefer using the rewritten_query as it contains already expanded names/terms
        last_turn = self.history[-1]
        previous_text = last_turn["rewritten_query"] or last_turn["user_query"]
        subject = extract_subject(previous_text)
        
        logger.info(f"Follow-up query detected. Extracted subject from context: '{subject}'")

        # 1. 'How many days' pattern -> 'How many {subject} days are employees allowed?'
        if "how many days" in query_lower:
            resolved = f"How many {subject} days are employees allowed?"
            logger.info(f"Resolved follow-up using 'how many days' rule: '{resolved}'")
            return resolved

        # 2. Relative temporal/logical prepositions -> '{subject} {current_query}'
        prepositions = ["after", "before", "since", "until", "in", "during"]
        first_word = words[0].rstrip("?").rstrip(".") if words else ""
        if first_word in prepositions:
            # Strip question mark from current query and combine
            current_clean = normalized.rstrip("?")
            resolved = f"{subject} {current_clean}"
            logger.info(f"Resolved follow-up using preposition rule: '{resolved}'")
            return resolved

        # 3. Pronoun-noun patterns (e.g. 'its applications') -> '{noun} of {subject}'
        pronoun_match = re.search(r'\bits\s+(\w+)\b', query_lower)
        if pronoun_match:
            noun = pronoun_match.group(1)
            resolved = f"{noun.capitalize()} of {subject}"
            logger.info(f"Resolved follow-up using pronoun-noun rule: '{resolved}'")
            return resolved

        # 4. General fallback combining subject and follow-up
        current_clean = normalized.rstrip("?")
        resolved = f"{subject} {current_clean}"
        logger.info(f"Resolved follow-up using general combination: '{resolved}'")
        return resolved
