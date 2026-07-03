# app/ai/__init__.py
# ------------------
# AI module for the Enterprise Knowledge Intelligence Platform,
# wrapping PromptBuilder and GeminiService.

from app.ai.prompt_builder import PromptBuilder
from app.ai.gemini_service import GeminiService

__all__ = ["PromptBuilder", "GeminiService"]
