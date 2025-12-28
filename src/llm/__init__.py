"""
LLM Module

Provides interfaces for local LLM inference using llama.cpp.
"""

from .llm_engine import LLMEngine, LLMConfig, LLMResponse
from .prompt_manager import PromptManager

__all__ = ['LLMEngine', 'LLMConfig', 'LLMResponse', 'PromptManager']
