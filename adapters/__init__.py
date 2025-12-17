"""Adapters layer - Concrete implementations of ports."""
from adapters.ollama_adapter import OllamaAdapter
from adapters.fallback_llm_adapter import FallbackLLMAdapter
from adapters.openai_compatible_adapter import OpenAICompatibleAdapter
from adapters.tavily_adapter import TavilySearchAdapter
from adapters.exa_adapter import ExaSearchAdapter
from adapters.local_storage import LocalStorageAdapter
from adapters.mock_adapters import MockLLMAdapter, MockSearchAdapter, MockStorageAdapter

__all__ = [
    "OllamaAdapter",
    "FallbackLLMAdapter",
    "OpenAICompatibleAdapter",
    "TavilySearchAdapter",
    "ExaSearchAdapter",
    "LocalStorageAdapter",
    "MockLLMAdapter",
    "MockSearchAdapter",
    "MockStorageAdapter",
]
