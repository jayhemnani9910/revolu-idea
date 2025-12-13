"""Abstract interface for Language Model interactions."""
from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMPort(ABC):
    """
    Port for Language Model interactions.
    Abstracts away provider-specific APIs (OpenAI, Anthropic, etc).

    This interface follows the Dependency Inversion Principle:
    high-level modules depend on this abstraction, not concrete implementations.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a plain text response.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text response
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """
        Generate a response that conforms to a Pydantic schema.
        Uses JSON mode or tool calling to ensure schema compliance.

        Args:
            prompt: The user prompt
            schema: Pydantic model class defining the response structure
            system_prompt: Optional system instructions
            temperature: Sampling temperature

        Returns:
            Instance of the schema class
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_list(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_items: int = 5,
    ) -> list[str]:
        """
        Generate a list of strings (e.g., search queries).

        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            max_items: Maximum number of items to generate

        Returns:
            List of generated strings
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        raise NotImplementedError

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for text.
        Default implementation uses rough heuristic.
        Adapters should override with accurate tokenization.
        """
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4
