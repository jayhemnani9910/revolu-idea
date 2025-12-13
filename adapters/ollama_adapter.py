"""Ollama LLM adapter implementation."""
import json
import httpx
from typing import TypeVar, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from ports.llm import LLMPort
from domain.exceptions import AdapterError

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter(LLMPort):
    """
    Adapter for Ollama local LLMs.
    """

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self._model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "ollama"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": prompt,
                        "system": system_prompt or "",
                        "stream": False,
                        "options": {
                            "temperature": temperature if temperature is not None else self._temperature,
                            "num_predict": self._max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")

        except Exception as e:
            raise AdapterError("OllamaAdapter", "generate", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> T:
        try:
            schema_json = schema.model_json_schema()

            structured_prompt = f"""{prompt}

You MUST respond with a valid JSON object that matches this schema:
{json.dumps(schema_json, indent=2)}

Respond ONLY with the JSON object, no other text, no markdown."""

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": structured_prompt,
                        "system": system_prompt or "You are a helpful assistant that responds only in valid JSON.",
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": temperature if temperature is not None else self._temperature,
                            "num_predict": self._max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()

                # Clean up response
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]

                parsed = json.loads(text.strip())
                return schema(**parsed)

        except json.JSONDecodeError as e:
            raise AdapterError("OllamaAdapter", "generate_structured (JSON parse)", e)
        except Exception as e:
            raise AdapterError("OllamaAdapter", "generate_structured", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_list(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_items: int = 5,
    ) -> list[str]:
        try:
            list_prompt = f"""{prompt}

Generate up to {max_items} items. Respond with a JSON object like: {{"items": ["item1", "item2", ...]}}
Respond ONLY with the JSON object."""

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_name,
                        "prompt": list_prompt,
                        "system": system_prompt or "Respond only in valid JSON.",
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": self._temperature,
                            "num_predict": self._max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()

                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]

                parsed = json.loads(text.strip())
                return parsed.get("items", [])[:max_items]

        except Exception as e:
            raise AdapterError("OllamaAdapter", "generate_list", e)

    def get_token_count(self, text: str) -> int:
        return len(text) // 4
