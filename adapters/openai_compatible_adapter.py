"""OpenAI-compatible LLM adapter (works with xAI/Grok, OpenAI, and similar APIs)."""
import json
from typing import Any, TypeVar, Type

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from ports.llm import LLMPort
from domain.exceptions import AdapterError

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleAdapter(LLMPort):
    """
    Adapter for OpenAI-compatible Chat Completions APIs.

    This is intentionally lightweight: it relies on simple prompting + JSON parsing
    for structured outputs so it can work across multiple providers.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.x.ai/v1",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        provider_name: str = "api",
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = self._normalize_base_url(base_url)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider_name = provider_name

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            return "https://api.x.ai/v1"
        if base_url.endswith("/v1"):
            return base_url
        return f"{base_url}/v1"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(url, headers=self._headers(), json=payload)

            # Some providers reject `response_format`. Retry once without it.
            if response_format and response.status_code in (400, 404, 422):
                print(f"  -> Provider rejected response_format (status {response.status_code}), retrying without...")
                payload.pop("response_format", None)
                response = await client.post(url, headers=self._headers(), json=payload)

            if response.status_code != 200:
                print(f"  -> LLM API Error: {response.status_code} - {response.text[:200]}")

            response.raise_for_status()
            data = response.json()

            try:
                choice0 = (data.get("choices") or [{}])[0]
                message = choice0.get("message") or {}
                content = message.get("content")
                if content is None:
                    content = choice0.get("text", "")
                return (content or "").strip()
            except Exception as e:
                raise AdapterError("OpenAICompatibleAdapter", "parse_response", e)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = (text or "").strip()
        if not text.startswith("```"):
            return text

        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        return text.strip()

    @classmethod
    def _extract_first_json_object(cls, text: str) -> str:
        text = cls._strip_code_fences(text)
        text = text.strip()

        # Fast path: whole string is JSON
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model response")

        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        raise ValueError("Unbalanced JSON object in model response")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            temp = temperature if temperature is not None else self._temperature
            return await self._chat_completion(messages=messages, temperature=temp)
        except Exception as e:
            raise AdapterError("OpenAICompatibleAdapter", "generate", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        try:
            schema_json = schema.model_json_schema()
            structured_prompt = f"""{prompt}

You MUST respond with a valid JSON object that matches this schema:
{json.dumps(schema_json, indent=2)}

Respond ONLY with the JSON object, no other text, no markdown."""

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds only in valid JSON.",
                }
            )
            messages.append({"role": "user", "content": structured_prompt})

            temp = temperature if temperature is not None else self._temperature
            text = await self._chat_completion(
                messages=messages,
                temperature=temp,
                response_format={"type": "json_object"},
            )

            json_text = self._extract_first_json_object(text)
            parsed = json.loads(json_text)
            return schema(**parsed)
        except json.JSONDecodeError as e:
            raise AdapterError("OpenAICompatibleAdapter", "generate_structured (JSON parse)", e)
        except Exception as e:
            raise AdapterError("OpenAICompatibleAdapter", "generate_structured", e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
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

            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "system", "content": "Respond only in valid JSON."})
            messages.append({"role": "user", "content": list_prompt})

            text = await self._chat_completion(
                messages=messages,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )

            json_text = self._extract_first_json_object(text)
            parsed = json.loads(json_text)
            items = parsed.get("items", [])
            return [str(item) for item in items][:max_items]
        except Exception as e:
            raise AdapterError("OpenAICompatibleAdapter", "generate_list", e)

    def get_token_count(self, text: str) -> int:
        return len(text) // 4

