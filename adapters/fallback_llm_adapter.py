"""Fallback LLM adapter - tries multiple providers/models in order."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
import time
from typing import TypeVar, Type

import httpx
from pydantic import BaseModel

from domain.exceptions import AdapterError
from ports.llm import LLMPort

T = TypeVar("T", bound=BaseModel)


class FallbackLLMAdapter(LLMPort):
    """
    Wraps multiple LLM adapters and falls back on transient failures.

    Sticky behavior:
    - Picks a starting adapter (via `start_index`)
    - Uses that adapter for the whole run
    - Switches to the next adapter only when a call fails in a fallback-eligible way
    """

    def __init__(
        self,
        adapters: list[LLMPort],
        start_index: int = 0,
        cooldown_seconds_default: float = 15.0,
    ):
        if not adapters:
            raise ValueError("FallbackLLMAdapter requires at least one adapter")
        self._adapters = adapters
        self._preferred_index = start_index % len(adapters)
        self._cooldown_seconds_default = float(cooldown_seconds_default)
        self._cooldown_until_by_index: dict[int, float] = {}

    @property
    def model_name(self) -> str:
        return self._adapters[self._preferred_index].model_name

    @property
    def provider(self) -> str:
        return "fallback"

    @staticmethod
    def _is_transient_http_status(status_code: int | None) -> bool:
        if status_code is None:
            return False
        return status_code in {408, 425, 429, 500, 502, 503, 504, 529}

    @staticmethod
    def _looks_like_model_not_found(status_code: int | None, body_lower: str) -> bool:
        if status_code in {404, 410}:
            return True
        if status_code in {400, 422}:
            if "model" in body_lower and (
                "not found" in body_lower
                or "does not exist" in body_lower
                or "unknown model" in body_lower
                or "not supported" in body_lower
                or "unsupported model" in body_lower
            ):
                return True
        return False

    @classmethod
    def _should_fallback(cls, error: Exception) -> bool:
        if isinstance(error, AdapterError):
            underlying = getattr(error, "original_error", None)
            if isinstance(underlying, httpx.HTTPStatusError):
                status_code = underlying.response.status_code
                try:
                    body_lower = (underlying.response.text or "").lower()
                except Exception:
                    body_lower = ""
                return cls._is_transient_http_status(status_code) or cls._looks_like_model_not_found(status_code, body_lower)
            if isinstance(underlying, (httpx.TimeoutException, httpx.TransportError)):
                return True
            if isinstance(underlying, json.JSONDecodeError):
                return True
            # Fall back for other adapter errors, but not for obviously non-retriable ones.
            return True

        if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
            return True

        return False

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        if not isinstance(error, AdapterError):
            return None

        underlying = getattr(error, "original_error", None)
        if not isinstance(underlying, httpx.HTTPStatusError):
            return None

        response = underlying.response
        if response.status_code != 429:
            return None

        header = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if not header:
            return None

        try:
            seconds = float(header.strip())
        except Exception:
            return None

        if seconds <= 0:
            return None
        return seconds

    @staticmethod
    def _describe_error(error: Exception) -> str:
        if isinstance(error, AdapterError):
            underlying = getattr(error, "original_error", None)
            if isinstance(underlying, httpx.HTTPStatusError):
                return f"HTTP {underlying.response.status_code}"
            if underlying is not None:
                return underlying.__class__.__name__
        return error.__class__.__name__

    def _is_in_cooldown(self, index: int) -> bool:
        until = self._cooldown_until_by_index.get(index)
        if until is None:
            return False
        if time.monotonic() >= until:
            self._cooldown_until_by_index.pop(index, None)
            return False
        return True

    def _next_index(self, current_index: int, tried: set[int]) -> int | None:
        # Prefer adapters that are not in cooldown.
        for offset in range(1, len(self._adapters) + 1):
            idx = (current_index + offset) % len(self._adapters)
            if idx in tried:
                continue
            if not self._is_in_cooldown(idx):
                return idx

        # If everything is in cooldown, still progress to any untried adapter.
        for offset in range(1, len(self._adapters) + 1):
            idx = (current_index + offset) % len(self._adapters)
            if idx not in tried:
                return idx

        return None

    async def _with_fallback(self, fn: Callable[[LLMPort], Awaitable[T]]) -> T:
        last_error: Exception | None = None
        tried: set[int] = set()
        idx = self._preferred_index

        # If preferred is cooling down, pick the next available.
        if self._is_in_cooldown(idx):
            next_idx = self._next_index(idx, tried=set())
            if next_idx is not None:
                idx = next_idx

        for _ in range(len(self._adapters)):
            tried.add(idx)
            adapter = self._adapters[idx]
            try:
                result = await fn(adapter)
                if idx != self._preferred_index:
                    print(f"LLM fallback: using {adapter.provider}/{adapter.model_name}")
                self._preferred_index = idx
                return result
            except Exception as e:
                last_error = e
                if self._should_fallback(e) and len(tried) < len(self._adapters):
                    retry_after = self._retry_after_seconds(e)
                    if retry_after is not None:
                        self._cooldown_until_by_index[idx] = time.monotonic() + retry_after
                    elif isinstance(e, AdapterError):
                        underlying = getattr(e, "original_error", None)
                        if isinstance(underlying, httpx.HTTPStatusError) and self._is_transient_http_status(
                            underlying.response.status_code
                        ):
                            self._cooldown_until_by_index[idx] = time.monotonic() + self._cooldown_seconds_default

                    print(
                        f"LLM fallback: {adapter.provider}/{adapter.model_name} failed "
                        f"({self._describe_error(e)}); trying next model..."
                    )
                    next_idx = self._next_index(idx, tried)
                    if next_idx is None:
                        break
                    idx = next_idx
                    # Sticky: future calls start from the last attempted index.
                    self._preferred_index = idx
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("FallbackLLMAdapter: no adapters executed")

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        return await self._with_fallback(
            lambda a: a.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        return await self._with_fallback(
            lambda a: a.generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        )

    async def generate_list(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_items: int = 5,
    ) -> list[str]:
        return await self._with_fallback(
            lambda a: a.generate_list(prompt=prompt, system_prompt=system_prompt, max_items=max_items)
        )
