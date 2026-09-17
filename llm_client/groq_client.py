"""Groq client provider.

Builds an AsyncOpenAI client for Groq's OpenAI-compatible endpoint with retry
and concurrency limiting applied, plus a module-level ``call_groq`` coroutine
for callers that want a plain completion without constructing their own client.
"""

from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI

from . import config
from .utils import _retry_decorator

logger = __import__('logging').getLogger(__name__)


def get_groq_client(model: str) -> AsyncOpenAI:
    """Return an AsyncOpenAI client configured for Groq with retries and concurrency limit."""
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set in environment")

    api_base = getattr(config, "GROQ_API_ENDPOINT", "https://api.groq.com/openai/v1")
    _client = AsyncOpenAI(
        base_url=api_base,
        api_key=config.GROQ_API_KEY,
    )
    _semaphore = asyncio.Semaphore(config.GROQ_MAX_CONCURRENCY)

    original_create = _client.chat.completions.create

    @_retry_decorator()
    async def wrapped_create(**kwargs):
        async with _semaphore:
            return await original_create(**kwargs)

    _client.chat.completions.create = wrapped_create  # type: ignore[assignment]
    # Exposed on the client so the test suite can assert on it directly.
    _client._semaphore = _semaphore  # type: ignore[attr-defined]
    return _client


# Module-level client so call_groq() below and importers of _client/_semaphore
# have one ready without building their own.
_shim_client = get_groq_client("placeholder-model")
_client = _shim_client
_semaphore = _shim_client._semaphore  # type: ignore[attr-defined]


@_retry_decorator()
async def call_groq(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    tools: list[dict] | None = None,
) -> Any:
    """Send a chat completion to Groq and return the raw response."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools is not None:
        kwargs["tools"] = tools

    return await _client.chat.completions.create(**kwargs)


__all__ = ["get_groq_client", "call_groq", "_client", "_semaphore"]