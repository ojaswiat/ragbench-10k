"""Tests for the Groq client provider (backward‑compatible shim)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from groq import APIStatusError

from llm_client.groq_client import get_groq_client, call_groq, _client, _semaphore


class _FakeResponse:
    status_code = 429
    request = None


class _FakeResult:
    def __init__(self):
        self.choices = [{"message": {"content": "ok"}}]


@pytest.mark.asyncio
async def test_groq_client_retries_on_429_then_succeeds():
    success_result = _FakeResult()
    call_count = {"n": 0}

    async def flaky_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise APIStatusError(
                message="rate limited",
                response=_FakeResponse(),
                body={"error": {"message": "rate limited"}},
            )
        return success_result

    # Patch the class where it is used in this module. get_groq_client()
    # captures the constructed client's chat.completions.create as
    # original_create and wraps it with retry+semaphore, so the mock's
    # create method must be configured before get_groq_client() runs.
    with patch("llm_client.groq_client.AsyncOpenAI") as mock_ctor:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=flaky_create)
        mock_ctor.return_value = mock_client

        client = get_groq_client("test-model")
        result = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result.choices == success_result.choices
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_groq_client_respects_semaphore_bound():
    # Force creation of a client to initialise the module-level _semaphore
    _ = get_groq_client("dummy")
    assert _semaphore._value == 5


@pytest.mark.asyncio
async def test_groq_client_passes_tools_through_when_given():
    success_result = _FakeResult()
    captured = {}

    async def capture_create(**kwargs):
        captured.update(kwargs)
        return success_result

    with patch("llm_client.groq_client.AsyncOpenAI") as mock_ctor:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=capture_create)
        mock_ctor.return_value = mock_client

        client = get_groq_client("test-model")
        await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search_filing"}}],
        )

    assert "tools" in captured
    assert captured["tools"][0]["function"]["name"] == "search_filing"


@pytest.mark.asyncio
async def test_groq_client_omits_tools_when_not_given():
    success_result = _FakeResult()
    captured = {}

    async def capture_create(**kwargs):
        captured.update(kwargs)
        return success_result

    with patch("llm_client.groq_client.AsyncOpenAI") as mock_ctor:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=capture_create)
        mock_ctor.return_value = mock_client

        client = get_groq_client("test-model")
        await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert "tools" not in captured


@pytest.mark.asyncio
async def test_call_groq_backward_compatibility():
    """Exercises the module-level call_groq wrapper, not the client class."""
    success_result = _FakeResult()
    call_count = {"n": 0}

    async def flaky_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise APIStatusError(
                message="rate limited",
                response=_FakeResponse(),
                body={"error": {"message": "rate limited"}},
            )
        return success_result

    with patch("llm_client.groq_client._client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=flaky_create)
        result = await call_groq(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert result.choices == success_result.choices
    assert call_count["n"] == 3