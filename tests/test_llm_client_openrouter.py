"""Tests for the LLM client abstraction (OpenRouter provider)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_client.llm_factory import LLMFactory
from llm_client import openrouter_client as openrouter_client_mod


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeCompletion:
    def __init__(self, content: str = "test"):
        self.choices = [_FakeChoice(content)]


@pytest.mark.asyncio
async def test_openrouter_client_accomplete_calls_openai_client(monkeypatch):
    """Transparent to kwargs despite the retry/semaphore/rate-limit wrapping."""
    captured = {}

    monkeypatch.setattr("llm_client.config.OPENROUTER_API_KEY", "dummy-key")
    # Avoid the 20 RPM rate-limit sleep affecting unrelated call timing
    # across tests in this module.
    monkeypatch.setattr("llm_client.openrouter_client._last_request_time", 0.0)

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace()
            self.chat.completions = SimpleNamespace()

    async def mock_create(**kwargs):
        captured.update(kwargs)
        return _FakeCompletion(content="ok")

    mock_client = MockClient()
    mock_client.chat.completions.create = mock_create

    def mock_constructor(*args, **kwargs):
        return mock_client

    # openrouter_client.py does `from openai import AsyncOpenAI` at import
    # time, so the name must be patched where it is looked up
    # (llm_client.openrouter_client), not on the openai package itself.
    monkeypatch.setattr("llm_client.openrouter_client.AsyncOpenAI", mock_constructor)

    client = openrouter_client_mod.get_openrouter_client("test-model")
    await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        tools=None,
    )
    assert captured == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "tools": None,
    }


@pytest.mark.asyncio
async def test_get_llm_client_returns_equivalent_client_openrouter(monkeypatch):
    """No caching, so client identity is not guaranteed and is not
    asserted here; only that separate calls are equivalently configured."""
    monkeypatch.setattr("llm_client.config.OPENROUTER_API_KEY", "dummy-key")
    client1 = LLMFactory.get_client("openrouter", "test-model")
    client2 = LLMFactory.get_client("openrouter", "test-model")
    assert client1 is not client2
    assert client1.model == client2.model == "test-model"
    assert client1.api_base == client2.api_base == "https://openrouter.ai/api/v1"


def test_get_openrouter_client_raises_without_api_key(monkeypatch):
    monkeypatch.setattr("llm_client.config.OPENROUTER_API_KEY", None)
    with pytest.raises(ValueError):
        openrouter_client_mod.get_openrouter_client("test-model")


@pytest.mark.asyncio
async def test_requests_do_not_serialise_behind_one_slow_call(monkeypatch):
    """The pacing lock must gate when a request STARTS, not span the call.

    Holding it across the await made the semaphore useless and let a single
    hung call block every other waiter, which stalled a full gate run.
    """
    monkeypatch.setattr("llm_client.config.OPENROUTER_API_KEY", "dummy-key")
    monkeypatch.setattr("llm_client.openrouter_client._MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr("llm_client.openrouter_client._last_request_time", 0.0)

    in_flight = 0
    peak = 0

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace()
            self.chat.completions = SimpleNamespace()

    async def fake_create(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return _FakeCompletion(content="ok")

    mock_client = MockClient()
    mock_client.chat.completions.create = fake_create

    def mock_constructor(*args, **kwargs):
        return mock_client

    # Same reasoning as test_openrouter_client_accomplete_calls_openai_client:
    # AsyncOpenAI must be patched where openrouter_client.py looks it up.
    monkeypatch.setattr("llm_client.openrouter_client.AsyncOpenAI", mock_constructor)

    client = openrouter_client_mod.get_openrouter_client("test-model")
    wrapped = client.chat.completions.create

    await asyncio.gather(*(wrapped(messages=[]) for _ in range(4)))

    assert peak > 1, "requests serialised; the lock still spans the API call"


@pytest.mark.asyncio
async def test_timeout_is_logged_with_a_visible_error(monkeypatch, caplog):
    """asyncio.TimeoutError carries no message, so logging it with %s emitted
    "error=" with nothing after it, making a request timeout indistinguishable
    from a blank field. Over 1,800 calls that is the difference between a
    diagnosable stall and a silent one."""
    monkeypatch.setattr("llm_client.openrouter_client._MIN_REQUEST_INTERVAL", 0.0)
    monkeypatch.setattr("llm_client.openrouter_client._last_request_time", 0.0)
    monkeypatch.setattr("llm_client.openrouter_client.REQUEST_TIMEOUT_SEC", 0.01)

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace()
            self.chat.completions = SimpleNamespace()

    async def never_returns(**kwargs):
        await asyncio.sleep(10)

    mock_client = MockClient()
    mock_client.chat.completions.create = never_returns
    monkeypatch.setattr(
        "llm_client.openrouter_client.AsyncOpenAI", lambda *positional_args, **keyword_args: mock_client
    )

    client = openrouter_client_mod.get_openrouter_client("test-model")
    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.TimeoutError):
            await client.chat.completions.create(messages=[])

    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings, "a timed-out call must log a warning"
    assert any("TimeoutError" in message for message in warnings), warnings
    assert not any(message.rstrip().endswith("error=") for message in warnings), (
        "the exception rendered as an empty string"
    )
