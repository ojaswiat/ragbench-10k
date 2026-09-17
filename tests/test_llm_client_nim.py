"""Tests for the LLM client abstraction (NIM provider)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from llm_client.llm_factory import LLMFactory
from llm_client import nim_client as nim_client_mod


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeCompletion:
    def __init__(self, content: str = "test"):
        self.choices = [_FakeChoice(content)]


@pytest.mark.asyncio
async def test_nim_client_accomplete_calls_openai_client(monkeypatch):
    """Transparent to kwargs despite the retry/semaphore/rate-limit wrapping."""
    captured = {}

    monkeypatch.setattr("llm_client.config.NIM_API_KEY", "dummy-key")
    # Avoid the 40 RPM rate-limit sleep affecting unrelated call timing
    # across tests in this module.
    monkeypatch.setattr("llm_client.nim_client._last_request_time", 0.0)

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

    # nim_client.py does `from openai import AsyncOpenAI` at import time, so
    # the name must be patched where it is looked up (llm_client.nim_client),
    # not on the openai package itself.
    monkeypatch.setattr("llm_client.nim_client.AsyncOpenAI", mock_constructor)

    # get_nim_client() is called directly, not through LLMFactory.get_client(),
    # which would wrap it in a llama_index OpenAILike LLM exposing chat() and
    # acomplete() rather than a raw chat.completions.create.
    client = nim_client_mod.get_nim_client("test-model")
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

    captured.clear()
    await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        tools=[{"type": "function", "function": {"name": "test"}}],
    )
    assert captured == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "tools": [{"type": "function", "function": {"name": "test"}}],
    }


@pytest.mark.asyncio
async def test_get_llm_client_returns_equivalent_client_nim(monkeypatch):
    """No caching: each call builds a new client, so identity is not
    asserted here, only that separate calls are equivalently configured.
    """
    monkeypatch.setattr("llm_client.config.NIM_API_KEY", "dummy-key")
    client1 = LLMFactory.get_client("nvidia", "test-model")
    client2 = LLMFactory.get_client("nvidia", "test-model")
    assert client1 is not client2
    assert client1.model == client2.model == "test-model"
    assert client1.api_base == client2.api_base


@pytest.mark.asyncio
async def test_get_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError):
        LLMFactory.get_client("unknown", "test-model")


def test_get_client_wires_callback_manager_when_provided(monkeypatch):
    """Forwarded so events from llama_index's llm_chat_callback() decorator
    (token-usage tracking) land on the caller's bus, not an empty default.
    """
    monkeypatch.setattr("llm_client.config.GROQ_API_KEY", "dummy-key")

    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])

    client = LLMFactory.get_client("groq", "some-model", callback_manager=callback_manager)

    assert client.callback_manager is callback_manager
    assert token_counter in client.callback_manager.handlers


def test_get_client_omitting_callback_manager_still_constructs(monkeypatch):
    """callback_manager omitted defaults to None; construction still succeeds."""
    monkeypatch.setattr("llm_client.config.GROQ_API_KEY", "dummy-key")

    client = LLMFactory.get_client("groq", "some-model")

    assert client is not None
    assert client.model == "some-model"