"""Tests for LLMFactory's client construction.

Confirms the retry/semaphore-wrapped client built by get_groq_client() (or
get_nim_client()) is the client actually used for network calls, not
silently discarded.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

import llm_client.config as config
from llm_client.llm_factory import LLMFactory


def _fake_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content, role="assistant", tool_calls=None))]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return response


@pytest.mark.asyncio
async def test_get_client_groq_routes_achat_through_injected_client():
    fake_raw_client = MagicMock()
    fake_raw_client.chat.completions.create = AsyncMock(return_value=_fake_response("hi"))

    with patch("llm_client.groq_client.get_groq_client", return_value=fake_raw_client):
        client = LLMFactory.get_client("groq", "some-model")
        response = await client.achat([ChatMessage(role=MessageRole.USER, content="hello")])

    assert response.message.content == "hi"
    fake_raw_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_client_nvidia_routes_achat_through_injected_client():
    fake_raw_client = MagicMock()
    fake_raw_client.chat.completions.create = AsyncMock(return_value=_fake_response("hi"))

    with patch("llm_client.nim_client.get_nim_client", return_value=fake_raw_client):
        client = LLMFactory.get_client("nvidia", "some-model")
        response = await client.achat([ChatMessage(role=MessageRole.USER, content="hello")])

    assert response.message.content == "hi"
    fake_raw_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_client_openrouter_routes_achat_through_injected_client():
    fake_raw_client = MagicMock()
    fake_raw_client.chat.completions.create = AsyncMock(return_value=_fake_response("hi"))

    with patch("llm_client.openrouter_client.get_openrouter_client", return_value=fake_raw_client):
        client = LLMFactory.get_client("openrouter", "some-model")
        response = await client.achat([ChatMessage(role=MessageRole.USER, content="hello")])

    assert response.message.content == "hi"
    fake_raw_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_client_openrouter_caps_reasoning_via_extra_body():
    """Reasoning-tuned models (nemotron, gpt-oss) can spend their entire
    completion-token budget on hidden reasoning and return
    message.content=None. OpenRouter's fix is the `reasoning` request
    field, and the openai SDK has no typed kwarg for it, so it must travel
    via `extra_body`. The extra_body is now passed to get_client rather than
    hardcoded, enabling per-stage customization.
    """
    fake_raw_client = MagicMock()
    fake_raw_client.chat.completions.create = AsyncMock(return_value=_fake_response("hi"))

    with patch("llm_client.openrouter_client.get_openrouter_client", return_value=fake_raw_client):
        client = LLMFactory.get_client(
            "openrouter", "some-model", extra_body={"reasoning": {"effort": "low"}}
        )
        await client.achat([ChatMessage(role=MessageRole.USER, content="hello")])

    _, call_kwargs = fake_raw_client.chat.completions.create.call_args
    assert call_kwargs.get("extra_body") == {"reasoning": {"effort": "low"}}


def test_get_client_for_stage_forwards_routing_extra_body(monkeypatch):
    """A stage's extra_body reaches OpenAILike.additional_kwargs verbatim."""
    monkeypatch.setitem(
        config.MODEL_ROUTING,
        "judge",
        {
            "model": "qwen/qwen3.6-27b",
            "provider": "openrouter",
            "extra_body": {"provider": {"order": ["Chutes"], "allow_fallbacks": False}},
        },
    )
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    llm = LLMFactory.get_client_for_stage("judge")
    assert llm.additional_kwargs["extra_body"] == {
        "provider": {"order": ["Chutes"], "allow_fallbacks": False}
    }


def test_openrouter_stage_without_extra_body_sends_none(monkeypatch):
    """No extra_body in the routing entry means none is sent.

    Llama 3.3 70B rejects `reasoning`; a hardcoded default would reach it.
    """
    monkeypatch.setitem(
        config.MODEL_ROUTING,
        "answerer",
        {"model": "meta-llama/llama-3.3-70b-instruct", "provider": "openrouter"},
    )
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    llm = LLMFactory.get_client_for_stage("answerer")
    assert "extra_body" not in llm.additional_kwargs


def test_generator_and_critic_receive_reasoning_from_live_routing(monkeypatch):
    """Generator and critic stages receive _REASONING_LOW from the real, unpatched
    config.MODEL_ROUTING. This is the regression guard for the hardcoded-default
    removal: a future edit that drops extra_body from either entry will fail here
    rather than silently changing what those stages send to OpenRouter.
    """
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")

    # Test generator
    gen_llm = LLMFactory.get_client_for_stage("generator")
    assert gen_llm.additional_kwargs["extra_body"] == {"reasoning": {"effort": "low"}}

    # Test critic
    crit_llm = LLMFactory.get_client_for_stage("critic")
    assert crit_llm.additional_kwargs["extra_body"] == {"reasoning": {"effort": "low"}}
