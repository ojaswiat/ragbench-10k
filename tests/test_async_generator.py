"""Pins generate_query()'s LLM call shape to LlamaIndex's real achat()
interface, using a real OpenAILike instance with its private async client
swapped out, rather than a raw OpenAI-SDK client.chat.completions.create()
pattern.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.llms.openai_like import OpenAILike

from dataset_generation.async_generator import generate_query

_SECTION = {
    "section_header": "Item 7",
    "node_ids": ["AAPL_2025_n0001"],
    "content": "Total net sales were $394.3 billion.",
    "document_id": "AAPL_2025",
}

_RESPONSE_JSON = (
    '{"query_text": "What were total net sales?", '
    '"ground_truth_answer": "$394.3 billion", '
    '"gt_citations": ["AAPL_2025_n0001"]}'
)


def _fake_llm_client(response_content: str) -> OpenAILike:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=response_content, role="assistant", tool_calls=None))]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    raw_client = MagicMock()
    raw_client.chat.completions.create = AsyncMock(return_value=response)

    client = OpenAILike(model="test-model", api_base="http://test", api_key="test", is_chat_model=True)
    client._aclient = raw_client
    return client


def _sent_user_content(client: OpenAILike) -> str:
    """The user-role prompt text read back from the serialised
    {"role", "content"} dicts OpenAILike.achat() actually sends as the
    `messages` kwarg.
    """
    messages = client._aclient.chat.completions.create.call_args.kwargs["messages"]
    user_messages = [message["content"] for message in messages if message["role"] == "user"]
    assert len(user_messages) == 1, f"expected exactly one user message, got {len(user_messages)}"
    return user_messages[0]


@pytest.mark.asyncio
async def test_generate_query_returns_parsed_payload():
    section = {
        "section_header": "Item 7",
        "node_ids": ["AAPL_2025_n0001"],
        "content": "Total net sales were $394.3 billion.",
        "document_id": "AAPL_2025",
    }
    response_json = (
        '{"query_text": "What were total net sales?", '
        '"ground_truth_answer": "$394.3 billion", '
        '"gt_citations": ["AAPL_2025_n0001"]}'
    )
    fake_client = _fake_llm_client(response_json)

    with patch(
        "dataset_generation.async_generator.LLMFactory.get_client_for_stage",
        return_value=fake_client,
    ):
        result = await generate_query(section, "Q1_Direct_Text")

    assert result == {
        "query_text": "What were total net sales?",
        "ground_truth_answer": "$394.3 billion",
        "gt_citations": ["AAPL_2025_n0001"],
        "quadrant": "Q1_Direct_Text",
        "document_id": "AAPL_2025",
    }


@pytest.mark.asyncio
async def test_generate_query_first_attempt_has_no_feedback_in_prompt():
    fake_client = _fake_llm_client(_RESPONSE_JSON)

    with patch(
        "dataset_generation.async_generator.LLMFactory.get_client_for_stage",
        return_value=fake_client,
    ):
        await generate_query(_SECTION, "Q1_Direct_Text")

    user_content = _sent_user_content(fake_client)
    assert "previous attempt" not in user_content.lower()
    assert _SECTION["content"] in user_content


@pytest.mark.asyncio
async def test_generate_query_retry_includes_previous_attempt_feedback_in_prompt():
    feedback = "Citation AAPL_2025_n0001 did not support the stated answer."
    fake_client = _fake_llm_client(_RESPONSE_JSON)

    with patch(
        "dataset_generation.async_generator.LLMFactory.get_client_for_stage",
        return_value=fake_client,
    ):
        await generate_query(
            _SECTION, "Q1_Direct_Text", previous_attempt_feedback=feedback
        )

    user_content = _sent_user_content(fake_client)
    assert feedback in user_content
    # temperature=0 makes an unchanged prompt reproduce the rejected output,
    # so the retry must explicitly steer toward a different candidate.
    assert "different" in user_content.lower()
