"""Pins critique_query()'s tool-calling loop to LlamaIndex's real
achat_with_tools() interface. The Critic is called with only the query and
retrieved nodes, never the Generator's answer or citations.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.llms.openai_like import OpenAILike

from dataset_generation.async_critic import _MAX_TOOL_ROUNDS, critique_query

_NODES = [
    {"node_id": "n1", "content": "Total revenue was $100 million in fiscal 2025."},
    {"node_id": "n2", "content": "The board of directors met quarterly."},
]

_FINAL_PAYLOAD = {"cited_node_ids": ["n1"], "computed_answer": "$100 million"}


def _openai_response(content=None, tool_calls=None):
    """An OpenAI-SDK-shaped chat completion, matching what LlamaIndex's
    from_openai_message actually parses off the wire.
    """
    message = MagicMock(role="assistant", content=content, tool_calls=tool_calls, audio=None)
    response = MagicMock()
    response.choices = [MagicMock(message=message, logprobs=None)]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return response


def _tool_call(call_id="call_1", query="total revenue"):
    return MagicMock(
        id=call_id,
        type="function",
        function=MagicMock(name="search_filing", arguments=json.dumps({"query": query})),
    )


def _search_tool_call(call_id="call_1", query="total revenue"):
    tool_call = _tool_call(call_id, query)
    # MagicMock(name=...) sets the mock's repr name, not the attribute.
    tool_call.function.name = "search_filing"
    return _openai_response(content=None, tool_calls=[tool_call])


def _final_response(payload=None):
    return _openai_response(content=json.dumps(payload or _FINAL_PAYLOAD))


def _fake_llm_client(responses) -> OpenAILike:
    raw_client = MagicMock()
    raw_client.chat.completions.create = AsyncMock(side_effect=list(responses))
    client = OpenAILike(
        model="test-model", api_base="http://test", api_key="test", is_chat_model=True
    )
    client._aclient = raw_client
    return client


def _cycling_llm_client(response_factory) -> OpenAILike:
    raw_client = MagicMock()

    async def create(*args, **kwargs):
        return response_factory()

    raw_client.chat.completions.create = AsyncMock(side_effect=create)
    client = OpenAILike(
        model="test-model", api_base="http://test", api_key="test", is_chat_model=True
    )
    client._aclient = raw_client
    return client


def _sent_calls(client: OpenAILike):
    return client._aclient.chat.completions.create.call_args_list


@pytest.mark.asyncio
async def test_critique_query_uses_search_tool_then_answers():
    client = _fake_llm_client([_search_tool_call(), _final_response()])

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        result = await critique_query("What was total revenue?", _NODES)

    assert result == _FINAL_PAYLOAD
    assert len(_sent_calls(client)) == 2


@pytest.mark.asyncio
async def test_critique_query_sends_search_tool_schema_on_the_wire():
    """The tool definition must reach the provider request, not just exist locally."""
    client = _fake_llm_client([_final_response()])

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        await critique_query("What was total revenue?", _NODES)

    tools = _sent_calls(client)[0].kwargs["tools"]
    assert [tool["function"]["name"] for tool in tools] == ["search_filing"]
    assert "query" in tools[0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_critique_query_feeds_tool_results_back_keyed_by_tool_call_id():
    client = _fake_llm_client([_search_tool_call(), _final_response()])

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        await critique_query("What was total revenue?", _NODES)

    second_request_messages = _sent_calls(client)[1].kwargs["messages"]
    tool_messages = [message for message in second_request_messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    # The searched node, not the whole corpus, comes back as the tool result.
    returned = json.loads(tool_messages[0]["content"])
    assert [result["node_id"] for result in returned] == ["n1"]


@pytest.mark.asyncio
async def test_critique_query_raises_after_max_rounds_without_final_answer():
    client = _cycling_llm_client(_search_tool_call)

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        with pytest.raises(RuntimeError, match="exceeded"):
            await critique_query("What was total revenue?", _NODES)

    assert len(_sent_calls(client)) == _MAX_TOOL_ROUNDS


@pytest.mark.asyncio
async def test_critique_query_default_return_shape_is_unchanged():
    client = _fake_llm_client([_search_tool_call(), _final_response()])

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        result = await critique_query("What was total revenue?", _NODES)

    assert result == _FINAL_PAYLOAD
    assert "messages" not in result


@pytest.mark.asyncio
async def _critique_with_message_history() -> dict:
    """One critique_query run against the fake client, asked for its message history."""
    client = _fake_llm_client([_search_tool_call(), _final_response()])

    with patch(
        "dataset_generation.async_critic.LLMFactory.get_client_for_stage",
        return_value=client,
    ):
        return await critique_query(
            "What was total revenue?", _NODES, return_messages=True
        )


async def test_critique_query_with_messages_still_returns_the_parsed_verdict():
    result = await _critique_with_message_history()

    assert result["cited_node_ids"] == ["n1"]
    assert result["computed_answer"] == "$100 million"


async def test_returned_history_opens_with_the_system_then_user_message():
    messages = (await _critique_with_message_history())["messages"]

    assert all(isinstance(message, ChatMessage) for message in messages)
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER


async def test_returned_history_records_the_search_filing_tool_call():
    messages = (await _critique_with_message_history())["messages"]

    tool_call_blocks = [
        block
        for message in messages
        for block in message.blocks
        if isinstance(block, ToolCallBlock)
    ]
    assert len(tool_call_blocks) == 1
    assert tool_call_blocks[0].tool_call_id == "call_1"
    assert tool_call_blocks[0].tool_name == "search_filing"


async def test_returned_history_records_the_matching_tool_result():
    messages = (await _critique_with_message_history())["messages"]

    tool_result_messages = [message for message in messages
                            if message.role == MessageRole.TOOL]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0].additional_kwargs["tool_call_id"] == "call_1"


async def test_returned_history_ends_with_the_assistant_payload():
    messages = (await _critique_with_message_history())["messages"]

    assert messages[-1].role == MessageRole.ASSISTANT
    assert messages[-1].content == json.dumps(_FINAL_PAYLOAD)
