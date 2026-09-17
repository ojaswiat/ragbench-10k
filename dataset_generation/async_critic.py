"""Critic: independently re-derives an answer using a local search tool.

Uses qwen/qwen3.6-27b, a different model family from the Generator, per
the anti-self-grading invariant. Given only the query text, never the
Generator's answer or citations, it must search the filing itself first.
"""
import json

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from llm_client.llm_factory import LLMFactory
from .search_tool import (
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_NAME,
    SearchFilingArgs,
    search_filing_nodes,
)

_MAX_TOOL_ROUNDS = 5

_SYSTEM_PROMPT = (
    "You are a financial-filing fact-checker. You will be given a question about "
    "a SEC 10-K filing. You do not know the answer yet. Use the search_filing tool "
    "to find the relevant passages, then respond with ONLY a JSON object (no "
    "markdown fences, no commentary): {\"cited_node_ids\": [\"node_id\", ...], "
    "\"computed_answer\": \"...\"}. Cite only node_ids you actually found via search_filing."
)


def _build_search_tool(all_nodes: list[dict]) -> FunctionTool:
    """Wraps search_filing_nodes as a LlamaIndex tool bound to one filing.

    The nodes are closed over rather than passed as a tool argument, so the
    Critic can only choose search terms, never which corpus it searches.
    """

    def search_filing(query: str) -> str:
        results = search_filing_nodes(all_nodes, query)
        return json.dumps(
            [{"node_id": result["node_id"], "content": result["content"]} for result in results]
        )

    return FunctionTool.from_defaults(
        fn=search_filing,
        name=SEARCH_TOOL_NAME,
        description=SEARCH_TOOL_DESCRIPTION,
        fn_schema=SearchFilingArgs,
    )


async def critique_query(
    query_text: str, all_nodes: list[dict], return_messages: bool = False
) -> dict:
    """Runs the Critic's search-then-answer loop.

    Returns {"cited_node_ids": [...], "computed_answer": "..."}. If
    return_messages is True, the dict also carries the full ChatMessage
    history under "messages", for tests asserting on tool-call behaviour.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=query_text),
    ]

    critic_client = LLMFactory.get_client_for_stage("critic")
    search_tool = _build_search_tool(all_nodes)

    for _ in range(_MAX_TOOL_ROUNDS):
        # client is a LlamaIndex LLM (OpenAILike, a FunctionCallingLLM), not a
        # raw OpenAI SDK client. Model and temperature=0 are already fixed at
        # construction, so tool calls go through achat_with_tools() rather
        # than a `tools=` kwarg on chat.completions.create(). chat_history is
        # copied because the library appends to the list it is given.
        response = await critic_client.achat_with_tools(
            tools=[search_tool],
            chat_history=list(messages),
            allow_parallel_tool_calls=True,
        )
        messages.append(response.message)

        tool_calls = critic_client.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if tool_calls:
            for tool_call in tool_calls:
                tool_output = search_tool.call(**tool_call.tool_kwargs)
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content=tool_output.content,
                        additional_kwargs={"tool_call_id": tool_call.tool_id},
                    )
                )
            continue

        final = json.loads(response.message.content)
        if return_messages:
            final["messages"] = messages
        return final

    raise RuntimeError(f"Critic exceeded {_MAX_TOOL_ROUNDS} tool-call rounds without a final answer")