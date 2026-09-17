"""Answerer: citation parsing, anti-leakage, token/latency capture.

The LLM stand-in is a real OpenAILike with its client swapped out, so
tests exercise the genuine achat() path rather than a hand-rolled mock.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai_like import OpenAILike

import llm_client.config as config
from pipelines.answerer import Answerer, build_prompt, parse_citations


def _fake_llm_client(response_content: str, prompt_tokens: int = 2100, completion_tokens: int = 40) -> OpenAILike:
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content=response_content, role="assistant", tool_calls=None), logprobs=None)
    ]
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    raw_client = MagicMock()
    raw_client.chat.completions.create = AsyncMock(return_value=response)

    client = OpenAILike(model="test-model", api_base="http://test", api_key="test", is_chat_model=True)
    client._aclient = raw_client
    return client


def _node(node_id: str, text: str, **metadata) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=metadata), score=1.0)


def _sent_messages(client: OpenAILike) -> list[dict]:
    return client._aclient.chat.completions.create.call_args.kwargs["messages"]


async def _run_answer(client: OpenAILike, query_text: str, nodes: list[NodeWithScore]):
    with patch("pipelines.answerer.LLMFactory.get_client_for_stage", return_value=client):
        return await Answerer().answer(query_text, nodes)


# --- citation parsing ---


def test_parse_citations_extracts_marker():
    assert parse_citations("Net sales were $394.3B. [[node:AAPL_2025_n0421]]") == ["AAPL_2025_n0421"]


def test_parse_citations_preserves_order_and_deduplicates():
    raw = (
        "Claim one. [[node:MSFT_2024_n0118]] "
        "Claim two. [[node:MSFT_2024_n0742]] "
        "Claim three, same source as the first. [[node:MSFT_2024_n0118]]"
    )
    assert parse_citations(raw) == ["MSFT_2024_n0118", "MSFT_2024_n0742"]


def test_parse_citations_returns_empty_when_no_markers():
    assert parse_citations("The sources do not contain this information.") == []


def test_parse_citations_accepts_hyphenated_ids():
    assert parse_citations("x [[node:doc-1_n-07]]") == ["doc-1_n-07"]


def test_parse_citations_ignores_malformed_markers():
    assert parse_citations("[node:A] [[node A]] [[node:]] [[node:GOOD_1]]") == ["GOOD_1"]


# --- anti-leakage ---


def test_build_prompt_contains_only_query_and_node_content():
    nodes = [_node("N1", "Revenue rose 12%."), _node("N2", "Costs fell 3%.")]
    prompt = build_prompt("How did revenue change?", nodes)

    assert "How did revenue change?" in prompt
    assert "Revenue rose 12%." in prompt
    assert "Costs fell 3%." in prompt
    assert "N1" in prompt and "N2" in prompt


def test_build_prompt_excludes_node_metadata():
    """parent_item_header and page numbers hint at where the answer sits,
    which retrieval is scored on."""
    nodes = [_node("N1", "Revenue rose 12%.", parent_item_header="Item 7", source_page_num=41, document_id="AAPL_2025")]
    prompt = build_prompt("How did revenue change?", nodes)

    assert "Item 7" not in prompt
    assert "41" not in prompt
    assert "AAPL_2025" not in prompt


def test_build_prompt_handles_empty_node_list():
    prompt = build_prompt("Anything?", [])
    assert "Anything?" in prompt


@pytest.mark.asyncio
async def test_prompt_on_the_wire_leaks_no_ground_truth():
    """The whole payload, both roles, carries only the question, node ids,
    node text and fixed instructions."""
    nodes = [_node("AAPL_2025_n0421", "Total net sales were $394.3 billion.")]
    client = _fake_llm_client("Answer. [[node:AAPL_2025_n0421]]")

    await _run_answer(client, "What were FY2025 net sales?", nodes)

    messages = _sent_messages(client)
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = "\n".join(message["content"] for message in messages)

    permitted = {
        "What were FY2025 net sales?",
        "Total net sales were $394.3 billion.",
        "AAPL_2025_n0421",
    }
    residue = payload
    for fragment in permitted:
        residue = residue.replace(fragment, "")

    # Whatever is left is fixed prompt scaffolding, so it must not vary with
    # the benchmark data: no ground-truth answer, no exemplar, no quadrant.
    for leaked in ("ground truth", "example", "exemplar", "quadrant", "gt_citations", "$394.3"):
        assert leaked.lower() not in residue.lower()


@pytest.mark.asyncio
async def test_prompt_instructs_the_citation_marker_format():
    client = _fake_llm_client("ok")
    await _run_answer(client, "q", [_node("N1", "text")])

    system = next(message["content"] for message in _sent_messages(client) if message["role"] == "system")
    assert "[[node:" in system


# --- AnswerResult population ---


@pytest.mark.asyncio
async def test_answer_populates_result_fields():
    nodes = [_node("AAPL_2025_n0421", "Total net sales were $394.3 billion.")]
    raw = "Apple's FY2025 net sales were $394.3B. [[node:AAPL_2025_n0421]]"
    client = _fake_llm_client(raw, prompt_tokens=2100, completion_tokens=40)

    result = await _run_answer(client, "What were FY2025 net sales?", nodes)

    assert result.raw_text == raw
    assert result.cited_node_ids == ["AAPL_2025_n0421"]
    assert result.input_tokens == 2100
    assert result.output_tokens == 40
    assert result.latency_sec >= 0


@pytest.mark.asyncio
async def test_answer_keeps_markers_in_raw_text():
    """Markers stay in raw_text; only parse_citations strips them."""
    raw = "Claim. [[node:N1]]"
    client = _fake_llm_client(raw)
    result = await _run_answer(client, "q", [_node("N1", "text")])
    assert "[[node:N1]]" in result.raw_text


# --- fixed model routing / temperature ---


def test_answerer_default_model_tracks_model_routing():
    """The no-args default must be derived from MODEL_ROUTING, not a literal,
    so a future repoint of "answerer" cannot silently go stale here."""
    assert Answerer().model == config.MODEL_ROUTING["answerer"]["model"]


def test_answerer_rejects_a_model_other_than_the_routed_one():
    with pytest.raises(ValueError, match="fixed"):
        Answerer(model="llama-3.1-8b-instant")


def test_answerer_rejects_nonzero_temperature():
    with pytest.raises(ValueError, match="temperature"):
        Answerer(temperature=0.7)


def test_answerer_construction_does_not_build_a_client():
    """Constructing an Answerer must not reach for credentials; the client
    is built on first use so a 900-cell run shares one."""
    with patch("pipelines.answerer.LLMFactory.get_client_for_stage") as factory:
        Answerer()
    factory.assert_not_called()
