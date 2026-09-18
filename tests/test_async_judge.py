"""async_judge: quadrant-filtered few-shot, no search tool, folded metrics.

The Judge scores each JEQ output 1-5. It sees only the 5 golden_queries
exemplars sharing the row's quadrant, never a search tool, and never a JEQ row
as an exemplar. The same pass computes the deterministic metrics in code, so a
scored row is written complete in one update.

The LLM stand-in is a real OpenAILike with its async client swapped, matching
what LLMFactory returns in production.
"""
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.llms.openai_like import OpenAILike

from core import database_manager as dbm
from judge import async_judge
from judge.async_judge import (
    Judge,
    _rescale_to_1_5,
    build_prefix,
    compute_deterministic_metrics,
    parse_judge_score,
)

TEST_DB = "test_benchmark_async_judge.db"


@pytest.fixture(autouse=True)
def clean_db():
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        if os.path.exists(path):
            os.remove(path)
    yield
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB + suffix
        if os.path.exists(path):
            os.remove(path)


def _fake_judge_client(response_content: str) -> OpenAILike:
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content=response_content, role="assistant", tool_calls=None), logprobs=None)
    ]
    response.usage = MagicMock(prompt_tokens=1500, completion_tokens=12, total_tokens=1512)
    raw_client = MagicMock()
    raw_client.chat.completions.create = AsyncMock(return_value=response)
    client = OpenAILike(model="test-model", api_base="http://test", api_key="test", is_chat_model=True)
    client._aclient = raw_client
    return client


def _row(quadrant="Q3_Direct_Table"):
    return {
        "result_id": "R1",
        "query_id": "JEQ_001",
        "pipeline": "P1_vector",
        "k_value": 5,
        "retrieved_node_ids": ["AAPL_2025_n0421", "AAPL_2025_n0420"],
        "pipeline_output": "Net sales were $394.3B. [[node:AAPL_2025_n0421]]",
        "cited_node_ids": ["AAPL_2025_n0421"],
        "quadrant": quadrant,
        "query_text": "What were FY2025 net sales?",
        "ground_truth_answer": "$394.3B",
        "gt_citations": ["AAPL_2025_n0421"],
        "document_id": "SEC_10K_AAPL_2025",
    }


def _exemplar(query_id, quadrant, score=90, output="It was $1B. [[node:n1]]"):
    return {
        "query_id": query_id,
        "quadrant": quadrant,
        "query_text": f"exemplar question {query_id}",
        "ground_truth_answer": "$1B",
        "gt_citations": ["n1"],
        "example_output": output,
        "human_score": score,
        "human_reasoning": "correct figure and citation.",
        "is_good": 1,
        "document_id": "SEC_10K_AAPL_2025",
    }


# --- parse_judge_score ---


def test_parse_score_from_json():
    assert parse_judge_score('{"score": 4, "justification": "good"}') == 4


def test_parse_score_from_json_alt_key():
    assert parse_judge_score('{"judge_score": 3}') == 3


def test_parse_score_from_string_value():
    assert parse_judge_score('{"score": "5"}') == 5


def test_parse_score_clamps_out_of_range():
    assert parse_judge_score('{"score": 12}') == 5
    assert parse_judge_score('{"score": 0}') == 1


def test_parse_score_bare_integer_fallback():
    assert parse_judge_score("I would rate this a 4 out of 5.") == 4


def test_parse_score_unparseable_raises():
    with pytest.raises(ValueError):
        assert parse_judge_score("no number here")


# --- compute_deterministic_metrics (no LLM) ---


def test_deterministic_metrics_q3():
    metrics = compute_deterministic_metrics(_row("Q3_Direct_Table"))
    assert metrics["precision_at_k"] == 0.2
    assert metrics["recall_at_k"] == 1.0
    assert metrics["evidence_hit"] == 1
    assert metrics["citation_match"] == 1
    assert metrics["exact_match"] == 1  # figure extracted from the sentence
    assert 0.0 <= metrics["token_f1"] <= 1.0


def test_deterministic_metrics_exact_match_null_for_q2():
    metrics = compute_deterministic_metrics(_row("Q2_Implicit_Text"))
    assert metrics["exact_match"] is None


def test_deterministic_metrics_evidence_hit_zero_when_missed():
    row = _row("Q3_Direct_Table")
    row["retrieved_node_ids"] = ["wrong_1", "wrong_2"]
    metrics = compute_deterministic_metrics(row)
    assert metrics["recall_at_k"] == 0.0
    assert metrics["evidence_hit"] == 0


def test_deterministic_metrics_citation_match_zero_on_bad_citation():
    row = _row("Q3_Direct_Table")
    row["cited_node_ids"] = ["hallucinated_node"]
    metrics = compute_deterministic_metrics(row)
    assert metrics["citation_match"] == 0


# --- build_prefix: only the quadrant's exemplars, none from other quadrants ---


def test_prefix_includes_quadrant_exemplars_only():
    exemplars = [_exemplar(f"GQ_T3_{index}", "Q3_Direct_Table") for index in range(5)]
    prefix = build_prefix("Q3_Direct_Table", exemplars)
    for exemplar in exemplars:
        assert exemplar["query_text"] in prefix
    assert "exemplar question GQ_T1" not in prefix  # no other-quadrant leakage


def test_prefix_rescales_exemplar_scores_to_1_5():
    """golden_queries.human_score is 0-100; the judge outputs 1-5, so the
    teaching scores are shown on the judge's own scale."""
    prefix = build_prefix("Q3_Direct_Table", [_exemplar("GQ_T3_1", "Q3_Direct_Table", score=90)])
    assert "Correct score (1-5): 5" in prefix
    assert "90" not in prefix


@pytest.mark.parametrize(
    "native,band",
    [(100, 5), (90, 5), (89, 4), (75, 4), (74, 3), (50, 3), (49, 2), (25, 2), (24, 1), (0, 1)],
)
def test_rescale_band_floors(native, band):
    """Band edges are explicit, not round(score / 20): banker's rounding would
    send the two 50-scored 'half the question answered' exemplars into band 2,
    where they would read as wrong-value answers rather than partial ones."""
    assert _rescale_to_1_5(native) == band


def test_every_band_is_demonstrated_by_at_least_one_exemplar():
    """The 1-10 scale left bands 1, 3, 6, 7 and 8 with no exemplar, so the
    Judge never emitted them and a reference scorer using them could not be
    matched. The 1-5 mapping must not reintroduce an unreachable band."""
    native_labels = [100, 98, 97, 97, 93, 93, 93, 90, 90, 88,
                     50, 50, 45, 45, 25, 22, 20, 18, 15, 15]
    assert {_rescale_to_1_5(native_label) for native_label in native_labels} == {1, 2, 3, 4, 5}


# --- Judge scoring: no tool call, judge model enforced ---


def test_judge_rejects_wrong_model():
    with pytest.raises(ValueError, match="fixed"):
        Judge(model="llama-3.3-70b-versatile")


def test_judge_rejects_nonzero_temperature():
    with pytest.raises(ValueError, match="temperature"):
        Judge(temperature=0.5)


@pytest.mark.asyncio
async def test_score_row_uses_plain_chat_not_tools():
    client = _fake_judge_client('{"score": 4}')
    prefix = build_prefix("Q3_Direct_Table", [_exemplar("GQ_T3_1", "Q3_Direct_Table")])
    with patch("judge.async_judge.LLMFactory.get_client_for_stage", return_value=client):
        score = await Judge().score_row(_row(), prefix)
    assert score == 4
    # plain achat path -> chat.completions.create, and never a search tool.
    client._aclient.chat.completions.create.assert_awaited()
    sent = client._aclient.chat.completions.create.call_args.kwargs
    assert "tools" not in sent or not sent["tools"]


# --- judge_jeq_rows: end-to-end over a seeded DB, one UPDATE per row ---


@pytest.mark.asyncio
async def test_judge_jeq_rows_scores_and_writes(monkeypatch):
    monkeypatch.setattr(async_judge.config, "LOCAL_TEST_THROTTLE", False, raising=False)
    monkeypatch.setattr(async_judge, "LOCAL_TEST_THROTTLE", False, raising=False)

    await dbm.init_db(TEST_DB)
    judge_validation = {
        "query_id": "JEQ_001",
        "quadrant": "Q3_Direct_Table",
        "query_text": "What were FY2025 net sales?",
        "ground_truth_answer": "$394.3B",
        "gt_citations": ["AAPL_2025_n0421"],
        "document_id": "SEC_10K_AAPL_2025",
    }
    await dbm.insert_judge_validation(TEST_DB, judge_validation)
    for index in range(5):
        await dbm.insert_golden_query(TEST_DB, _exemplar(f"GQ_T3_{index}", "Q3_Direct_Table"))
    result_row = {
        "result_id": "R1",
        "source_set": "JEQ",
        "query_id": "JEQ_001",
        "pipeline": "P1_vector",
        "k_value": 5,
        "retrieved_node_ids": ["AAPL_2025_n0421", "AAPL_2025_n0420"],
        "pipeline_output": "Net sales were $394.3B. [[node:AAPL_2025_n0421]]",
        "cited_node_ids": ["AAPL_2025_n0421"],
    }
    await dbm.upsert_result(TEST_DB, result_row)

    client = _fake_judge_client('{"score": 5}')
    with patch("judge.async_judge.LLMFactory.get_client_for_stage", return_value=client):
        summary = await async_judge.judge_jeq_rows(TEST_DB)

    assert summary["judged"] == 1
    rows = await dbm.get_results(TEST_DB, "JEQ")
    row = next(stored_row for stored_row in rows if stored_row["result_id"] == "R1")
    assert row["judge_score"] == 5
    assert row["precision_at_k"] == 0.2
    assert row["recall_at_k"] == 1.0
    assert row["evidence_hit"] == 1
    assert row["citation_match"] == 1
    assert row["exact_match"] == 1


# --- rubric/exemplar consistency (Task 3b) ---


def test_rubric_does_not_ask_the_judge_to_grade_citations():
    """Guardrails 4b reserves citation matching for deterministic code.

    citation_audit() already computes it into results.citation_match, so a
    rubric clause about citing valid sources both duplicates that check and
    invites the Judge to dock marks for a property no exemplar demonstrates.
    """
    rubric = async_judge._RUBRIC.lower()
    assert "cite" not in rubric
    assert "citation" not in rubric
    assert "source" not in rubric


def test_rubric_avoids_words_that_misdescribe_exemplars():
    """The rubric is calibrated against exemplars that show no citations, so
    words like 'basis', 'verify', 'audit', or 'check it against' (which presume
    citations) would teach the Judge a standard it is never shown evidence for.

    Real pipeline answers carry [[node:id]] markers; bare exemplars do not.
    Embedding one but not the other invites the Judge to anchor low on all
    real answers for a property it never saw fail in exemplars.
    """
    rubric = async_judge._RUBRIC.lower()
    forbidden = ("basis", "verify", "verifiable", "audit", "check it against")
    for word in forbidden:
        assert word not in rubric, f"forbidden word '{word}' found in rubric"


def test_rubric_still_pins_the_json_response_shape():
    """parse_judge_score() prefers the JSON object; the rubric must keep
    asking for it."""
    assert '{"score"' in async_judge._RUBRIC
    assert "justification" in async_judge._RUBRIC
