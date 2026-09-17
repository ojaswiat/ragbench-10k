"""ONE live, throttled smoke test proving Judge.score_row() speaks Groq's real
chat protocol and returns a parseable 1-10 score, not just the mocked shape in
tests/test_async_judge.py.

It makes exactly one real achat() call, with a frozen Q3 exemplar as the prefix
and a frozen JEQ-shaped row as the target. No DB read, no loop, one Groq call
per opt-in run.

Skipped by default. To run: set RUN_LIVE_GROQ_TESTS=1 and resolve a real
GROQ_API_KEY via config.py, then run pytest as usual. Plain `pytest -q` never
runs it and never spends quota.
"""
import os

import pytest

import llm_client.config as config
from judge.async_judge import Judge, build_prefix

_RUN_LIVE = os.getenv("RUN_LIVE_GROQ_TESTS") == "1"

# Frozen, not a DB read: one graded exemplar and one target row, both Q3.
_EXEMPLAR = {
    "query_id": "GQ_T3_LIVE",
    "quadrant": "Q3_Direct_Table",
    "query_text": "What was Microsoft's total revenue for FY2024?",
    "ground_truth_answer": "$245.1B",
    "gt_citations": ["MSFT_2024_n0421"],
    "example_output": "Microsoft's FY2024 total revenue was $245.1B. [[node:MSFT_2024_n0421]]",
    "human_score": 90,
    "human_reasoning": "Correct figure, cites the valid source.",
    "is_good": 1,
    "document_id": "SEC_10K_MSFT_2024",
}

_TARGET_ROW = {
    "result_id": "R_LIVE",
    "query_id": "JEQ_LIVE",
    "pipeline": "P1_vector",
    "k_value": 5,
    "retrieved_node_ids": ["AAPL_2025_n0421", "AAPL_2025_n0420"],
    "pipeline_output": "Apple's FY2025 net sales were $394.3B. [[node:AAPL_2025_n0421]]",
    "cited_node_ids": ["AAPL_2025_n0421"],
    "quadrant": "Q3_Direct_Table",
    "query_text": "What were Apple's total net sales for FY2025?",
    "ground_truth_answer": "$394.3B",
    "gt_citations": ["AAPL_2025_n0421"],
    "document_id": "SEC_10K_AAPL_2025",
}


@pytest.mark.live
@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="Live Groq smoke test; set RUN_LIVE_GROQ_TESTS=1 to run (spends real Groq quota).",
)
@pytest.mark.skipif(
    not config.GROQ_API_KEY,
    reason="No GROQ_API_KEY resolved via config.py; cannot make a real Groq call.",
)
@pytest.mark.asyncio
async def test_judge_score_row_live_round_trip_against_real_groq_api():
    """Exercises the real Qwen judge call end to end, with no client mocking.

    Proves the prompt Judge.score_row() builds is accepted by the live API and
    that its reply parses into a valid 1-10 score, the one thing the mocked
    test cannot cover. The target answer is correct, but this asserts only a
    parseable in-range integer, not a specific value.
    """
    prefix = build_prefix("Q3_Direct_Table", [_EXEMPLAR])
    score = await Judge().score_row(_TARGET_ROW, prefix)

    assert isinstance(score, int)
    assert 1 <= score <= 10
