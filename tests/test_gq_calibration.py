"""Shape guards on the Judge's calibration set.

These assert properties of the live golden_queries table rather than of a
function: a mis-calibrated exemplar set silently corrupts every judge_score
in the benchmark, and nothing else in the suite would catch it.
"""
import sqlite3

import pytest

import database_manager as dbm

DB_PATH = "benchmark.db"


@pytest.fixture
async def golden_queries():
    """Skips rather than errors when the table has not been built yet.

    These guard the populated set, so on a clean checkout, before
    ingestion and dataset generation have run, there is nothing to guard.
    """
    try:
        return await dbm.get_golden_queries(DB_PATH)
    except sqlite3.OperationalError:
        pytest.skip("golden_queries not populated; run ingestion and dataset generation first")


async def test_no_placeholder_labels_remain(golden_queries):
    unlabelled = [golden_query["query_id"] for golden_query in golden_queries if golden_query["human_reasoning"] == "PENDING_HUMAN_LABEL"]
    assert unlabelled == []


async def test_scores_span_the_range(golden_queries):
    """A set clustered at one end teaches the Judge only that end."""
    scores = [golden_query["human_score"] for golden_query in golden_queries]
    assert min(scores) <= 30
    assert max(scores) >= 90


async def test_ten_good_ten_bad(golden_queries):
    flags = [golden_query["is_good"] for golden_query in golden_queries]
    assert flags.count(1) == 10
    assert flags.count(0) == 10


async def test_every_quadrant_has_both_polarities(golden_queries):
    """build_prefix() shows only one quadrant's 5 exemplars, so each
    quadrant must carry a low anchor of its own."""
    by_quadrant: dict[str, list[int]] = {}
    for golden_query in golden_queries:
        by_quadrant.setdefault(golden_query["quadrant"], []).append(golden_query["is_good"])
    assert len(by_quadrant) == 4
    for quadrant, flags in by_quadrant.items():
        assert 1 in flags, quadrant
        assert 0 in flags, quadrant


async def test_bad_exemplars_differ_from_ground_truth(golden_queries):
    """A 'bad' exemplar whose output equals the ground truth is incoherent."""
    for golden_query in golden_queries:
        if golden_query["is_good"] == 0:
            assert golden_query["example_output"] != golden_query["ground_truth_answer"], golden_query["query_id"]


async def test_reasoning_is_substantive(golden_queries):
    for golden_query in golden_queries:
        assert len(golden_query["human_reasoning"].split()) >= 6, golden_query["query_id"]
