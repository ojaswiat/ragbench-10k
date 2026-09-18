"""Generalised judging over any source set, with a rubric-aware resume.

The resume rule is the point of this module. A resume keyed on "this row has
a score" cannot distinguish a score produced by the current rubric from one
produced by a retired rubric, and when the scale moved from 1-10 to 1-5 the
retired scores that happened to land in 1-5 stayed individually plausible.
Keeping them would have mixed two rubrics inside one reported figure without
anything visibly failing, so the tests below assert re-judging on a stale
fingerprint directly rather than inferring it from a row count.
"""
import os
import sqlite3

import pytest

from core import database_manager as dbm
from judge.async_judge import judge_rows, rubric_fingerprint

TEST_DB = "test_benchmark_judge_resume.db"

QUADRANT = "Q1_Direct_Text"


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Retry backoff is real time; the tests assert the retry, not the wait."""
    monkeypatch.setattr("judge.async_judge._RETRY_BASE_DELAY_SEC", 0.0)


@pytest.fixture(autouse=True)
def clean_db():
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(TEST_DB + suffix):
            os.remove(TEST_DB + suffix)
    yield
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(TEST_DB + suffix):
            os.remove(TEST_DB + suffix)


def _query_row(query_id):
    return {"query_id": query_id, "quadrant": QUADRANT,
            "query_text": f"question for {query_id}?", "ground_truth_answer": "$394.3B",
            "gt_citations": ["n1"], "document_id": "AAPL_2025"}


def _result(result_id, query_id, source_set):
    return {"result_id": result_id, "source_set": source_set, "query_id": query_id,
            "pipeline": "P1_vector", "k_value": 5, "retrieved_node_ids": ["n1"],
            "pipeline_output": "Net sales were $394.3B. [[node:n1]]", "cited_node_ids": ["n1"]}


def _golden(query_id, output="It was $1B. [[node:n1]]"):
    return {"query_id": query_id, "quadrant": QUADRANT, "query_text": "gq?",
            "ground_truth_answer": "$1B", "gt_citations": ["n1"], "example_output": output,
            "human_score": 90, "human_reasoning": "correct.", "document_id": "AAPL_2025"}


class FakeJudge:
    """Records which rows it was asked to score and returns a fixed band."""

    def __init__(self, score=4, model="qwen/qwen3.6-27b", fail_times=0):
        self.model = model
        self.scored: list[str] = []
        self._score = score
        self._fail_times = fail_times
        self.calls = 0

    async def score_row(self, row, system_prompt):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TimeoutError()
        self.scored.append(row["result_id"])
        return self._score


async def _seed(source_set="PQ", row_count=2):
    await dbm.init_db(TEST_DB)
    table = "queries" if source_set == "PQ" else "judge_validation"
    insert = dbm.insert_query if source_set == "PQ" else dbm.insert_judge_validation
    for index in range(row_count):
        await insert(TEST_DB, _query_row(f"Q{index}"))
        await dbm.upsert_result(TEST_DB, _result(f"R{index}", f"Q{index}", source_set))
    await dbm.insert_golden_query(TEST_DB, _golden("G1"))


def _stored(column="judge_fingerprint"):
    conn = sqlite3.connect(TEST_DB)
    rows = dict(conn.execute(f"SELECT result_id, {column} FROM results"))
    conn.close()
    return rows


# --- generalisation over source sets ------------------------------------------

@pytest.mark.asyncio
async def test_pq_rows_join_their_ground_truth_in_queries():
    await _seed("PQ")
    rows = await dbm.get_judging_rows(TEST_DB, "PQ")
    assert len(rows) == 2
    assert all(row["ground_truth_answer"] == "$394.3B" for row in rows)
    assert all(row["quadrant"] == QUADRANT for row in rows)


@pytest.mark.asyncio
async def test_a_source_set_never_sees_another_source_sets_rows():
    await _seed("PQ")
    assert await dbm.get_judging_rows(TEST_DB, "JEQ") == []


@pytest.mark.asyncio
async def test_an_unknown_source_set_is_rejected_rather_than_returning_nothing():
    await _seed("PQ")
    with pytest.raises(ValueError, match="source_set"):
        await dbm.get_judging_rows(TEST_DB, "GQ")


# --- the fingerprint ----------------------------------------------------------

def test_the_fingerprint_changes_when_the_rubric_text_changes():
    fingerprint_a = rubric_fingerprint("rubric A\nexemplars", "m")
    fingerprint_b = rubric_fingerprint("rubric B\nexemplars", "m")
    assert fingerprint_a != fingerprint_b


def test_the_fingerprint_changes_when_the_model_changes():
    assert rubric_fingerprint("same prompt", "model-a") != rubric_fingerprint("same prompt", "model-b")


def test_the_fingerprint_is_stable_for_identical_inputs():
    assert rubric_fingerprint("p", "m") == rubric_fingerprint("p", "m")


@pytest.mark.asyncio
async def test_changing_an_exemplar_changes_the_fingerprint_the_run_records():
    await _seed("PQ", row_count=1)
    first = await judge_rows(TEST_DB, "PQ", FakeJudge())
    assert first["judged"] == 1
    before = _stored()["R0"]

    conn = sqlite3.connect(TEST_DB)
    conn.execute("UPDATE golden_queries SET example_output = 'a different exemplar answer'")
    conn.commit()
    conn.close()

    judge = FakeJudge()
    await judge_rows(TEST_DB, "PQ", judge)
    assert judge.scored == ["R0"], "an exemplar change must invalidate the stored score"
    assert _stored()["R0"] != before


# --- resume -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_fingerprint_is_written_alongside_the_score():
    await _seed("PQ", row_count=1)
    await judge_rows(TEST_DB, "PQ", FakeJudge())
    assert _stored()["R0"]
    assert _stored("judge_score")["R0"] == 4


@pytest.mark.asyncio
async def test_resume_skips_rows_already_scored_under_the_current_rubric():
    await _seed("PQ", row_count=2)
    await judge_rows(TEST_DB, "PQ", FakeJudge())

    second = FakeJudge()
    summary = await judge_rows(TEST_DB, "PQ", second)
    assert second.scored == []
    assert summary["judged"] == 0
    assert summary["skipped"] == 2


@pytest.mark.asyncio
async def test_resume_rejudges_a_row_whose_score_came_from_a_retired_rubric():
    """The failure this whole mechanism exists to prevent."""
    await _seed("PQ", row_count=2)
    await judge_rows(TEST_DB, "PQ", FakeJudge(score=4))

    # A score that is individually plausible on the current scale, but was
    # produced by a rubric that is no longer in use.
    conn = sqlite3.connect(TEST_DB)
    conn.execute("UPDATE results SET judge_fingerprint = 'retired_rubric' WHERE result_id = 'R0'")
    conn.commit()
    conn.close()

    judge = FakeJudge(score=2)
    summary = await judge_rows(TEST_DB, "PQ", judge)
    assert judge.scored == ["R0"], "a stale-rubric row must be re-judged, not kept"
    assert summary["skipped"] == 1
    assert _stored("judge_score")["R0"] == 2


@pytest.mark.asyncio
async def test_resume_rejudges_a_score_with_no_recorded_fingerprint():
    await _seed("PQ", row_count=1)
    await judge_rows(TEST_DB, "PQ", FakeJudge())
    conn = sqlite3.connect(TEST_DB)
    conn.execute("UPDATE results SET judge_fingerprint = NULL")
    conn.commit()
    conn.close()

    judge = FakeJudge()
    await judge_rows(TEST_DB, "PQ", judge)
    assert judge.scored == ["R0"], "unknown provenance must not be treated as current"


@pytest.mark.asyncio
async def test_resume_off_rejudges_everything():
    await _seed("PQ", row_count=2)
    await judge_rows(TEST_DB, "PQ", FakeJudge())
    judge = FakeJudge()
    await judge_rows(TEST_DB, "PQ", judge, resume=False)
    assert sorted(judge.scored) == ["R0", "R1"]


# --- retry --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_rather_than_lost():
    await _seed("PQ", row_count=1)
    judge = FakeJudge(fail_times=2)
    summary = await judge_rows(TEST_DB, "PQ", judge, attempts=3)
    assert summary["judged"] == 1
    assert summary["failures"] == []
    assert judge.calls == 3


@pytest.mark.asyncio
async def test_a_row_that_fails_every_attempt_is_reported_and_left_unscored():
    await _seed("PQ", row_count=1)
    judge = FakeJudge(fail_times=99)
    summary = await judge_rows(TEST_DB, "PQ", judge, attempts=2)
    assert summary["judged"] == 0
    assert len(summary["failures"]) == 1
    assert "TimeoutError" in summary["failures"][0]["error"]
    assert _stored("judge_score")["R0"] is None


@pytest.mark.asyncio
async def test_a_failed_row_is_picked_up_by_the_next_resume():
    await _seed("PQ", row_count=1)
    await judge_rows(TEST_DB, "PQ", FakeJudge(fail_times=99), attempts=1)
    judge = FakeJudge()
    summary = await judge_rows(TEST_DB, "PQ", judge)
    assert judge.scored == ["R0"]
    assert summary["judged"] == 1


# --- migration ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_database_without_the_column_gains_it_without_losing_rows():
    await _seed("PQ", row_count=1)
    conn = sqlite3.connect(TEST_DB)
    conn.execute("ALTER TABLE results DROP COLUMN judge_fingerprint")
    conn.commit()
    assert "judge_fingerprint" not in {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    conn.close()

    await dbm.init_db(TEST_DB)

    conn = sqlite3.connect(TEST_DB)
    assert "judge_fingerprint" in {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 1
    conn.close()
