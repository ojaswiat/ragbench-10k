"""Read/update helpers on database_manager for the Judge.

Additive: upsert_result stays INSERT-only, so per-row scoring needs its own
UPDATE paths, and the gate needs a reader that joins each JEQ results row to
its quadrant and ground truth in judge_validation.
"""
import os

import pytest

import database_manager as dbm

TEST_DB = "test_benchmark_judge.db"


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


def _jeq(query_id, quadrant, doc="SEC_10K_AAPL_2025"):
    return {
        "query_id": query_id,
        "quadrant": quadrant,
        "query_text": f"question for {query_id}?",
        "ground_truth_answer": "$394.3B",
        "gt_citations": ["AAPL_2025_n0421"],
        "document_id": doc,
    }


def _result_row(result_id, query_id, pipeline, k_value=5):
    return {
        "result_id": result_id,
        "source_set": "JEQ",
        "query_id": query_id,
        "pipeline": pipeline,
        "k_value": k_value,
        "retrieved_node_ids": ["AAPL_2025_n0421", "AAPL_2025_n0420"],
        "pipeline_output": "Net sales were $394.3B. [[node:AAPL_2025_n0421]]",
        "cited_node_ids": ["AAPL_2025_n0421"],
    }


def _golden(query_id, quadrant):
    return {
        "query_id": query_id,
        "quadrant": quadrant,
        "query_text": f"gq {query_id}?",
        "ground_truth_answer": "$1B",
        "gt_citations": ["n1"],
        "example_output": "It was $1B. [[node:n1]]",
        "human_score": 90,
        "human_reasoning": "correct figure and citation.",
        "document_id": "SEC_10K_AAPL_2025",
    }


# --- get_jeq_judging_rows: results JEQ rows enriched from judge_validation ---


@pytest.mark.asyncio
async def test_get_jeq_judging_rows_joins_ground_truth():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))

    rows = await dbm.get_jeq_judging_rows(TEST_DB)

    assert len(rows) == 1
    row = rows[0]
    # results columns, JSON decoded to list[str]
    assert row["result_id"] == "R1"
    assert row["pipeline"] == "P1_vector"
    assert row["retrieved_node_ids"] == ["AAPL_2025_n0421", "AAPL_2025_n0420"]
    assert row["cited_node_ids"] == ["AAPL_2025_n0421"]
    # judge_validation columns joined in
    assert row["quadrant"] == "Q3_Direct_Table"
    assert row["ground_truth_answer"] == "$394.3B"
    assert row["gt_citations"] == ["AAPL_2025_n0421"]


@pytest.mark.asyncio
async def test_get_jeq_judging_rows_excludes_pq():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))
    pq_row = _result_row("R2", "PQ_001", "P1_vector")
    pq_row["source_set"] = "PQ"
    await dbm.upsert_result(TEST_DB, pq_row)

    rows = await dbm.get_jeq_judging_rows(TEST_DB)
    assert {stored_row["result_id"] for stored_row in rows} == {"R1"}


# --- get_golden_queries_by_quadrant: exactly the quadrant's exemplars ---


@pytest.mark.asyncio
async def test_get_golden_queries_by_quadrant_filters():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, _golden("GQ_T3_1", "Q3_Direct_Table"))
    await dbm.insert_golden_query(TEST_DB, _golden("GQ_T3_2", "Q3_Direct_Table"))
    await dbm.insert_golden_query(TEST_DB, _golden("GQ_T1_1", "Q1_Direct_Text"))

    rows = await dbm.get_golden_queries_by_quadrant(TEST_DB, "Q3_Direct_Table")

    assert {stored_row["query_id"] for stored_row in rows} == {"GQ_T3_1", "GQ_T3_2"}
    assert rows[0]["gt_citations"] == ["n1"]  # JSON decoded


# --- update_result_scores: judge_score + deterministic metric columns ---


@pytest.mark.asyncio
async def test_update_result_scores_writes_all_columns():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))

    await dbm.update_result_scores(
        TEST_DB,
        "R1",
        {
            "precision_at_k": 0.2,
            "recall_at_k": 1.0,
            "evidence_hit": 1,
            "citation_match": 1,
            "token_f1": 0.95,
            "exact_match": 1,
            "judge_score": 5,
        },
    )

    rows = await dbm.get_results(TEST_DB, "JEQ")
    row = next(stored_row for stored_row in rows if stored_row["result_id"] == "R1")
    assert row["precision_at_k"] == 0.2
    assert row["recall_at_k"] == 1.0
    assert row["evidence_hit"] == 1
    assert row["citation_match"] == 1
    assert row["token_f1"] == 0.95
    assert row["exact_match"] == 1
    assert row["judge_score"] == 5


@pytest.mark.asyncio
async def test_update_result_scores_allows_null_exact_match():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_002", "Q2_Implicit_Text"))
    await dbm.upsert_result(TEST_DB, _result_row("R9", "JEQ_002", "P2_bm25"))

    await dbm.update_result_scores(TEST_DB, "R9", {"exact_match": None, "judge_score": 4})

    rows = await dbm.get_results(TEST_DB, "JEQ")
    row = next(stored_row for stored_row in rows if stored_row["result_id"] == "R9")
    assert row["exact_match"] is None
    assert row["judge_score"] == 4


@pytest.mark.asyncio
async def test_update_result_scores_rejects_unknown_column():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))

    with pytest.raises(ValueError, match="unknown"):
        await dbm.update_result_scores(TEST_DB, "R1", {"source_set": "PQ"})


@pytest.mark.asyncio
async def test_update_result_scores_unknown_result_id_raises():
    await dbm.init_db(TEST_DB)
    with pytest.raises(ValueError, match="no results row"):
        await dbm.update_result_scores(TEST_DB, "NOPE", {"judge_score": 5})


# --- update_result_human_score ---


@pytest.mark.asyncio
async def test_update_result_human_score():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))

    await dbm.update_result_human_score(TEST_DB, "R1", 4)

    rows = await dbm.get_results(TEST_DB, "JEQ")
    assert next(stored_row for stored_row in rows if stored_row["result_id"] == "R1")["human_score"] == 4


@pytest.mark.asyncio
async def test_update_result_human_score_rejects_out_of_range():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, _jeq("JEQ_001", "Q3_Direct_Table"))
    await dbm.upsert_result(TEST_DB, _result_row("R1", "JEQ_001", "P1_vector"))

    with pytest.raises(ValueError, match="1.*5"):
        await dbm.update_result_human_score(TEST_DB, "R1", 6)


# --- results score CHECK migration (1-10 -> 1-5) ---


async def _build_under_old_schema(db_path):
    """A results table carrying the retired 1-10 CHECK, with one real row."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(dbm._SCHEMA.replace("BETWEEN 1 AND 5", "BETWEEN 1 AND 10"))
    conn.commit()
    conn.close()
    await dbm.insert_judge_validation(db_path, {
        "query_id": "JEQ_001", "quadrant": "Q3_Direct_Table", "query_text": "q?",
        "ground_truth_answer": "$1", "gt_citations": ["n1"], "document_id": "D1",
    })
    await dbm.upsert_result(db_path, {
        "result_id": "R1", "source_set": "JEQ", "query_id": "JEQ_001",
        "pipeline": "P1_vector", "k_value": 5, "retrieved_node_ids": ["n1"],
        "pipeline_output": "x", "cited_node_ids": ["n1"],
    })


async def _results_ddl(db_path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='results'"
        ).fetchone()[0]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_init_db_tightens_results_score_checks_and_keeps_rows(tmp_path):
    """SQLite cannot alter a CHECK and CREATE TABLE IF NOT EXISTS never touches
    an existing table, so a database built before the 1-5 collapse would keep a
    constraint two bands wider than the scale. The rebuild must preserve the
    rows, since 1-5 is a subset of 1-10 and every stored score is still valid."""
    db_path = str(tmp_path / "old.db")
    await _build_under_old_schema(db_path)
    await dbm.update_result_human_score(db_path, "R1", 4)
    assert "BETWEEN 1 AND 10" in await _results_ddl(db_path)

    await dbm.init_db(db_path)

    ddl = await _results_ddl(db_path)
    assert "judge_score BETWEEN 1 AND 5" in ddl
    assert "human_score BETWEEN 1 AND 5" in ddl
    rows = await dbm.get_results(db_path, "JEQ")
    assert len(rows) == 1
    assert rows[0]["human_score"] == 4


@pytest.mark.asyncio
async def test_results_migration_recreates_the_indexes(tmp_path):
    """DROP TABLE takes its indexes with it, and the crash-resume path queries
    by query_id and by (pipeline, k_value)."""
    import sqlite3

    db_path = str(tmp_path / "old.db")
    await _build_under_old_schema(db_path)
    await dbm.init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        names = {
            stored_row[0] for stored_row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='results' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()
    assert names == {"idx_results_query_id", "idx_results_pipeline_k"}


@pytest.mark.asyncio
async def test_results_migration_refuses_to_rebuild_over_retired_scale_scores(tmp_path):
    """A leftover 6-10 score predates the collapse. The rebuild would carry it
    into a table whose CHECK forbids it, so abort loudly instead."""
    import sqlite3

    db_path = str(tmp_path / "old.db")
    await _build_under_old_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE results SET judge_score = 9 WHERE result_id = 'R1'")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="retired 1-10 scale"):
        await dbm.init_db(db_path)


@pytest.mark.asyncio
async def test_results_migration_is_idempotent(tmp_path):
    db_path = str(tmp_path / "old.db")
    await _build_under_old_schema(db_path)
    await dbm.init_db(db_path)
    first = await _results_ddl(db_path)
    await dbm.init_db(db_path)
    assert await _results_ddl(db_path) == first
