import asyncio
import json
import os

import pytest

import database_manager as dbm

TEST_DB = "test_benchmark.db"


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


@pytest.mark.asyncio
async def test_init_db_creates_five_tables():
    await dbm.init_db(TEST_DB)
    import aiosqlite

    async with aiosqlite.connect(TEST_DB) as conn:
        cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        rows = await cursor.fetchall()
        table_names = {row[0] for row in rows}
    assert {"nodes", "queries", "golden_queries", "judge_validation", "results"} <= table_names


@pytest.mark.asyncio
async def test_wal_mode_enabled():
    await dbm.init_db(TEST_DB)
    import aiosqlite

    async with aiosqlite.connect(TEST_DB) as conn:
        cursor = await conn.execute("PRAGMA journal_mode;")
        (mode,) = await cursor.fetchone()
    assert mode.lower() == "wal"


_OLD_GOLDEN_QUERIES_SCHEMA = """
CREATE TABLE golden_queries (
    query_id            TEXT PRIMARY KEY,
    quadrant            TEXT NOT NULL,
    query_text          TEXT NOT NULL,
    ground_truth_answer TEXT NOT NULL,
    gt_citations        TEXT NOT NULL,
    example_output      TEXT NOT NULL,
    human_score         INTEGER NOT NULL CHECK (human_score BETWEEN 1 AND 10),
    human_reasoning     TEXT NOT NULL,
    document_id         TEXT NOT NULL
);
"""


@pytest.mark.asyncio
async def test_init_db_migrates_stale_golden_queries_schema_when_empty():
    import aiosqlite

    async with aiosqlite.connect(TEST_DB) as conn:
        await conn.executescript(_OLD_GOLDEN_QUERIES_SCHEMA)
        await conn.commit()

    await dbm.init_db(TEST_DB)

    async with aiosqlite.connect(TEST_DB) as conn:
        cursor = await conn.execute("PRAGMA table_info(golden_queries)")
        columns = {row[1] for row in await cursor.fetchall()}
    assert "is_good" in columns


@pytest.mark.asyncio
async def test_init_db_refuses_to_migrate_stale_golden_queries_schema_with_real_rows():
    import aiosqlite

    async with aiosqlite.connect(TEST_DB) as conn:
        await conn.executescript(_OLD_GOLDEN_QUERIES_SCHEMA)
        await conn.execute(
            """INSERT INTO golden_queries
               (query_id, quadrant, query_text, ground_truth_answer, gt_citations,
                example_output, human_score, human_reasoning, document_id)
               VALUES ('Q1_GQ_0001', 'Q1_Direct_Text', 'q', 'a', '[]', 'a', 8, 'r', 'AAPL_2023')""",
        )
        await conn.commit()

    with pytest.raises(RuntimeError, match="1 real row"):
        await dbm.init_db(TEST_DB)

    async with aiosqlite.connect(TEST_DB) as conn:
        cursor = await conn.execute("PRAGMA table_info(golden_queries)")
        columns = {row[1] for row in await cursor.fetchall()}
        cursor = await conn.execute("SELECT query_id FROM golden_queries")
        rows = await cursor.fetchall()
    assert "is_good" not in columns
    assert rows == [("Q1_GQ_0001",)]


@pytest.mark.asyncio
async def test_insert_and_get_node():
    await dbm.init_db(TEST_DB)
    node = {
        "node_id": "TEST_2025_n0001",
        "document_id": "SEC_10K_TEST_2025",
        "parent_item_header": "Item 8",
        "node_type": "table",
        "source_page_num": 12,
        "content": "| Year | Net Sales |\n|---|---|\n| 2025 | 100 |",
        "token_count": 20,
    }
    await dbm.insert_node(TEST_DB, node)
    fetched = await dbm.get_nodes_by_document(TEST_DB, "SEC_10K_TEST_2025")
    assert len(fetched) == 1
    assert fetched[0]["node_id"] == "TEST_2025_n0001"


@pytest.mark.asyncio
async def test_upsert_result_and_get_completed_keys():
    await dbm.init_db(TEST_DB)
    row = {
        "result_id": "R_TEST_001",
        "source_set": "PQ",
        "query_id": "Q1_001",
        "pipeline": "P1_vector",
        "k_value": 5,
        "retrieved_node_ids": ["TEST_2025_n0001"],
        "pipeline_output": "Net sales were 100. [[node:TEST_2025_n0001]]",
        "cited_node_ids": ["TEST_2025_n0001"],
        "precision_at_k": 1.0,
        "recall_at_k": 1.0,
        "evidence_hit": 1,
        "citation_match": 1,
        "token_f1": 0.9,
        "exact_match": 1,
        "judge_score": 4,
        "human_score": None,
        "latency_sec": 1.2,
        "input_tokens": 500,
        "output_tokens": 30,
    }
    await dbm.upsert_result(TEST_DB, row)
    completed = await dbm.get_completed_keys(TEST_DB, "PQ")
    assert ("Q1_001", "P1_vector", 5) in completed


@pytest.mark.asyncio
async def test_results_unique_constraint_blocks_duplicate():
    await dbm.init_db(TEST_DB)
    row = {
        "result_id": "R_TEST_002",
        "source_set": "PQ",
        "query_id": "Q1_002",
        "pipeline": "P2_bm25",
        "k_value": 3,
        "retrieved_node_ids": [],
        "pipeline_output": None,
        "cited_node_ids": None,
        "precision_at_k": None,
        "recall_at_k": None,
        "evidence_hit": None,
        "citation_match": None,
        "token_f1": None,
        "exact_match": None,
        "judge_score": None,
        "human_score": None,
        "latency_sec": None,
        "input_tokens": None,
        "output_tokens": None,
    }
    await dbm.upsert_result(TEST_DB, row)
    row["result_id"] = "R_TEST_003"  # different PK, same (source_set, query_id, pipeline, k_value)
    with pytest.raises(Exception):
        await dbm.upsert_result(TEST_DB, row)


@pytest.mark.asyncio
async def test_insert_query_and_quadrant_counts():
    await dbm.init_db(TEST_DB)
    await dbm.insert_query(TEST_DB, {
        "query_id": "Q1_TEST_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "document_id": "SEC_10K_TEST_2025",
    })
    counts = await dbm.get_quadrant_counts(TEST_DB, "queries")
    assert counts == {
        "Q1_Direct_Text": 1,
        "Q2_Implicit_Text": 0,
        "Q3_Direct_Table": 0,
        "Q4_Implicit_Table": 0,
    }


@pytest.mark.asyncio
async def test_get_quadrant_counts_rejects_unknown_table():
    await dbm.init_db(TEST_DB)
    with pytest.raises(ValueError):
        await dbm.get_quadrant_counts(TEST_DB, "results")


@pytest.mark.asyncio
async def test_insert_golden_query_and_update_labels():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })
    golden = await dbm.get_golden_queries(TEST_DB)
    assert len(golden) == 1
    assert golden[0]["gt_citations"] == ["TEST_2025_n0001"]
    assert golden[0]["human_reasoning"] == "PENDING_HUMAN_LABEL"

    await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_001", 8, "Clean, unambiguous fact retrieval.")
    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 8
    assert golden[0]["human_reasoning"] == "Clean, unambiguous fact retrieval."


@pytest.mark.asyncio
async def test_update_golden_query_labels_raises_on_unknown_query_id():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    with pytest.raises(ValueError, match="Q1_GQ_999"):
        await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_999", 9, "typo'd id")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert len(golden) == 1
    assert golden[0]["human_score"] == 1
    assert golden[0]["human_reasoning"] == "PENDING_HUMAN_LABEL"


@pytest.mark.asyncio
async def test_update_golden_query_labels_happy_path_no_raise():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_002",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_002", 7, "Solid citation match.")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 7
    assert golden[0]["human_reasoning"] == "Solid citation match."


@pytest.mark.asyncio
async def test_update_golden_query_labels_rejects_score_above_100():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_003",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    with pytest.raises(ValueError, match="150"):
        await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_003", 150, "too high")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 1
    assert golden[0]["human_reasoning"] == "PENDING_HUMAN_LABEL"


@pytest.mark.asyncio
async def test_update_golden_query_labels_rejects_negative_score():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_004",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    with pytest.raises(ValueError, match="-1"):
        await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_004", -1, "too low")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 1


@pytest.mark.asyncio
async def test_update_golden_query_labels_rejects_non_integer_score():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_005",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    with pytest.raises(ValueError):
        await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_005", 7.5, "non-integer")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 1


@pytest.mark.asyncio
async def test_update_golden_query_labels_rejects_invalid_is_good():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_006",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    with pytest.raises(ValueError, match="maybe"):
        await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_006", 50, "unsure", is_good="maybe")

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 1


@pytest.mark.asyncio
async def test_update_golden_query_labels_persists_valid_is_good():
    await dbm.init_db(TEST_DB)
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_007",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$100 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })

    await dbm.update_golden_query_labels(TEST_DB, "Q1_GQ_007", 90, "great exemplar", is_good=True)

    golden = await dbm.get_golden_queries(TEST_DB)
    assert golden[0]["human_score"] == 90
    assert golden[0]["is_good"] == 1


@pytest.mark.asyncio
async def test_get_all_query_texts_returns_rows_from_all_three_tables():
    await dbm.init_db(TEST_DB)
    await dbm.insert_query(TEST_DB, {
        "query_id": "Q1_PQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "document_id": "SEC_10K_TEST_2025",
    })
    await dbm.insert_golden_query(TEST_DB, {
        "query_id": "Q1_GQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What was net income?",
        "ground_truth_answer": "$50 million",
        "gt_citations": ["TEST_2025_n0001"],
        "example_output": "$50 million",
        "human_score": 1,
        "human_reasoning": "PENDING_HUMAN_LABEL",
        "document_id": "SEC_10K_TEST_2025",
    })
    await dbm.insert_judge_validation(TEST_DB, {
        "query_id": "Q1_JEQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What was total assets?",
        "ground_truth_answer": "$200 million",
        "gt_citations": ["TEST_2025_n0001"],
        "document_id": "SEC_10K_TEST_2025",
    })

    all_texts = await dbm.get_all_query_texts(TEST_DB)

    assert set(all_texts) == {
        ("Q1_PQ_001", "What is the total revenue?"),
        ("Q1_GQ_001", "What was net income?"),
        ("Q1_JEQ_001", "What was total assets?"),
    }


@pytest.mark.asyncio
async def test_insert_judge_validation():
    await dbm.init_db(TEST_DB)
    await dbm.insert_judge_validation(TEST_DB, {
        "query_id": "Q1_JEQ_001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What is the total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["TEST_2025_n0001"],
        "document_id": "SEC_10K_TEST_2025",
    })
    counts = await dbm.get_quadrant_counts(TEST_DB, "judge_validation")
    assert counts["Q1_Direct_Text"] == 1
