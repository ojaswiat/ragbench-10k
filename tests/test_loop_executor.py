"""loop_executor: cell construction, resume-skip, throttle, row shape.

Uses a stub Retriever (the real ABC, sync retrieve()) and a stub Answerer
against a real temporary SQLite database, so the resume path is asserted
through the actual UNIQUE constraint and the actual get_completed_keys()
query rather than a mock of them.
"""
import importlib
from unittest.mock import AsyncMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from core import database_manager as dbm
import llm_client.config as config
from benchmark import loop_executor
from core import loop_template
from pipelines.answerer import AnswerResult
from pipelines.base import Retriever


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def throttle_off(monkeypatch):
    """Default the suite to unthrottled; the throttle tests opt back in."""
    monkeypatch.setattr(config, "LOCAL_TEST_THROTTLE", False)
    importlib.reload(loop_template)
    importlib.reload(loop_executor)
    yield
    monkeypatch.setattr(config, "LOCAL_TEST_THROTTLE", False)
    importlib.reload(loop_template)
    importlib.reload(loop_executor)


class StubRetriever(Retriever):
    def __init__(self, node_ids=("N1", "N2")):
        self._node_ids = node_ids
        self.calls = []

    def retrieve(self, query_text: str, document_id: str, k: int) -> list[NodeWithScore]:
        self.calls.append((query_text, document_id, k))
        return [
            NodeWithScore(node=TextNode(id_=node_id, text=f"content of {node_id}"), score=1.0)
            for node_id in self._node_ids[:k]
        ]


class StubAnswerer:
    def __init__(self, raw_text="Claim. [[node:N1]]"):
        self._raw_text = raw_text
        self.calls = []

    async def answer(self, query_text, nodes):
        self.calls.append((query_text, [node.node.node_id for node in nodes]))
        return AnswerResult(
            raw_text=self._raw_text,
            cited_node_ids=["N1"],
            input_tokens=1800,
            output_tokens=22,
            latency_sec=0.91,
        )


def _query(query_id: str, quadrant: str = "Q1_Direct_Text") -> dict:
    return {
        "query_id": query_id,
        "quadrant": quadrant,
        "query_text": f"question for {query_id}",
        "ground_truth_answer": "gt",
        "gt_citations": ["N1"],
        "document_id": "SEC_10K_AAPL_2025",
    }


async def _seed(db_path: str, query_ids, table="queries"):
    await dbm.init_db(db_path)
    inserter = dbm.insert_query if table == "queries" else dbm.insert_judge_validation
    for query_id in query_ids:
        await inserter(db_path, _query(query_id))


# --- build_cells ---


def test_build_cells_is_query_major_across_pipelines():
    queries = [_query("Q_A"), _query("Q_B")]
    cells = loop_executor.build_cells(queries, ("P1_vector", "P2_bm25"), (3, 5), set())

    assert len(cells) == 8
    assert [cell["query"]["query_id"] for cell in cells[:4]] == ["Q_A"] * 4
    assert [(cell["pipeline"], cell["k_value"]) for cell in cells[:4]] == [
        ("P1_vector", 3), ("P1_vector", 5), ("P2_bm25", 3), ("P2_bm25", 5),
    ]


def test_build_cells_skips_completed_keys():
    queries = [_query("Q_A")]
    completed = {("Q_A", "P1_vector", 3), ("Q_A", "P2_bm25", 2)}
    cells = loop_executor.build_cells(queries, ("P1_vector", "P2_bm25"), (2, 3, 5), completed)

    assert ("Q_A", "P1_vector", 3) not in {(cell["query"]["query_id"], cell["pipeline"], cell["k_value"]) for cell in cells}
    assert len(cells) == 4


def test_result_id_is_derived_from_the_natural_key():
    assert loop_executor.result_id("PQ", "QT3_PQ_017", "P1_vector", 5) == "R_PQ_QT3_PQ_017_P1_vector_K5"


# --- run_cell row shape ---


@pytest.mark.asyncio
async def test_run_cell_writes_a_row_with_judge_columns_null(db_path):
    await _seed(db_path, ["QT1_PQ_001"])
    retriever = StubRetriever()
    cell = {"query": _query("QT1_PQ_001"), "pipeline": "P2_bm25", "k_value": 5}

    row = await loop_executor.run_cell(db_path, "PQ", cell, retriever, StubAnswerer())

    assert row["retrieved_node_ids"] == ["N1", "N2"]
    assert row["cited_node_ids"] == ["N1"]
    assert row["input_tokens"] == 1800
    assert row["output_tokens"] == 22
    assert row["latency_sec"] == 0.91

    stored = await dbm.get_completed_keys(db_path, "PQ")
    assert stored == {("QT1_PQ_001", "P2_bm25", 5)}

    import aiosqlite

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT * FROM results")
        saved = dict(await cursor.fetchone())

    for judge_column in (
        "precision_at_k", "recall_at_k", "evidence_hit", "citation_match",
        "token_f1", "exact_match", "judge_score", "human_score",
    ):
        assert saved[judge_column] is None, judge_column
    assert saved["retrieved_node_ids"] == '["N1", "N2"]'
    assert "[[node:N1]]" in saved["pipeline_output"]


@pytest.mark.asyncio
async def test_run_cell_passes_document_id_and_k_to_the_retriever(db_path):
    await _seed(db_path, ["QT1_PQ_001"])
    retriever = StubRetriever()
    cell = {"query": _query("QT1_PQ_001"), "pipeline": "P1_vector", "k_value": 3}

    await loop_executor.run_cell(db_path, "PQ", cell, retriever, StubAnswerer())

    assert retriever.calls == [("question for QT1_PQ_001", "SEC_10K_AAPL_2025", 3)]


@pytest.mark.asyncio
async def test_run_cell_tolerates_a_sync_retriever_that_uses_asyncio_run(db_path):
    """P2 drives its DB read with asyncio.run() inside sync retrieve(), which
    raises if a loop is already running on the calling thread. run_cell must
    off-load to a worker thread so that stays legal."""
    import asyncio

    await _seed(db_path, ["QT1_PQ_001"])

    class AsyncioRunRetriever(Retriever):
        def retrieve(self, query_text, document_id, k):
            async def _noop():
                return None

            asyncio.run(_noop())
            return [NodeWithScore(node=TextNode(id_="N1", text="t"), score=1.0)]

    cell = {"query": _query("QT1_PQ_001"), "pipeline": "P2_bm25", "k_value": 3}
    row = await loop_executor.run_cell(db_path, "PQ", cell, AsyncioRunRetriever(), StubAnswerer())
    assert row["retrieved_node_ids"] == ["N1"]


# --- main: resume, throttle, failure isolation ---


@pytest.mark.asyncio
async def test_main_runs_every_cell_for_one_pipeline(db_path):
    await _seed(db_path, ["QT1_PQ_001", "QT1_PQ_002"])
    answerer = StubAnswerer()

    summary = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=answerer,
    )

    assert summary["outstanding"] == 6  # 2 queries x 1 pipeline x 3 K values
    assert summary["completed"] == 6
    assert summary["failures"] == []
    assert len(answerer.calls) == 6


@pytest.mark.asyncio
async def test_main_skips_already_completed_cells_on_a_second_run(db_path):
    await _seed(db_path, ["QT1_PQ_001"])
    retrievers = {"P2_bm25": StubRetriever()}

    first = await loop_executor.main(
        db_path=db_path, source_set="PQ", retrievers=retrievers, answerer=StubAnswerer()
    )
    assert first["completed"] == 3

    second_answerer = StubAnswerer()
    second = await loop_executor.main(
        db_path=db_path, source_set="PQ", retrievers=retrievers, answerer=second_answerer,
    )

    assert second["outstanding"] == 0
    assert second["attempted"] == 0
    assert second_answerer.calls == [], "a resumed run must not re-spend LLM quota"


@pytest.mark.asyncio
async def test_main_resumes_only_the_missing_cells(db_path):
    await _seed(db_path, ["QT1_PQ_001"])
    await loop_executor.run_cell(
        db_path, "PQ",
        {"query": _query("QT1_PQ_001"), "pipeline": "P2_bm25", "k_value": 5},
        StubRetriever(), StubAnswerer(),
    )

    answerer = StubAnswerer()
    summary = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=answerer,
    )

    assert summary["outstanding"] == 2
    assert summary["completed"] == 2


@pytest.mark.asyncio
async def test_main_separates_pq_and_jeq_source_sets(db_path):
    await _seed(db_path, ["QT1_PQ_001"], table="queries")
    await _seed(db_path, ["QT1_JEQ_002"], table="judge_validation")

    pq_result = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=StubAnswerer(),
    )
    jeq = await loop_executor.main(
        db_path=db_path, source_set="JEQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=StubAnswerer(), k_values=(5,),
    )

    assert pq_result["completed"] == 3
    assert jeq["completed"] == 1
    assert await dbm.get_completed_keys(db_path, "JEQ") == {("QT1_JEQ_002", "P2_bm25", 5)}


@pytest.mark.asyncio
async def test_local_test_throttle_caps_the_run_at_three_cells(db_path, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_TEST_THROTTLE", True)
    importlib.reload(loop_template)
    importlib.reload(loop_executor)

    await _seed(db_path, ["QT1_PQ_001", "QT1_PQ_002", "QT1_PQ_003", "QT1_PQ_004"])
    answerer = StubAnswerer()

    summary = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P1_vector": StubRetriever(), "P2_bm25": StubRetriever()},
        answerer=answerer,
    )

    assert summary["outstanding"] == 24
    assert summary["attempted"] == config.THROTTLE_LIMIT == 3
    assert summary["completed"] == 3
    assert len(answerer.calls) == 3


@pytest.mark.asyncio
async def test_unthrottled_run_is_not_capped(db_path):
    await _seed(db_path, ["QT1_PQ_001", "QT1_PQ_002"])
    summary = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=StubAnswerer(),
    )
    assert summary["completed"] == 6


@pytest.mark.asyncio
async def test_a_failing_cell_does_not_abort_the_run(db_path):
    await _seed(db_path, ["QT1_PQ_001"])

    answerer = StubAnswerer()
    answerer.answer = AsyncMock(side_effect=[
        RuntimeError("groq exploded"),
        AnswerResult("ok [[node:N1]]", ["N1"], 10, 2, 0.1),
        AnswerResult("ok [[node:N1]]", ["N1"], 10, 2, 0.1),
    ])

    summary = await loop_executor.main(
        db_path=db_path, source_set="PQ",
        retrievers={"P2_bm25": StubRetriever()}, answerer=answerer,
    )

    assert summary["completed"] == 2
    assert len(summary["failures"]) == 1
    assert summary["failures"][0]["exception_type"] == "RuntimeError"
    # The failed key was never written, so a later run retries it. The failing
    # cell is the first one built, which is the lowest K in K_VALUES.
    failed_k = loop_executor.K_VALUES[0]
    assert ("QT1_PQ_001", "P2_bm25", failed_k) not in await dbm.get_completed_keys(db_path, "PQ")


# --- guards ---


@pytest.mark.asyncio
async def test_main_rejects_an_empty_retriever_map(db_path):
    with pytest.raises(ValueError, match="at least one retriever"):
        await loop_executor.main(db_path=db_path, source_set="PQ", retrievers={})


@pytest.mark.asyncio
async def test_main_rejects_an_unknown_pipeline_name(db_path):
    with pytest.raises(ValueError, match="Unknown pipeline"):
        await loop_executor.main(
            db_path=db_path, source_set="PQ", retrievers={"P4_magic": StubRetriever()},
        )


@pytest.mark.asyncio
async def test_get_queries_rejects_the_exemplar_pool(db_path):
    """golden_queries is the Judge's exemplar set and must stay unreachable
    from any pipeline-facing code path (anti-leakage)."""
    await dbm.init_db(db_path)
    with pytest.raises(ValueError):
        await dbm.get_queries(db_path, "GQ")
