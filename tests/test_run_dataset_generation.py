import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from dataset_generation import run_dataset_generation as rdg
from dataset_generation.run_dataset_generation import (
    main,
    next_target,
    classify_section,
    chunk_section,
    build_pools,
    _round_robin_interleave,
    format_underfill_summary,
    _QUADRANTS,
    _TEXT_QUADRANTS,
    _TABLE_QUADRANTS,
    _TARGETS,
    _TABLE_ORDER,
)


@pytest.fixture(autouse=True)
def _isolate_progress_log(monkeypatch, tmp_path):
    """Every test here calls main(), which calls _configure_logging() on entry.
    _configure_logging is a no-op once logger.handlers is non-empty, so
    without isolation the first test to run would attach handlers pointing at
    the real logs/dataset_generation_progress.log and every later test would
    reuse them. Point PROGRESS_LOG_PATH at a per-test tmp_path and clear
    handlers around each test."""
    monkeypatch.setattr(rdg, "PROGRESS_LOG_PATH", tmp_path / "progress.log")
    rdg.logger.handlers.clear()
    yield
    rdg.logger.handlers.clear()


def _empty_counts():
    return {
        table: {quadrant_name: 0 for quadrant_name in _QUADRANTS}
        for table in ("queries", "golden_queries", "judge_validation")
    }


def test_next_target_fills_queries_before_golden_before_judge():
    counts = _empty_counts()
    assert next_target(counts) == ("queries", "Q1_Direct_Text")


def test_next_target_moves_to_golden_queries_once_pq_full():
    counts = _empty_counts()
    counts["queries"]["Q1_Direct_Text"] = _TARGETS["queries"]
    assert next_target(counts) == ("golden_queries", "Q1_Direct_Text")


def test_next_target_moves_to_next_quadrant_once_all_three_full():
    counts = _empty_counts()
    counts["queries"]["Q1_Direct_Text"] = _TARGETS["queries"]
    counts["golden_queries"]["Q1_Direct_Text"] = _TARGETS["golden_queries"]
    counts["judge_validation"]["Q1_Direct_Text"] = _TARGETS["judge_validation"]
    assert next_target(counts) == ("queries", "Q2_Implicit_Text")


def test_next_target_returns_none_when_all_140_filled():
    counts = _empty_counts()
    for table in counts:
        for quadrant_name in _QUADRANTS:
            counts[table][quadrant_name] = _TARGETS[table]
    assert next_target(counts) is None


@pytest.mark.asyncio
async def test_main_smoke_runs_under_throttle(monkeypatch):
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    fake_nodes = [
        {
            "node_id": "n1",
            "document_id": "DOC_A",
            "parent_item_header": "Item 1A. Risk Factors",
            "node_type": "text",
            "source_page_num": 1,
            "content": "Total revenue was $100 million.",
            "token_count": 10,
        }
    ]

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=fake_nodes),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value={
                "query_text": "What was total revenue?",
                "ground_truth_answer": "$100 million",
                "gt_citations": ["n1"],
                "quadrant": "Q1_Direct_Text",
                "document_id": "DOC_A",
            }),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_after_restart_does_not_collide_with_existing_query_ids(monkeypatch):
    """Simulates process A having already committed 2 accepted queries into
    (queries, Q1_Direct_Text) before crashing. Process B restarts main(), and
    the newly generated query_id must be QT1_PQ_003, not a collision with the _001/_002 rows process A already committed.
    """
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    fake_nodes = [
        {
            "node_id": "n1",
            "document_id": "DOC_A",
            "parent_item_header": "Item 1A. Risk Factors",
            "node_type": "text",
            "source_page_num": 1,
            "content": "Total revenue was $100 million.",
            "token_count": 10,
        }
    ]

    pre_existing_counts = {quadrant_name: 0 for quadrant_name in _QUADRANTS}
    pre_existing_counts["Q1_Direct_Text"] = 2  # process A already committed 2 rows

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value=pre_existing_counts),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=fake_nodes),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value={
                "query_text": "What was total revenue?",
                "ground_truth_answer": "$100 million",
                "gt_citations": ["n1"],
                "quadrant": "Q1_Direct_Text",
                "document_id": "DOC_A",
            }),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_awaited_once()
    inserted_row = mock_insert.await_args.args[1]
    assert inserted_row["query_id"] == "QT1_PQ_003"


# --- Empty document_ids tuple must not fall back to _ALL_FILINGS ------------


@pytest.mark.asyncio
async def test_main_with_empty_document_ids_tuple_stays_empty_not_all_filings(monkeypatch):
    """document_ids=() means 'restrict to zero filings', a distinct intent from document_ids=None ('not provided'). An explicitly empty
    tuple must stay empty: get_nodes_by_document must never be called, and
    build_pools must receive an empty documents list."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(),
        ) as mock_get_nodes,
        patch(
            "dataset_generation.run_dataset_generation.build_pools",
            return_value=([], []),
        ) as mock_build_pools,
    ):
        await main(db_path="unused.db", document_ids=())

    mock_get_nodes.assert_not_awaited()
    mock_build_pools.assert_called_once_with([])


@pytest.mark.asyncio
async def test_main_with_document_ids_none_defaults_to_all_filings(monkeypatch):
    """Guards the unaffected case: document_ids=None (or omitted entirely)
    must still default to _ALL_FILINGS."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=[]),
        ) as mock_get_nodes,
        patch(
            "dataset_generation.run_dataset_generation.build_pools",
            return_value=([], []),
        ) as mock_build_pools,
    ):
        await main(db_path="unused.db", document_ids=None)

    assert mock_get_nodes.await_count == len(rdg._ALL_FILINGS)
    called_document_ids = {call.args[1] for call in mock_get_nodes.await_args_list}
    assert called_document_ids == set(rdg._ALL_FILINGS)

    mock_build_pools.assert_called_once()
    documents_arg = mock_build_pools.call_args.args[0]
    assert {doc_id for doc_id, _nodes in documents_arg} == set(rdg._ALL_FILINGS)


def test_all_filings_derived_from_manifest_includes_all_13_current_filings():
    """_ALL_FILINGS must track data/filings_manifest.json, not a stale hardcoded
    list. A mismatch silently excludes filings from every run, with no error
    or warning."""
    assert len(rdg._ALL_FILINGS) == 13
    assert "JPM_2023" in rdg._ALL_FILINGS
    assert "JPM_2024" in rdg._ALL_FILINGS
    assert "JNJ_2023" in rdg._ALL_FILINGS
    assert "JNJ_2024" in rdg._ALL_FILINGS


_FAKE_NODES = [
    {
        "node_id": "n1",
        "document_id": "DOC_A",
        "parent_item_header": "Item 1A. Risk Factors",
        "node_type": "text",
        "source_page_num": 1,
        "content": "Total revenue was $100 million.",
        "token_count": 10,
    }
]

_FAKE_GENERATED = {
    "query_text": "What was total revenue?",
    "ground_truth_answer": "$100 million",
    "gt_citations": ["n1"],
    "quadrant": "Q1_Direct_Text",
    "document_id": "DOC_A",
}


# --- Duplicate-question check before acceptance ------------------------------


@pytest.mark.asyncio
async def test_duplicate_check_below_threshold_accepted_normally(monkeypatch, tmp_path):
    """A candidate whose embedding_similarity score against an existing
    accepted query is below _DUPLICATE_QUESTION_SIMILARITY_THRESHOLD must be
    accepted normally."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[("QT1_PQ_001", "What was net income last year?")]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ) as mock_generate,
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.embedding_similarity",
            return_value=0.5,
        ) as mock_similarity,
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_similarity.assert_called_once_with(_FAKE_GENERATED["query_text"], "What was net income last year?")
    mock_insert.assert_awaited_once()
    mock_generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_check_at_or_above_threshold_rejected_and_retried(monkeypatch, tmp_path, caplog):
    """A candidate whose embedding_similarity score against an existing
    accepted query is >= _DUPLICATE_QUESTION_SIMILARITY_THRESHOLD (0.92) must
    NOT be inserted via _accept_query, must be logged/handled as a duplicate
    (not silently dropped), and the attempt loop must continue to the next
    attempt rather than crashing or accepting anyway."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    existing = [("QT1_PQ_001", "What was total revenue last fiscal year?")]

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=existing),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ) as mock_generate,
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.embedding_similarity",
            return_value=0.95,
        ),
        caplog.at_level(logging.INFO, logger="dataset_generation.run_dataset_generation"),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    # Every attempt is a duplicate -> never inserted, all _MAX_ATTEMPTS_PER_SECTION
    # attempts exhausted.
    mock_insert.assert_not_awaited()
    assert mock_generate.await_count == rdg._MAX_ATTEMPTS_PER_SECTION

    duplicate_records = [record for record in caplog.records if "DUPLICATE" in record.message]
    assert len(duplicate_records) == rdg._MAX_ATTEMPTS_PER_SECTION
    assert "QT1_PQ_001" in duplicate_records[0].message
    assert "0.950" in duplicate_records[0].message

    # The retry feedback must steer the next attempt away from the duplicate
    # rather than repeat it.
    second_call_feedback = mock_generate.await_args_list[1].kwargs.get("previous_attempt_feedback")
    assert second_call_feedback is not None
    assert "too similar" in second_call_feedback.lower()
    assert "QT1_PQ_001" in second_call_feedback


@pytest.mark.asyncio
async def test_first_attempt_rejected_second_attempt_receives_rejection_feedback(monkeypatch, tmp_path):
    """Attempt 1 is rejected by check_query (citation mismatch: Critic cites
    a node the Generator didn't). Because every Groq call runs at
    temperature=0, retrying generate_query with an identical prompt would
    deterministically reproduce the same rejected candidate, so attempt 2's call to generate_query must receive a `previous_attempt_feedback`
    string describing what specifically failed on attempt 1, and attempt 1's
    call must not have received any feedback (nothing to report yet)."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ) as mock_generate,
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(
                side_effect=[
                    # attempt 1: Critic cites an unrelated node -> no
                    # citation overlap -> check_query rejects.
                    {"cited_node_ids": ["n_unrelated"], "computed_answer": "$100 million"},
                    # attempt 2: Critic's citation now overlaps -> accepted.
                    {"cited_node_ids": ["n1"], "computed_answer": "$100 million"},
                ]
            ),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_awaited_once()
    assert mock_generate.await_count == 2

    first_call_kwargs = mock_generate.await_args_list[0].kwargs
    second_call_kwargs = mock_generate.await_args_list[1].kwargs

    assert first_call_kwargs.get("previous_attempt_feedback") is None

    second_feedback = second_call_kwargs.get("previous_attempt_feedback")
    assert second_feedback is not None
    assert "citation" in second_feedback.lower()
    assert "n_unrelated" in second_feedback


@pytest.mark.asyncio
async def test_first_attempt_accepted_no_feedback_passed(monkeypatch, tmp_path):
    """When the first attempt is accepted outright, generate_query must be
    called exactly once, with no previous_attempt_feedback (there was
    nothing to fail yet)."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ) as mock_generate,
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_awaited_once()
    mock_generate.assert_awaited_once()
    call_kwargs = mock_generate.await_args_list[0].kwargs
    assert call_kwargs.get("previous_attempt_feedback") is None


@pytest.mark.asyncio
async def test_critic_runtime_error_on_first_attempt_recovers_on_second(monkeypatch, tmp_path):
    """critique_query raises RuntimeError (Critic exceeded tool-call rounds) on
    attempt 1; attempt 2 succeeds and the query still gets accepted. main()
    must not crash and must not lose the section; the per-attempt retry loop
    absorbs the exception exactly like a rejection."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(
                side_effect=[
                    RuntimeError("Critic exceeded 5 tool-call rounds without a final answer"),
                    {"cited_node_ids": ["n1"], "computed_answer": "$100 million"},
                ]
            ),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_awaited_once()


@pytest.mark.asyncio
async def test_all_attempts_raise_json_decode_error_section_skipped_not_crashed(monkeypatch, tmp_path):
    """generate_query raises json.JSONDecodeError on every attempt (malformed
    LLM output). main() must not raise, must not insert anything, and must
    move on, skipping the section exactly as an all-rejected section is."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    failure_log = tmp_path / "failures.json"
    monkeypatch.setattr("dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", failure_log)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(
                side_effect=json.JSONDecodeError("Expecting value", "not json", 0)
            ),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(),
        ),
    ):
        # main() must complete without raising.
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_not_awaited()

    # Every one of the _MAX_ATTEMPTS_PER_SECTION attempts should have logged
    # a failure durably, identifying the section/quadrant/attempt.
    assert failure_log.exists()
    logged = json.loads(failure_log.read_text())
    assert len(logged) == 3
    for index, entry in enumerate(logged, start=1):
        assert entry["document_id"] == "DOC_A"
        assert entry["quadrant"] == "Q1_Direct_Text"
        assert entry["table"] == "queries"
        assert entry["attempt"] == index
        assert entry["exception_type"] == "JSONDecodeError"


@pytest.mark.asyncio
async def test_api_status_error_is_caught_and_logged(monkeypatch, tmp_path):
    """A non-429 openai.APIStatusError bubbling out of generate_query/
    critique_query must be caught per-attempt, logged, and must not crash
    main()."""
    import httpx
    from openai import APIStatusError

    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    failure_log = tmp_path / "failures.json"
    monkeypatch.setattr("dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", failure_log)

    fake_response = httpx.Response(
        status_code=500, request=httpx.Request("POST", "https://api.groq.com/x")
    )
    api_error = APIStatusError("server error", response=fake_response, body=None)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(side_effect=api_error),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_not_awaited()
    logged = json.loads(failure_log.read_text())
    assert len(logged) == 3
    assert all(entry["exception_type"] == "APIStatusError" for entry in logged)


@pytest.mark.asyncio
async def test_openai_api_status_error_is_caught_and_logged(monkeypatch, tmp_path):
    """The real Groq call path raises openai.APIStatusError (via LlamaIndex's
    OpenAILike -> openai.AsyncOpenAI), not groq.APIStatusError. A genuine
    rate-limit or status failure must be caught and logged like any other
    per-attempt failure, not crash the whole orchestrator."""
    import httpx
    import openai

    failure_log = tmp_path / "failures.json"
    monkeypatch.setattr(rdg, "FAILURE_LOG_PATH", failure_log)

    fake_response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": {"message": "rate limited", "type": "tokens", "code": "rate_limit_exceeded"}},
    )
    api_error = openai.APIStatusError(
        message="rate limited", response=fake_response, body={"error": {"message": "rate limited"}}
    )

    async def fake_generate_query(section, quadrant, previous_attempt_feedback=None):
        raise api_error

    monkeypatch.setattr(rdg, "generate_query", fake_generate_query)

    section = {"document_id": "DOC_A", "section_header": "Item 1A", "node_ids": ["n1"], "content": "x"}
    counts = {table: {quadrant_name: 0 for quadrant_name in _QUADRANTS} for table in _TABLE_ORDER}

    result = await rdg._attempt_fill(
        "unused.db", section, "queries", "Q1_Direct_Text", [], counts, "DOC_A"
    )

    assert result is False
    logged = json.loads(failure_log.read_text())
    assert logged[-1]["exception_type"] == "APIStatusError"
    assert logged[-1]["document_id"] == "DOC_A"


@pytest.mark.asyncio
async def test_null_gt_citations_raises_type_error_caught_and_logged(monkeypatch, tmp_path):
    """generate_query returns valid JSON but gt_citations: null (a malformed
    but non-exception-raising LLM response). check_query -> citations_overlap
    calls set(None), raising TypeError. main() must catch this per-attempt,
    log it, and not crash, exactly like the other malformed-response exceptions
    in this file."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    failure_log = tmp_path / "failures.json"
    monkeypatch.setattr("dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", failure_log)

    generated_with_null_citations = dict(_FAKE_GENERATED, gt_citations=None)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=generated_with_null_citations),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        # main() must complete without raising TypeError.
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_not_awaited()
    logged = json.loads(failure_log.read_text())
    assert len(logged) == 3
    assert all(entry["exception_type"] == "TypeError" for entry in logged)


@pytest.mark.asyncio
async def test_null_computed_answer_raises_attribute_error_caught_and_logged(monkeypatch, tmp_path):
    """critique_query returns a valid dict but computed_answer: null, and
    neither answer contains a number. check_query -> values_match calls
    critic_answer.strip() on None, raising AttributeError. main() must catch
    this per-attempt, log it, and not crash."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    failure_log = tmp_path / "failures.json"
    monkeypatch.setattr("dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", failure_log)

    generated_no_numbers = dict(_FAKE_GENERATED, ground_truth_answer="Not disclosed")

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=generated_no_numbers),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": None}),
        ),
    ):
        # main() must complete without raising AttributeError.
        await main(db_path="unused.db", document_ids=["DOC_A"])

    mock_insert.assert_not_awaited()
    logged = json.loads(failure_log.read_text())
    assert len(logged) == 3
    assert all(entry["exception_type"] == "AttributeError" for entry in logged)


# --- Content-aware, company-balanced pool routing ----------------------------


def _node(node_id, document_id, header, node_type="text", content="text", token_count=10):
    return {
        "node_id": node_id,
        "document_id": document_id,
        "parent_item_header": header,
        "node_type": node_type,
        "source_page_num": 1,
        "content": content,
        "token_count": token_count,
    }


def _section(node_ids, document_id="DOC_A"):
    return {
        "document_id": document_id,
        "section_header": "Item 1A. Risk Factors",
        "node_ids": node_ids,
        "content": "irrelevant",
        "token_count": 10,
    }


def test_classify_section_text_when_no_table_nodes():
    nodes = [_node("n1", "DOC_A", "Item 1A", node_type="text"), _node("n2", "DOC_A", "Item 1A", node_type="text")]
    section = _section(["n1", "n2"])
    assert classify_section(section, nodes) == "text"


def test_classify_section_table_when_any_node_is_table():
    """A single table node among mostly-text nodes is still enough material for a
    Q3/Q4 question, so classification uses 'any', not 'majority'."""
    nodes = [
        _node("n1", "DOC_A", "Item 8", node_type="text"),
        _node("n2", "DOC_A", "Item 8", node_type="text"),
        _node("n3", "DOC_A", "Item 8", node_type="table"),
    ]
    section = _section(["n1", "n2", "n3"])
    assert classify_section(section, nodes) == "table"


def test_classify_section_all_table_nodes_is_table():
    nodes = [_node("n1", "DOC_A", "Item 8", node_type="table")]
    section = _section(["n1"])
    assert classify_section(section, nodes) == "table"


def test_round_robin_interleave_alternates_companies_per_round():
    buckets = {
        "AAPL": [{"id": "a1"}, {"id": "a2"}],
        "MSFT": [{"id": "m1"}, {"id": "m2"}],
        "TSLA": [{"id": "t1"}, {"id": "t2"}],
    }
    result = _round_robin_interleave(buckets, ["AAPL", "MSFT", "TSLA"])
    assert [item["id"] for item in result] == ["a1", "m1", "t1", "a2", "m2", "t2"]


def test_round_robin_interleave_shorter_bucket_stops_contributing_without_breaking_order():
    buckets = {
        "AAPL": [{"id": "a1"}],
        "MSFT": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}],
    }
    result = _round_robin_interleave(buckets, ["AAPL", "MSFT"])
    # AAPL only has one section: contributes round 1, then MSFT keeps going alone.
    assert [item["id"] for item in result] == ["a1", "m1", "m2", "m3"]


def test_build_pools_routes_by_content_type_and_interleaves_companies():
    """Two companies, each with one text section and one table section.
    Pools must (a) route text -> text_pool / table -> table_pool, and
    (b) interleave across companies rather than grouping AAPL's sections
    before MSFT's."""
    documents = [
        (
            "AAPL_2023",
            [
                _node("a_text1", "AAPL_2023", "Item 1A", node_type="text"),
                _node("a_table1", "AAPL_2023", "Item 8", node_type="table"),
            ],
        ),
        (
            "MSFT_2023",
            [
                _node("m_text1", "MSFT_2023", "Item 1A", node_type="text"),
                _node("m_table1", "MSFT_2023", "Item 8", node_type="table"),
            ],
        ),
    ]
    text_pool, table_pool = build_pools(documents)

    assert [section["document_id"] for section in text_pool] == ["AAPL_2023", "MSFT_2023"]
    assert [section["document_id"] for section in table_pool] == ["AAPL_2023", "MSFT_2023"]
    assert all(section["section_header"] == "Item 1A" for section in text_pool)
    assert all(section["section_header"] == "Item 8" for section in table_pool)


def _words(word_count):
    """`n` single-token words. tiktoken's cl100k_base encoding tokenizes the bare
    word "x" to exactly 1 token, so `_words(n)` gives content with an exact
    token count of `n`, checked against the real encoding."""
    return " ".join(["x"] * word_count)


def test_chunk_section_under_limit_returns_itself_unchanged():
    """The common case: a section under _MAX_SECTION_TOKENS must produce exactly
    one chunk, and that chunk must be the original section dict, unchanged."""
    nodes = [_node("n1", "DOC_A", "Item 1A", node_type="text", content="alpha beta gamma")]
    section = _section(["n1"])

    result = chunk_section(section, nodes)

    assert result == [section]


def test_chunk_section_over_limit_splits_into_multiple_chunks_each_under_limit(monkeypatch):
    monkeypatch.setattr(rdg, "_MAX_SECTION_TOKENS", 10)
    nodes = [
        _node("n1", "DOC_A", "Item 1A", node_type="text", content=_words(6)),
        _node("n2", "DOC_A", "Item 1A", node_type="text", content=_words(6)),
        _node("n3", "DOC_A", "Item 1A", node_type="text", content=_words(6)),
    ]
    section = {
        "document_id": "DOC_A",
        "section_header": "Item 1A",
        "node_ids": ["n1", "n2", "n3"],
        "content": "irrelevant",
        "token_count": 18,
    }

    chunks = chunk_section(section, nodes)

    assert len(chunks) > 1
    encoding = rdg._get_encoding()
    for chunk in chunks:
        assert len(encoding.encode(chunk["content"])) <= 10
    # every original node lands in exactly one chunk, order preserved.
    assert [nid for packed_chunk in chunks for nid in packed_chunk["node_ids"]] == ["n1", "n2", "n3"]
    for chunk in chunks:
        assert chunk["document_id"] == "DOC_A"
        assert chunk["section_header"] == "Item 1A"


def test_chunk_section_never_splits_text_node_away_from_following_table(monkeypatch):
    """Naive greedy packing would close the chunk right before n2 (the
    6-token text node n1 already fills the 10-token budget, and n2 alone
    would push it over), but n2 is a table immediately following text
    node n1, so the boundary must back up and keep n1+n2 together in one
    (over-budget) chunk instead of splitting them."""
    monkeypatch.setattr(rdg, "_MAX_SECTION_TOKENS", 10)
    nodes = [
        _node("n1", "DOC_A", "Item 8", node_type="text", content=_words(6)),
        _node("n2", "DOC_A", "Item 8", node_type="table", content=_words(6)),
        _node("n3", "DOC_A", "Item 8", node_type="text", content=_words(6)),
    ]
    section = {
        "document_id": "DOC_A",
        "section_header": "Item 8",
        "node_ids": ["n1", "n2", "n3"],
        "content": "irrelevant",
        "token_count": 18,
    }

    chunks = chunk_section(section, nodes)

    # n1 (text) and n2 (table) must land in the same chunk, even though
    # that chunk exceeds _MAX_SECTION_TOKENS as a result.
    first_chunk_ids = chunks[0]["node_ids"]
    assert "n1" in first_chunk_ids
    assert "n2" in first_chunk_ids
    assert first_chunk_ids.index("n1") < first_chunk_ids.index("n2")
    # n3 was pushed out to its own chunk.
    assert "n3" not in first_chunk_ids
    assert any("n3" in packed_chunk["node_ids"] for packed_chunk in chunks[1:])


def test_chunk_section_single_oversized_node_logged_and_skipped_rest_packed(monkeypatch, caplog):
    """A single node (almost always one huge table) that alone exceeds
    _MAX_SECTION_TOKENS cannot be split further without breaking table
    atomicity, so it must be logged and skipped, without blocking the rest
    of the section's nodes from being packed normally."""
    monkeypatch.setattr(rdg, "_MAX_SECTION_TOKENS", 10)
    nodes = [
        _node("n1", "DOC_A", "Item 8", node_type="text", content=_words(5)),
        _node("huge", "DOC_A", "Item 8", node_type="table", content=_words(20)),
        _node("n2", "DOC_A", "Item 8", node_type="text", content=_words(5)),
    ]
    section = {
        "document_id": "DOC_A",
        "section_header": "Item 8",
        "node_ids": ["n1", "huge", "n2"],
        "content": "irrelevant",
        "token_count": 30,
    }

    with caplog.at_level(logging.WARNING, logger=rdg.logger.name):
        chunks = chunk_section(section, nodes)

    all_node_ids = [nid for packed_chunk in chunks for nid in packed_chunk["node_ids"]]
    assert "huge" not in all_node_ids
    assert "n1" in all_node_ids
    assert "n2" in all_node_ids
    assert any("huge" in record.getMessage() and "20" in record.getMessage() for record in caplog.records)


def test_next_target_restricts_search_to_given_quadrants():
    """A pool restricted to _TABLE_QUADRANTS must never report a target in
    Q1/Q2, even if those quadrants are the emptiest overall. That is what keeps table sections from ever being routed at a text quadrant."""
    counts = {
        table: {quadrant_name: 0 for quadrant_name in _QUADRANTS}
        for table in ("queries", "golden_queries", "judge_validation")
    }
    target = next_target(counts, _TABLE_QUADRANTS)
    assert target == ("queries", "Q3_Direct_Table")

    # Even once Q3/Q4 are completely full, a table-restricted search must
    # not spill over into Q1/Q2.
    for table in counts:
        for quadrant_name in _TABLE_QUADRANTS:
            counts[table][quadrant_name] = _TARGETS[table]
    assert next_target(counts, _TABLE_QUADRANTS) is None


def test_format_underfill_summary_marks_only_below_target_quadrants():
    counts = {table: {quadrant_name: _TARGETS[table] for quadrant_name in _QUADRANTS} for table in _TABLE_ORDER}
    counts["queries"]["Q3_Direct_Table"] = 18  # underfilled

    summary = format_underfill_summary(counts)

    assert "Q3_Direct_Table 18/25 (underfilled)" in summary
    assert "Q4_Implicit_Table 25/25" in summary
    assert "(underfilled)" not in summary.split("queries:")[-1].split("\n")[1]  # golden_queries line is clean


@pytest.mark.asyncio
async def test_main_pool_cycling_is_capped_and_terminates(monkeypatch, tmp_path):
    """If accepted counts never actually rise from the DB's point of view
    (e.g. get_quadrant_counts stubbed to always return 0, simulating a slot
    that structurally can never be satisfied), main() must not loop forever. It visits the one available section exactly
    _MAX_POOL_CYCLES times, then stops."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", False)
    monkeypatch.setattr("dataset_generation.run_dataset_generation._MAX_POOL_CYCLES", 2)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.SUMMARY_LOG_PATH", tmp_path / "summary.json"
    )

    fake_nodes = [_node("n1", "DOC_A", "Item 1A. Risk Factors", node_type="text")]

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=fake_nodes),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()) as mock_insert,
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value={
                "query_text": "What was total revenue?",
                "ground_truth_answer": "$100 million",
                "gt_citations": ["n1"],
                "quadrant": "Q1_Direct_Text",
                "document_id": "DOC_A",
            }),
        ) as mock_generate,
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    # One text section, cap of 2 cycles -> visited (and accepted-then-still-
    # counted-as-0) exactly twice, never more.
    assert mock_generate.await_count == 2
    assert mock_insert.await_count == 2


@pytest.mark.asyncio
async def test_main_routes_table_sections_only_to_table_quadrants(monkeypatch, tmp_path):
    """End-to-end content-aware routing check: a document with one text
    section and one table section must only ever have generate_query called
    with a text quadrant for the text section and a table quadrant for the
    table section, never mismatched."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", False)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.SUMMARY_LOG_PATH", tmp_path / "summary.json"
    )
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation._TARGETS",
        {"queries": 1, "golden_queries": 0, "judge_validation": 0},
    )

    fake_nodes = [
        _node("n_text", "DOC_A", "Item 1A. Risk Factors", node_type="text"),
        _node("n_table", "DOC_A", "Item 8. Financial Statements", node_type="table"),
    ]

    # Stateful fake DB: tracks accepted counts per (table, quadrant) so
    # next_target actually advances instead of looping forever.
    fake_counts = {
        table: {quadrant_name: 0 for quadrant_name in _QUADRANTS}
        for table in ("queries", "golden_queries", "judge_validation")
    }

    async def fake_get_quadrant_counts(db_path, table):
        return dict(fake_counts[table])

    async def fake_insert_query(db_path, row):
        fake_counts["queries"][row["quadrant"]] += 1

    calls = []

    async def fake_generate_query(section, quadrant, previous_attempt_feedback=None):
        calls.append((section["section_header"], quadrant))
        return {
            "query_text": f"query for {section['section_header']}",
            "ground_truth_answer": "$100 million",
            "gt_citations": [section["node_ids"][0]],
            "quadrant": quadrant,
            "document_id": section["document_id"],
        }

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=fake_get_quadrant_counts,
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=fake_nodes),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=fake_insert_query),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch("dataset_generation.run_dataset_generation.generate_query", new=fake_generate_query),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n_text"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    text_calls = [quadrant_name for header, quadrant_name in calls if header == "Item 1A. Risk Factors"]
    table_calls = [quadrant_name for header, quadrant_name in calls if header == "Item 8. Financial Statements"]

    assert text_calls and all(quadrant_name in _TEXT_QUADRANTS for quadrant_name in text_calls)
    assert table_calls and all(quadrant_name in _TABLE_QUADRANTS for quadrant_name in table_calls)


# --- Per-attempt progress logging ---------------------------------------------


@pytest.mark.asyncio
async def test_accepted_attempt_logs_document_id_quadrant_and_accepted(monkeypatch, caplog):
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
        caplog.at_level(logging.INFO, logger="dataset_generation.run_dataset_generation"),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    accepted_records = [record for record in caplog.records if "ACCEPTED" in record.message]
    assert accepted_records, "expected an ACCEPTED progress line"
    message = accepted_records[0].message
    assert "DOC_A" in message
    assert "Q1_Direct_Text" in message
    assert "ACCEPTED" in message


@pytest.mark.asyncio
async def test_rejected_attempt_logs_diagnose_rejection_reason(monkeypatch, caplog, tmp_path):
    """Attempt 1 is rejected via a citation mismatch (Critic cites an
    unrelated node); the progress line for that attempt must surface
    diagnose_rejection's specific reason, not just a generic 'REJECTED'."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(
                side_effect=[
                    {"cited_node_ids": ["n_unrelated"], "computed_answer": "$100 million"},
                    {"cited_node_ids": ["n1"], "computed_answer": "$100 million"},
                ]
            ),
        ),
        caplog.at_level(logging.INFO, logger="dataset_generation.run_dataset_generation"),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    rejected_records = [record for record in caplog.records if "REJECTED" in record.message]
    assert rejected_records, "expected a REJECTED progress line"
    message = rejected_records[0].message
    assert "citation mismatch" in message.lower()
    assert "n_unrelated" in message


@pytest.mark.asyncio
async def test_exception_caught_attempt_logs_exception_type(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)
    monkeypatch.setattr(
        "dataset_generation.run_dataset_generation.FAILURE_LOG_PATH", tmp_path / "failures.json"
    )

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(
                side_effect=json.JSONDecodeError("Expecting value", "not json", 0)
            ),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(),
        ),
        caplog.at_level(logging.INFO, logger="dataset_generation.run_dataset_generation"),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])

    exception_records = [record for record in caplog.records if "EXCEPTION" in record.message]
    assert len(exception_records) == 3  # one per _MAX_ATTEMPTS_PER_SECTION
    assert all("JSONDecodeError" in record.message for record in exception_records)


@pytest.mark.asyncio
async def test_calling_main_twice_does_not_duplicate_log_handlers(monkeypatch, caplog):
    """main() calls _configure_logging() on every entry; a process that (in
    theory) calls main() more than once must not accumulate a second
    console+file handler pair, which would otherwise print/write every
    subsequent progress line twice."""
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.LOCAL_TEST_THROTTLE", True)
    monkeypatch.setattr("dataset_generation.run_dataset_generation.config.THROTTLE_LIMIT", 1)

    with (
        patch("dataset_generation.run_dataset_generation.dbm.init_db", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_quadrant_counts",
            new=AsyncMock(return_value={quadrant_name: 0 for quadrant_name in _QUADRANTS}),
        ),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_nodes_by_document",
            new=AsyncMock(return_value=_FAKE_NODES),
        ),
        patch("dataset_generation.run_dataset_generation.dbm.insert_query", new=AsyncMock()),
        patch(
            "dataset_generation.run_dataset_generation.dbm.get_all_query_texts",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "dataset_generation.run_dataset_generation.generate_query",
            new=AsyncMock(return_value=_FAKE_GENERATED),
        ),
        patch(
            "dataset_generation.run_dataset_generation.critique_query",
            new=AsyncMock(return_value={"cited_node_ids": ["n1"], "computed_answer": "$100 million"}),
        ),
    ):
        await main(db_path="unused.db", document_ids=["DOC_A"])
        handlers_after_first_call = len(rdg.logger.handlers)
        await main(db_path="unused.db", document_ids=["DOC_A"])
        handlers_after_second_call = len(rdg.logger.handlers)

    assert handlers_after_first_call == 2  # console + file
    assert handlers_after_second_call == handlers_after_first_call
