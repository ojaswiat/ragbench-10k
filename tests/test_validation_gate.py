"""validation_gate: run the 60 JEQ gate outputs, then judge them.

The gate runs the 20 JEQ through P1/P2/P3 at a single K=5, giving 60 rows, via
the existing loop_executor, then the Judge scores them. This module is wiring
only: retrieval and answering stay in loop_executor, scoring stays in
async_judge, and both callees carry their own LOCAL_TEST_THROTTLE.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import validation_gate


@pytest.mark.asyncio
async def test_runs_jeq_at_k5_then_judges():
    retrievers = {"P1_vector": MagicMock(), "P2_bm25": MagicMock(), "P3_structural": MagicMock()}
    with patch.object(validation_gate.loop_executor, "main", new=AsyncMock(return_value={"completed": 60})) as gen, \
         patch.object(validation_gate, "judge_jeq_rows", new=AsyncMock(return_value={"judged": 60})) as judge:
        summary = await validation_gate.run_validation_gate("db.sqlite", retrievers=retrievers)

    gen.assert_awaited_once()
    kwargs = gen.await_args.kwargs
    assert kwargs["source_set"] == "JEQ"
    assert kwargs["k_values"] == (5,)
    assert kwargs["retrievers"] is retrievers
    judge.assert_awaited_once()
    assert judge.await_args.args[0] == "db.sqlite" or judge.await_args.kwargs.get("db_path") == "db.sqlite" \
        or judge.await_args.args == ("db.sqlite",)


@pytest.mark.asyncio
async def test_k_value_is_exactly_five():
    assert validation_gate.JEQ_K_VALUE == 5


@pytest.mark.asyncio
async def test_defaults_to_all_three_pipelines():
    built = {"P1_vector": MagicMock(), "P2_bm25": MagicMock(), "P3_structural": MagicMock()}
    with patch.object(validation_gate.loop_executor, "build_retrievers", return_value=built) as build, \
         patch.object(validation_gate.loop_executor, "main", new=AsyncMock(return_value={})), \
         patch.object(validation_gate, "judge_jeq_rows", new=AsyncMock(return_value={})):
        await validation_gate.run_validation_gate("db.sqlite")

    build.assert_called_once()
    assert set(build.call_args.args[0]) == {"P1_vector", "P2_bm25", "P3_structural"}


@pytest.mark.asyncio
async def test_returns_both_summaries():
    with patch.object(validation_gate.loop_executor, "main", new=AsyncMock(return_value={"completed": 60})), \
         patch.object(validation_gate, "judge_jeq_rows", new=AsyncMock(return_value={"judged": 60})):
        summary = await validation_gate.run_validation_gate("db.sqlite", retrievers={"P1_vector": MagicMock()})

    assert summary["generation"] == {"completed": 60}
    assert summary["judging"] == {"judged": 60}
