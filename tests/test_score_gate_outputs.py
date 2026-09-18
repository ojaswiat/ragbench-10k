"""score_gate_outputs: reference scoring and the Concordance-Rate gate.

Two jobs: collect the reference human_score for each of the 60 JEQ rows
without showing the scorer the Judge's score, then compute the Concordance
Rate and enforce the > 80% gate. Scores are 1-5; a row agrees when the two,
rescaled to 0-100 by RESCALE_FACTOR, differ by no more than 10 points. One
band step is 20 points, so only identical bands agree.
"""
import os

import pytest

from core import database_manager as dbm
from gate import gate_reference_scores
from gate import score_gate_outputs as sg

TEST_DB = "test_benchmark_score_gate.db"


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


def _rows(pairs):
    return [
        {"result_id": f"R{index}", "human_score": human_score, "judge_score": judge_score}
        for index, (human_score, judge_score) in enumerate(pairs)
    ]


# --- per-row agreement (0-100 scale, +/-10 band = exact band match) ---


def test_agreement_exact():
    assert sg.rows_agree(5, 5) is True
    assert sg.rows_agree(1, 1) is True


def test_adjacent_bands_do_not_agree():
    """One band step is RESCALE_FACTOR (20) points, above the 10-point
    tolerance. Carrying the old 1-10 scale's plus-or-minus-one-band rule down
    to five bands would have widened the acceptance window from 30% of the
    scale to 60%, inflating the Concordance Rate without the Judge improving."""
    assert sg.rows_agree(4, 5) is False
    assert sg.rows_agree(4, 3) is False


def test_disagreement_two_bands():
    assert sg.rows_agree(4, 2) is False


def test_tolerance_is_narrower_than_one_band():
    """Guards the gate's strictness against a tolerance edit that would
    silently re-admit adjacent bands."""
    assert sg.AGREEMENT_TOLERANCE < sg.RESCALE_FACTOR


# --- Agreement Rate + gate ---


def test_agreement_rate_all_agree_passes():
    summary = sg.compute_agreement_rate(_rows([(5, 5), (4, 4), (3, 3)]))
    assert summary["n"] == 3
    assert summary["agreement_rate"] == 100.0
    assert summary["gate_passed"] is True


def test_agreement_rate_eighty_percent_fails_strict_gate():
    # 4 of 5 agree -> 80.0%, and the gate is strictly > 80.
    summary = sg.compute_agreement_rate(_rows([(5, 5), (4, 4), (3, 3), (2, 2), (1, 4)]))
    assert summary["agreement_rate"] == 80.0
    assert summary["gate_passed"] is False


def test_agreement_rate_just_above_threshold_passes():
    # 9 of 10 agree -> 90%.
    pairs = [(4, 4)] * 9 + [(4, 1)]
    summary = sg.compute_agreement_rate(_rows(pairs))
    assert summary["agreement_rate"] == 90.0
    assert summary["gate_passed"] is True


def test_compute_agreement_rate_requires_both_scores():
    with pytest.raises(ValueError, match="missing"):
        sg.compute_agreement_rate(_rows([(5, None)]))


def test_compute_agreement_rate_empty_raises():
    with pytest.raises(ValueError, match="no"):
        sg.compute_agreement_rate([])


# --- human prompting: does not reveal the judge's score ---


def _jv(query_id, quadrant="Q3_Direct_Table"):
    return {
        "query_id": query_id,
        "quadrant": quadrant,
        "query_text": f"question {query_id}?",
        "ground_truth_answer": "$394.3B",
        "gt_citations": ["AAPL_2025_n0421"],
        "document_id": "SEC_10K_AAPL_2025",
    }


async def _seed_row(db_path, result_id="R1", query_id="JEQ_001", *, judge_score=None, human_score=None):
    """Seed one JEQ gate row plus its judge_validation ground truth.

    Self-contained (creates the schema and the judge_validation row itself)
    so a test can seed a fresh db_path -- e.g. one under tmp_path -- with a
    single call.
    """
    await dbm.init_db(db_path)
    await dbm.insert_judge_validation(db_path, _jv(query_id))
    row = {
        "result_id": result_id,
        "source_set": "JEQ",
        "query_id": query_id,
        "pipeline": "P1_vector",
        "k_value": 5,
        "retrieved_node_ids": ["AAPL_2025_n0421"],
        "pipeline_output": "Net sales were $394.3B. [[node:AAPL_2025_n0421]]",
        "cited_node_ids": ["AAPL_2025_n0421"],
    }
    await dbm.upsert_result(db_path, row)
    if judge_score is not None:
        await dbm.update_result_scores(db_path, result_id, {"judge_score": judge_score})
    if human_score is not None:
        await dbm.update_result_human_score(db_path, result_id, human_score)


@pytest.mark.asyncio
async def test_prompt_human_scores_writes_and_hides_judge_score():
    await _seed_row(TEST_DB, "R1", "JEQ_001", judge_score=5)  # judge already scored

    shown: list[str] = []
    scored = await sg.prompt_human_scores(
        TEST_DB, input_fn=lambda _prompt="": "4", output_fn=shown.append
    )

    assert scored == 1
    rows = await dbm.get_results(TEST_DB, "JEQ")
    assert rows[0]["human_score"] == 4
    # The researcher must not be shown the judge's verdict before scoring.
    blob = "\n".join(shown).lower()
    assert "judge" not in blob


@pytest.mark.asyncio
async def test_prompt_human_scores_skips_already_scored():
    await _seed_row(TEST_DB, "R1", "JEQ_001", judge_score=5, human_score=4)

    scored = await sg.prompt_human_scores(
        TEST_DB, input_fn=lambda _prompt="": "1", output_fn=lambda _output: None
    )
    assert scored == 0  # nothing left to score


@pytest.mark.asyncio
async def test_prompt_human_scores_reprompts_on_invalid():
    await _seed_row(TEST_DB, "R1", "JEQ_001", judge_score=5)

    answers = iter(["fifteen", "12", "4"])  # non-int, out-of-range, then valid
    await sg.prompt_human_scores(
        TEST_DB, input_fn=lambda _prompt="": next(answers), output_fn=lambda _output: None
    )
    rows = await dbm.get_results(TEST_DB, "JEQ")
    assert rows[0]["human_score"] == 4


# --- run_scoring_gate: prompt then compute ---


@pytest.mark.asyncio
async def test_run_scoring_gate_end_to_end():
    for index in range(2):
        await _seed_row(TEST_DB, f"R{index}", f"JEQ_00{index}", judge_score=4)

    summary = await sg.run_scoring_gate(
        TEST_DB, input_fn=lambda _prompt="": "4", output_fn=lambda _output: None
    )
    assert summary["n"] == 2
    assert summary["agreement_rate"] == 100.0
    assert summary["gate_passed"] is True


# --- gate_reference_scores: non-interactive path, judge_score excluded by construction ---


async def test_rendered_rows_never_carry_the_judge_score(tmp_path):
    """Independence is the only property the gate figure retains on this
    path, so the judge_score must be absent from the rendering by
    construction rather than by the caller's restraint."""
    db_path = str(tmp_path / "t.db")
    await _seed_row(db_path, judge_score=5)
    rows = await gate_reference_scores.render_rows_for_scoring(db_path)
    assert rows
    for row in rows:
        assert "judge_score" not in row


_PIPELINE_SUBSTRINGS = ("P1_vector", "P2_bm25", "P3_structural")


async def test_rendered_rows_are_blind_to_pipeline_and_citation_evidence(tmp_path):
    """Two more leaks beyond judge_score: pipeline identity (both the
    `pipeline` column and `result_id`, which embeds it verbatim) and
    citation evidence the Judge itself never sees. Checking only keys would
    miss result_id's leak, since the pipeline name sits inside its *value*
    -- so this scans every rendered value, not just the field names."""
    db_path = str(tmp_path / "t.db")
    await _seed_row(db_path, judge_score=5)
    rows = await gate_reference_scores.render_rows_for_scoring(db_path)
    assert rows
    for row in rows:
        for forbidden_key in (
            "judge_score",
            "pipeline",
            "result_id",
            "gt_citations",
            "cited_node_ids",
        ):
            assert forbidden_key not in row
        for value in row.values():
            text = str(value)
            for substring in _PIPELINE_SUBSTRINGS:
                assert substring not in text


_PIPELINES = ("P1_vector", "P2_bm25", "P3_structural")


async def _seed_all_pipelines(db_path, n_queries=5):
    """n_queries JEQ queries, each with a JEQ row for all three pipelines --
    the natural result_id order this produces (queries.ORDER BY result_id)
    is exactly the pipeline-sorted-within-query cycle that a naive
    positional token would reproduce."""
    await dbm.init_db(db_path)
    for query_index in range(n_queries):
        query_id = f"JEQ_{query_index:03d}"
        await dbm.insert_judge_validation(db_path, _jv(query_id))
        for pipeline in _PIPELINES:
            row = {
                "result_id": f"R_JEQ_{query_id}_{pipeline}_K5",
                "source_set": "JEQ",
                "query_id": query_id,
                "pipeline": pipeline,
                "k_value": 5,
                "retrieved_node_ids": ["AAPL_2025_n0421"],
                "pipeline_output": f"answer from {pipeline} for {query_id}",
                "cited_node_ids": ["AAPL_2025_n0421"],
            }
            await dbm.upsert_result(db_path, row)


async def test_token_order_does_not_track_pipeline(tmp_path):
    """The field whitelist alone is not enough: get_jeq_judging_rows orders
    by result_id, which sorts alphabetically by pipeline within each query,
    so assigning tokens in that natural order would let token_index % 3
    recover the pipeline even though no rendered field names it. Builds
    JEQ rows across all three pipelines for several queries and asserts the
    pipeline sequence behind the tokens is not that repeating cycle."""
    db_path = str(tmp_path / "t.db")
    await _seed_all_pipelines(db_path)

    rows = await gate_reference_scores.render_rows_for_scoring(db_path)
    token_map = await gate_reference_scores._token_map(db_path)
    pipeline_by_result_id = {
        stored_row["result_id"]: stored_row["pipeline"] for stored_row in await dbm.get_results(db_path, "JEQ")
    }
    pipeline_sequence = [pipeline_by_result_id[token_map[row["token"]]] for row in rows]

    # The natural (unshuffled) order would put every 3rd row on the same
    # pipeline -- token_index % 3 recovering the pipeline exactly.
    natural_cycle = [_PIPELINES[index % 3] for index in range(len(pipeline_sequence))]
    assert pipeline_sequence != natural_cycle
    # Same check the leak report used directly: indices 0, 3, 6, ... are
    # not all the same pipeline.
    stride_three = pipeline_sequence[0::3]
    assert len(set(stride_three)) > 1


async def test_apply_scores_writes_every_supplied_row(tmp_path):
    db_path = str(tmp_path / "t.db")
    await _seed_row(db_path, judge_score=5)
    rows = await gate_reference_scores.render_rows_for_scoring(db_path)
    written = await gate_reference_scores.apply_scores(
        db_path, {row["token"]: 4 for row in rows}
    )
    assert written == len(rows)
    stored = await dbm.get_results(db_path, "JEQ")
    assert all(stored_row["human_score"] == 4 for stored_row in stored)


async def test_apply_scores_rejects_an_unknown_token(tmp_path):
    db_path = str(tmp_path / "t.db")
    await _seed_row(db_path, judge_score=5)
    with pytest.raises(ValueError, match="no results row"):
        await gate_reference_scores.apply_scores(db_path, {"row_9999": 5})


def test_verdict_uses_concordance_wording():
    summary = {"agreement_rate": 91.0, "agreements": 55, "n": 60, "gate_passed": True}
    text = sg._render_verdict(summary)
    assert "Concordance" in text
    assert "human" not in text.lower()
