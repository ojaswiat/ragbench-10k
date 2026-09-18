"""aggregate_results: tri-pillar aggregation and paired pipeline comparison.

The statistics are implemented here rather than imported, so they are tested
against hand-computable cases rather than trusted. The pairing is the part
most likely to be silently wrong: pairing on the wrong key would still
produce plausible-looking numbers, so it is asserted directly.
"""
import math
import sqlite3

import pytest

from analysis import aggregate_results as agg


def _row(query_id, pipeline, k_value, judge=None, recall=None, **extra):
    row = {"query_id": query_id, "pipeline": pipeline, "k_value": k_value,
           "judge_score": judge, "recall_at_k": recall, "precision_at_k": None,
           "evidence_hit": None, "citation_match": None, "token_f1": None,
           "exact_match": None, "latency_sec": None, "input_tokens": None,
           "output_tokens": None, "quadrant": "Q1_Direct_Text", "document_id": "AAPL_2023"}
    row.update(extra)
    return row


# --- summarise ----------------------------------------------------------------

def test_an_empty_group_reports_no_cells_rather_than_dividing_by_zero():
    assert agg.summarise([]) == {"n": 0}


def test_a_column_that_is_null_everywhere_reports_none_not_zero():
    """exact_match is NULL by design on Q2/Q4; that must not read as 0.0."""
    out = agg.summarise([_row("Q1", "P1_vector", 5, judge=5)])
    assert out["exact_match"] is None
    assert out["judge_mean"] == 5


def test_nulls_are_skipped_rather_than_counted_as_zero():
    rows = [_row("Q1", "P1_vector", 5, recall=1.0), _row("Q2", "P1_vector", 5, recall=None)]
    assert agg.summarise(rows)["recall_at_k"] == 1.0


def test_pass_rate_counts_four_and_five_and_fail_rate_counts_only_one():
    rows = [_row(f"Q{index}", "P1_vector", 5, judge=score) for index, score in enumerate([1, 1, 3, 4, 5])]
    out = agg.summarise(rows)
    assert out["judge_pass_rate"] == pytest.approx(2 / 5)
    assert out["judge_fail_rate"] == pytest.approx(2 / 5)
    assert out["judge_mean"] == pytest.approx(14 / 5)


def test_pass_rate_uses_the_declared_threshold():
    assert agg.PASS_THRESHOLD == 4
    rows = [_row("Q1", "P1_vector", 5, judge=3)]
    assert agg.summarise(rows)["judge_pass_rate"] == 0.0


# --- grouping -----------------------------------------------------------------

def test_grouping_splits_on_every_key_given():
    rows = [_row("Q1", "P1_vector", 2, judge=5), _row("Q1", "P1_vector", 5, judge=1),
            _row("Q1", "P2_bm25", 2, judge=3)]
    by_pipeline = agg.group_by(rows, "pipeline")
    assert by_pipeline[("P1_vector",)]["n"] == 2
    by_both = agg.group_by(rows, "pipeline", "k_value")
    assert by_both[("P1_vector", 2)]["judge_mean"] == 5
    assert by_both[("P1_vector", 5)]["judge_mean"] == 1


# --- pairing ------------------------------------------------------------------

def test_pairing_matches_on_both_query_and_k_not_query_alone():
    rows = [_row("Q1", "P1_vector", 2, judge=5), _row("Q1", "P1_vector", 5, judge=1),
            _row("Q1", "P2_bm25", 2, judge=2), _row("Q1", "P2_bm25", 5, judge=4)]
    pairs = agg._paired_values(rows, "P1_vector", "P2_bm25", "judge_score")
    assert sorted(pairs) == [(1, 4), (5, 2)], "k must be part of the pairing key"


def test_a_cell_only_one_pipeline_answered_is_excluded_from_the_pairing():
    rows = [_row("Q1", "P1_vector", 5, judge=5), _row("Q2", "P1_vector", 5, judge=5),
            _row("Q1", "P2_bm25", 5, judge=1)]
    assert agg._paired_values(rows, "P1_vector", "P2_bm25", "judge_score") == [(5, 1)]


def test_a_null_metric_drops_its_pair_rather_than_counting_as_zero():
    rows = [_row("Q1", "P1_vector", 5, judge=None), _row("Q1", "P2_bm25", 5, judge=5)]
    assert agg._paired_values(rows, "P1_vector", "P2_bm25", "judge_score") == []


# --- the sign test ------------------------------------------------------------

def test_sign_test_matches_a_hand_computed_binomial():
    # 9 wins, 1 loss: two-sided p = 2 * (C(10,0) + C(10,1)) / 2^10
    pairs = [(1, 0)] * 9 + [(0, 1)]
    expected = 2 * (math.comb(10, 0) + math.comb(10, 1)) / 2 ** 10
    assert agg._sign_test(pairs) == pytest.approx(expected)


def test_an_even_split_is_not_significant():
    assert agg._sign_test([(1, 0)] * 5 + [(0, 1)] * 5) == pytest.approx(1.0)


def test_all_ties_carry_no_evidence():
    assert agg._sign_test([(3, 3)] * 50) == 1.0


def test_ties_are_dropped_rather_than_counted_as_agreement():
    """Ties padding the sample must not make a lopsided split look weaker."""
    lopsided = [(1, 0)] * 9 + [(0, 1)]
    assert agg._sign_test(lopsided + [(2, 2)] * 100) == pytest.approx(agg._sign_test(lopsided))


# --- the bootstrap ------------------------------------------------------------

def test_the_interval_is_reproducible_for_the_same_input():
    pairs = [(5, 1), (4, 2), (1, 1), (5, 5), (2, 4)]
    assert agg._bootstrap_ci(pairs) == agg._bootstrap_ci(pairs)


def test_an_all_positive_difference_gives_an_interval_above_zero():
    interval_low, interval_high = agg._bootstrap_ci([(5, 1)] * 40)
    assert interval_low > 0 and interval_high > 0


def test_a_difference_straddling_zero_gives_an_interval_containing_it():
    interval_low, interval_high = agg._bootstrap_ci([(5, 1), (1, 5)] * 20)
    assert interval_low <= 0 <= interval_high


# --- compare ------------------------------------------------------------------

def test_a_consistent_win_is_reported_as_significant():
    rows = []
    for index in range(30):
        rows += [_row(f"Q{index}", "P1_vector", 5, judge=5), _row(f"Q{index}", "P2_bm25", 5, judge=1)]
    out = agg.compare(rows, "P1_vector", "P2_bm25")
    assert out["wins"] == 30 and out["losses"] == 0
    assert out["mean_difference"] == pytest.approx(4.0)
    assert out["significant"] is True


def test_a_coin_flip_difference_is_not_reported_as_significant():
    rows = []
    for index in range(30):
        left_value, right_value = (5, 1) if index % 2 else (1, 5)
        rows += [_row(f"Q{index}", "P1_vector", 5, judge=left_value), _row(f"Q{index}", "P2_bm25", 5, judge=right_value)]
    out = agg.compare(rows, "P1_vector", "P2_bm25")
    assert out["significant"] is False


def test_significance_requires_the_interval_to_exclude_zero():
    """A large point estimate whose interval spans zero is not evidence."""
    rows = []
    for index in range(6):
        left_value, right_value = (5, 1) if index < 4 else (1, 5)
        rows += [_row(f"Q{index}", "P1_vector", 5, judge=left_value), _row(f"Q{index}", "P2_bm25", 5, judge=right_value)]
    out = agg.compare(rows, "P1_vector", "P2_bm25")
    assert out["mean_difference"] > 0
    assert (out["ci_low"] <= 0 <= out["ci_high"]) == (not out["significant"])


# --- end to end ---------------------------------------------------------------

def test_the_report_is_built_from_the_database_alone():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, quadrant TEXT, document_id TEXT);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT, query_id TEXT,
            pipeline TEXT, k_value INTEGER, precision_at_k REAL, recall_at_k REAL,
            evidence_hit INTEGER, citation_match INTEGER, token_f1 REAL, exact_match INTEGER,
            judge_score INTEGER, latency_sec REAL, input_tokens INTEGER, output_tokens INTEGER);
        INSERT INTO queries VALUES ('Q1','Q1_Direct_Text','AAPL_2023');
        INSERT INTO results VALUES ('R1','PQ','Q1','P1_vector',5,0.2,1.0,1,1,0.5,1,5,3.0,100,20);
        INSERT INTO results VALUES ('R2','PQ','Q1','P2_bm25',5,0.0,0.0,0,0,0.0,0,1,4.0,110,25);
    """)
    report = agg.build_report(conn, "PQ")
    assert report["cells"] == 2
    assert report["overall"]["P1_vector"]["judge_mean"] == 5
    assert report["overall"]["P2_bm25"]["hit_rate"] == 0.0
    assert "Q1_Direct_Text|P1_vector" in report["by_quadrant"]
    assert any(comparison["left"] == "P1_vector" and comparison["right"] == "P2_bm25" for comparison in report["comparisons"])
