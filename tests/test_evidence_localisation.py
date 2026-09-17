"""evidence_localisation: reducing gt_citations to the minimal evidence set.

The rule must shrink an over-broad answer key without ever inventing evidence,
and must refuse rather than guess when it cannot show that a node yields the
answer. The safety properties (never introduces an uncited node, never empties
a row, never writes an unverified repair) matter more than the shrinking, so
they are asserted directly rather than inferred from a happy-path result.
"""
import json
import sqlite3

import pytest

import evidence_localisation as el


def _row(citations, answer, query="What is the value?"):
    return {"query_id": "Q1", "query_text": query,
            "ground_truth_answer": answer, "gt_citations": list(citations)}


def _wide(citations):
    """A citation list long enough to count as a defect."""
    return list(citations) + [f"filler_{index}" for index in range(el.MAX_GOLD_SET)]


# --- rows that carry no defect ------------------------------------------------

def test_a_row_within_the_max_gold_set_is_returned_unchanged():
    row = _row(["a", "b"], "$100")
    after, reasons = el.localise(row, {"a": "irrelevant", "b": "$100 here"})
    assert after == ["a", "b"]
    assert reasons == {}


def test_a_short_row_is_untouched_even_when_a_citation_supports_nothing():
    row = _row(["a", "b", "c"], "$100")
    after, _ = el.localise(row, {"a": "", "b": "", "c": "$100"})
    assert after == ["a", "b", "c"], "rows already able to reach recall 1.0 must not be touched"


# --- tier A: the answer is stated outright ------------------------------------

def test_only_nodes_containing_the_answer_are_retained():
    cited = _wide(["hit", "miss"])
    contents = {cited_id: "unrelated prose" for cited_id in cited}
    contents["hit"] = "Net income was $72,361 million for the year."
    after, reasons = el.localise(_row(cited, "$72,361 million"), contents)
    assert after == ["hit"]
    assert reasons["hit"] == "contains answer literal"


def test_a_thousands_separator_does_not_hide_the_answer():
    cited = _wide(["hit"])
    contents = {cited_id: "" for cited_id in cited}
    contents["hit"] = "the figure 1013870 appears unseparated"
    after, _ = el.localise(_row(cited, "1,013,870 shares"), contents)
    assert after == ["hit"]


def test_single_digit_answers_are_not_matched_on_digits_alone():
    assert el.salient_numbers("7") == set()
    assert "72" in el.salient_numbers("72 facilities")


def test_a_prose_answer_needs_a_contiguous_run_not_shared_vocabulary():
    answer = "the acquisition closed on May 1 2023"
    grams = el.answer_ngrams(answer)
    assert grams
    scrambled = "2023 closed the on 1 acquisition May"
    assert not el.contains_answer_literal(scrambled, "the acquisition closed peacefully today")


# --- tier B: the answer is derived --------------------------------------------

@pytest.mark.parametrize("text, answer, expected", [
    ("| Europe | 4,900 |\n| Asia | 1,243 |\n| US | 2,973 |", "6,143", "subset sum"),
    ("2023: $3,089 and 2022: $4,022", "$933 million", "difference"),
    ("| 125,000 | $147.61 |\n| 1,265,000 | $156.76 |", "$216,752,650", "quantity by price"),
    ("/s/ A\n/s/ B\n/s/ C", "3", "signature markers"),
])
def test_a_derived_answer_is_traced_back_to_its_operands(text, answer, expected):
    cited = _wide(["table"])
    contents = {cited_id: "" for cited_id in cited}
    contents["table"] = text
    after, reasons = el.localise(_row(cited, answer), contents)
    assert after == ["table"], f"expected the operand node to be retained for {answer}"
    assert expected in reasons["table"]


def test_prose_about_a_table_is_not_mistaken_for_the_table():
    """Word overlap alone would keep the preamble; the arithmetic test must not."""
    cited = _wide(["preamble", "table"])
    contents = {cited_id: "" for cited_id in cited}
    contents["preamble"] = ("The following table provides information concerning the "
                            "aggregate number of shares to be sold by each director.")
    contents["table"] = "| Dimon | 1,000,000 |\n| Other | 13,870 |"
    after, _ = el.localise(_row(cited, "1,013,870 shares",
                                query="What is the total aggregate number of shares to be sold?"),
                           contents)
    assert after == ["table"]
    assert "preamble" not in after


# --- safety properties --------------------------------------------------------

def test_an_unverifiable_row_is_left_exactly_as_it_was():
    cited = _wide(["a", "b"])
    contents = {cited_id: "nothing that yields the answer" for cited_id in cited}
    after, reasons = el.localise(_row(cited, "$999,999,999"), contents)
    assert after == cited, "a row the rule cannot verify must not be narrowed"
    assert reasons == {}


def test_the_rule_never_introduces_a_node_that_was_not_already_cited():
    cited = _wide(["hit"])
    contents = {cited_id: "" for cited_id in cited}
    contents["hit"] = "$100"
    contents["never_cited"] = "$100 also appears here"
    after, _ = el.localise(_row(cited, "$100"), contents)
    assert set(after) <= set(cited)
    assert "never_cited" not in after


def test_the_rule_never_empties_a_row():
    cited = _wide(["a"])
    after, _ = el.localise(_row(cited, "unmatchable"), {cited_id: "" for cited_id in cited})
    assert after


def test_a_missing_node_is_treated_as_unsupported_not_as_a_crash():
    cited = _wide(["absent"])
    after, reasons = el.localise(_row(cited, "$100"), {})
    assert after == cited
    assert reasons == {}


def test_subset_sum_gives_up_instead_of_hanging_on_a_large_table():
    from decimal import Decimal
    values = [Decimal(2 ** index) for index in range(60)]
    assert el._subset_sums_to(values, Decimal(-1)) is False


# --- planning and applying ----------------------------------------------------

def _plan_for(citations, answer, contents):
    return el.plan_repairs([_row(citations, answer)], contents)


def test_planning_is_idempotent_over_its_own_output():
    cited = _wide(["hit"])
    contents = {cited_id: "" for cited_id in cited}
    contents["hit"] = "$100"
    first = _plan_for(cited, "$100", contents)
    assert len(first) == 1

    repaired = _row(first[0]["after"], "$100")
    assert el.plan_repairs([repaired], contents) == [], "a second run must plan nothing"


def test_a_row_already_fully_load_bearing_is_not_replanned():
    cited = [f"n{index}" for index in range(el.MAX_GOLD_SET + 1)]
    contents = {cited_id: "$100" for cited_id in cited}
    assert _plan_for(cited, "$100", contents) == []


def test_an_unverified_row_is_planned_but_flagged():
    cited = _wide(["a"])
    plan = _plan_for(cited, "$999,999,999", {cited_id: "" for cited_id in cited})
    assert len(plan) == 1
    assert plan[0]["verified"] is False
    assert plan[0]["after"] == plan[0]["before"]


def test_apply_repairs_refuses_to_write_an_unverified_entry():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE queries (query_id TEXT PRIMARY KEY, gt_citations TEXT)")
    conn.execute("INSERT INTO queries VALUES ('Q1', ?)", (json.dumps(["a", "b"]),))
    plan = [{"query_id": "Q1", "before": ["a", "b"], "after": ["a"],
             "reasons": {}, "verified": False}]
    assert el.apply_repairs(conn, "queries", plan) == 0
    stored = conn.execute("SELECT gt_citations FROM queries").fetchone()[0]
    assert json.loads(stored) == ["a", "b"]


def test_apply_repairs_writes_a_verified_entry():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE queries (query_id TEXT PRIMARY KEY, gt_citations TEXT)")
    conn.execute("INSERT INTO queries VALUES ('Q1', ?)", (json.dumps(["a", "b"]),))
    plan = [{"query_id": "Q1", "before": ["a", "b"], "after": ["a"],
             "reasons": {"a": "contains answer literal"}, "verified": True}]
    assert el.apply_repairs(conn, "queries", plan) == 1
    stored = conn.execute("SELECT gt_citations FROM queries").fetchone()[0]
    assert json.loads(stored) == ["a"]
