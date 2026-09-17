"""Deterministic Judge metrics.

Citation validity, retrieval precision and recall, and answer token overlap are
all computed in code with no LLM call, so they stay reproducible and cannot be
swayed by the Judge model.
"""
from judge.metrics import citation_audit, precision_at_k, recall_at_k, token_f1


# --- citation_audit: cited node ids must be a subset of gt_citations ---


def test_citation_audit_subset_is_true():
    assert citation_audit(["n1"], ["n1", "n2"]) is True


def test_citation_audit_exact_set_is_true():
    assert citation_audit(["n1", "n2"], ["n2", "n1"]) is True


def test_citation_audit_extra_citation_is_false():
    assert citation_audit(["n1", "n3"], ["n1", "n2"]) is False


def test_citation_audit_empty_cited_is_true():
    """Citing nothing introduces no invalid citation, so it is vacuously a subset."""
    assert citation_audit([], ["n1"]) is True


def test_citation_audit_cited_against_empty_gt_is_false():
    assert citation_audit(["n1"], []) is False


# --- precision_at_k: hits in top-k over k (Architecture.md §3.3 worked row) ---


def test_precision_at_k_matches_worked_example():
    # gt has one node; top-5 retrieval surfaces it once -> 1/5 = 0.20.
    assert precision_at_k(["AAPL_2025_n0421", "AAPL_2025_n0420"], ["AAPL_2025_n0421"], 5) == 0.20


def test_precision_at_k_all_relevant():
    assert precision_at_k(["n1", "n2", "n3"], ["n1", "n2", "n3"], 3) == 1.0


def test_precision_at_k_truncates_to_k():
    # Only the first k retrieved count; the relevant node past k is ignored.
    assert precision_at_k(["x", "y", "n1"], ["n1"], 2) == 0.0


def test_precision_at_k_zero_k_is_zero():
    assert precision_at_k(["n1"], ["n1"], 0) == 0.0


# --- recall_at_k: fraction of gt found anywhere in retrieved ---


def test_recall_at_k_full():
    assert recall_at_k(["n1", "n2", "x"], ["n1", "n2"]) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["n1", "x"], ["n1", "n2"]) == 0.5


def test_recall_at_k_none_found():
    assert recall_at_k(["x", "y"], ["n1"]) == 0.0


def test_recall_at_k_empty_gt_is_zero():
    assert recall_at_k(["x"], []) == 0.0


# --- token_f1: SQuAD-style overlap, citation markers ignored ---


def test_token_f1_identical_is_one():
    assert token_f1("net sales rose", "net sales rose") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_token_f1_partial_overlap():
    # pred={a,b,c}, gt={b,c,d}: common=2, p=2/3, r=2/3, f1=2/3.
    assert token_f1("a b c", "b c d") == round(2 / 3, 10) or abs(token_f1("a b c", "b c d") - 2 / 3) < 1e-9


def test_token_f1_ignores_citation_markers():
    """The [[node:...]] marker is scaffolding, not answer content."""
    assert token_f1("net sales rose [[node:AAPL_2025_n0421]]", "net sales rose") == 1.0


def test_token_f1_is_case_and_punctuation_insensitive():
    assert token_f1("Net sales, rose!", "net sales rose") == 1.0


def test_token_f1_both_empty_is_one():
    assert token_f1("", "") == 1.0


def test_token_f1_one_empty_is_zero():
    assert token_f1("", "something") == 0.0
