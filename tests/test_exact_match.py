"""exact_match: Q1/Q3 only, extraction and containment semantics.

A figure or short answer counts as an exact match when it can be found inside
the model's sentence, not only when the whole sentence equals the ground truth.
EM is undefined (None) for the implicit quadrants Q2 and Q4.
"""
from judge.metrics import exact_match


# --- quadrant gating: EM is Q1/Q3 only ---


def test_exact_match_none_for_q2():
    assert exact_match("anything at all", "anything at all", quadrant="Q2_Implicit_Text") is None


def test_exact_match_none_for_q4():
    assert exact_match("42", "42", quadrant="Q4_Implicit_Table") is None


# --- numeric ground truth: extract the figure from the sentence ---


def test_extracts_figure_from_sentence():
    out = "Apple's FY2025 net sales were $394.3B. [[node:AAPL_2025_n0421]]"
    assert exact_match(out, "$394.3B", quadrant="Q3_Direct_Table") is True


def test_equivalent_spelling_matches():
    assert exact_match("net sales were 394,300 million", "$394.3B", quadrant="Q3_Direct_Table") is True


def test_within_default_tolerance():
    # 394 vs 394.3: 0.3B gap is inside 1% of 394.3B (~3.94B).
    assert exact_match("net sales were $394B", "$394.3B", quadrant="Q3_Direct_Table") is True


def test_outside_default_tolerance():
    # 390 vs 394.3: 4.3B gap exceeds 1% of 394.3B.
    assert exact_match("net sales were $390B", "$394.3B", quadrant="Q3_Direct_Table") is False


def test_tolerance_param_widens_match():
    assert exact_match("about $390B", "$394.3B", quadrant="Q3_Direct_Table", numeric_tolerance=0.05) is True


def test_numeric_gt_no_figure_in_output_is_false():
    assert exact_match("the figure is not disclosed", "$394.3B", quadrant="Q3_Direct_Table") is False


# --- text ground truth: containment, case/punctuation insensitive ---


def test_text_answer_contained_in_sentence():
    out = "The chief executive is Tim Cook. [[node:AAPL_2025_n0001]]"
    assert exact_match(out, "Tim Cook", quadrant="Q1_Direct_Text") is True


def test_text_answer_case_insensitive():
    assert exact_match("chief executive TIM COOK", "tim cook", quadrant="Q1_Direct_Text") is True


def test_text_answer_absent_is_false():
    assert exact_match("The CFO is Luca Maestri.", "Tim Cook", quadrant="Q1_Direct_Text") is False


def test_defaults_to_scoring_when_quadrant_omitted():
    # No quadrant given -> compute (do not return None).
    assert exact_match("Tim Cook", "Tim Cook") is True
