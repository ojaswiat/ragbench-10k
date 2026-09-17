from dataset_generation.cross_check import (
    EMBEDDING_SIMILARITY_THRESHOLD,
    check_query,
    citations_overlap,
    diagnose_rejection,
    embedding_similarity,
    embedding_similarity_ok,
    extract_numbers,
    values_match,
)


def test_citations_overlap_true_on_any_shared_node():
    assert citations_overlap(["n1", "n2"], ["n2", "n9"]) is True


def test_citations_overlap_false_on_disjoint_sets():
    assert citations_overlap(["n1", "n2"], ["n8", "n9"]) is False


def test_extract_numbers_handles_currency_and_commas():
    assert extract_numbers("$52,866 million") == [52866.0]


def test_extract_numbers_handles_multiple_values_in_order():
    assert extract_numbers(
        "Operating expenses decreased 4.2% YoY. Factoring in the $15M impairment, normalized expenses rose 1.1%."
    ) == [4.2, 15.0, 1.1]


def test_values_match_exact_after_normalization():
    assert values_match("$52,866 million", "52866") is True


def test_values_match_within_tolerance():
    assert values_match("4.2%", "4.205%") is True


def test_values_match_false_on_real_difference():
    assert values_match("4.2%", "6.7%") is False


def test_values_match_false_on_mismatched_number_count():
    # A gt number ("1.1%") absent from the critic answer still fails;
    # subset matching permits extra numbers on the critic side, not missing ones.
    assert values_match("4.2% and 1.1%", "4.2%") is False


def test_values_match_true_when_critic_has_extra_unrelated_number():
    # An equally-correct answer with an extra unrelated number (e.g. a
    # year) must not be rejected purely on count mismatch.
    assert values_match("$100 million", "$100 million in 2025") is True


def test_values_match_false_on_genuinely_wrong_number():
    assert values_match("$100 million", "$150 million") is False


def test_values_match_falls_back_to_text_equality_with_no_numbers():
    assert values_match("increased", "increased") is True
    assert values_match("increased", "decreased") is False


def test_check_query_accepts_on_overlap_and_matching_value():
    assert check_query(
        gt_citations=["n1", "n2"],
        gt_answer="$100 million",
        critic_cited_ids=["n2", "n9"],
        critic_answer="100 million",
    ) is True


def test_check_query_rejects_on_no_citation_overlap():
    assert check_query(
        gt_citations=["n1", "n2"],
        gt_answer="$100 million",
        critic_cited_ids=["n8", "n9"],
        critic_answer="100 million",
    ) is False


def test_check_query_rejects_on_value_mismatch():
    assert check_query(
        gt_citations=["n1", "n2"],
        gt_answer="$100 million",
        critic_cited_ids=["n2"],
        critic_answer="200 million",
    ) is False


def test_diagnose_rejection_reports_citation_mismatch():
    reason = diagnose_rejection(
        gt_citations=["n1", "n2"],
        gt_answer="$100 million",
        critic_cited_ids=["n8", "n9"],
        critic_answer="100 million",
    )
    assert "citation" in reason.lower()
    assert "n8" in reason and "n9" in reason
    assert "n1" in reason and "n2" in reason


def test_diagnose_rejection_reports_value_mismatch():
    reason = diagnose_rejection(
        gt_citations=["n1", "n2"],
        gt_answer="$100 million",
        critic_cited_ids=["n2"],
        critic_answer="200 million",
    )
    assert "value" in reason.lower()
    assert "200 million" in reason
    assert "$100 million" in reason


# -- Embedding similarity gate (third gate) -----------------------------------

_SEMANTICALLY_SIMILAR_A = (
    "Net revenue increased to $100 million in fiscal 2025."
)
_SEMANTICALLY_SIMILAR_B = (
    "The company reported $100 million in net revenue for fiscal year 2025."
)
_SEMANTICALLY_UNRELATED_A = "Net revenue increased to $100 million in fiscal 2025."
_SEMANTICALLY_UNRELATED_B = "Operating expenses decreased by 4.2% year over year."


def test_embedding_similarity_high_for_similar_answers():
    score = embedding_similarity(_SEMANTICALLY_SIMILAR_A, _SEMANTICALLY_SIMILAR_B)
    assert score >= EMBEDDING_SIMILARITY_THRESHOLD


def test_embedding_similarity_low_for_unrelated_answers():
    score = embedding_similarity(_SEMANTICALLY_UNRELATED_A, _SEMANTICALLY_UNRELATED_B)
    assert score < EMBEDDING_SIMILARITY_THRESHOLD


def test_embedding_similarity_ok_matches_threshold():
    assert embedding_similarity_ok(_SEMANTICALLY_SIMILAR_A, _SEMANTICALLY_SIMILAR_B) is True
    assert embedding_similarity_ok(_SEMANTICALLY_UNRELATED_A, _SEMANTICALLY_UNRELATED_B) is False


def test_check_query_rejects_when_numbers_coincidentally_match_but_text_unrelated():
    # Numeric subset and citation overlap both pass; the answers are about
    # unrelated topics sharing a number, so only the embedding gate catches this.
    assert check_query(
        gt_citations=["n1", "n2"],
        gt_answer="Net income was $100 million.",
        critic_cited_ids=["n2"],
        critic_answer="Headcount reached $100 million... employees across 12 offices.",
    ) is False


def test_check_query_three_gate_and_logic():
    gt_citations = ["n1", "n2"]
    gt_answer = "Net revenue increased to $100 million in fiscal 2025."
    similar_answer = "The company reported $100 million in net revenue for fiscal year 2025."
    unrelated_answer = "Operating expenses decreased by 4.2% year over year."

    # All three gates pass -> accept.
    assert check_query(
        gt_citations=gt_citations,
        gt_answer=gt_answer,
        critic_cited_ids=["n2", "n9"],
        critic_answer=similar_answer,
    ) is True

    # Citation gate fails alone (values + embedding would pass).
    assert check_query(
        gt_citations=gt_citations,
        gt_answer=gt_answer,
        critic_cited_ids=["n8", "n9"],
        critic_answer=similar_answer,
    ) is False

    # Numeric gate fails alone (citations + embedding would pass: the
    # restated sentence with a different number is still semantically
    # similar prose).
    wrong_number_answer = "The company reported $150 million in net revenue for fiscal year 2025."
    assert check_query(
        gt_citations=gt_citations,
        gt_answer=gt_answer,
        critic_cited_ids=["n2"],
        critic_answer=wrong_number_answer,
    ) is False

    # Embedding gate fails alone; this variant shares gt_answer's number so
    # the numeric gate cannot also fail, isolating the embedding gate.
    assert check_query(
        gt_citations=gt_citations,
        gt_answer=gt_answer,
        critic_cited_ids=["n2"],
        critic_answer="Headcount reached $100 million... employees across 12 offices.",
    ) is False


def test_diagnose_rejection_reports_embedding_similarity_too_low():
    reason = diagnose_rejection(
        gt_citations=["n1", "n2"],
        gt_answer="Net income was $100 million.",
        critic_cited_ids=["n2"],
        critic_answer="Headcount reached $100 million... employees across 12 offices.",
    )
    assert "embedding" in reason.lower()
    assert "similarity" in reason.lower()
