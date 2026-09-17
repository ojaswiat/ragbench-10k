import pytest

from dataset_generation.gq_label_export import render_label_markdown
from dataset_generation.gq_label_import import parse_label_markdown

_GOLDEN = [
    {
        "query_id": "Q1_GQ_0001",
        "quadrant": "Q1_Direct_Text",
        "query_text": "What was total revenue?",
        "ground_truth_answer": "$100 million",
        "gt_citations": ["n1"],
        "document_id": "AAPL_2023",
    },
]


def test_render_label_markdown_includes_query_and_blank_fields():
    markdown = render_label_markdown(_GOLDEN)
    assert "Q1_GQ_0001" in markdown
    assert "What was total revenue?" in markdown
    assert "$100 million" in markdown
    assert "Score (0-100):" in markdown
    assert "Good Example (yes/no):" in markdown
    assert "Why this answer is good:" in markdown


def test_parse_label_markdown_round_trips_filled_fields():
    filled = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):** 90
**Good Example (yes/no):** yes
**Why this answer is good:** Unambiguous single-fact retrieval, directly stated.

---
"""
    parsed = parse_label_markdown(filled)
    assert parsed == [{
        "query_id": "Q1_GQ_0001",
        "human_score": 90,
        "human_reasoning": "Unambiguous single-fact retrieval, directly stated.",
        "is_good": True,
    }]


def test_parse_label_markdown_skips_unfilled_entries():
    unfilled = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):**
**Good Example (yes/no):**
**Why this answer is good:**

---
"""
    assert parse_label_markdown(unfilled) == []


def test_parse_label_markdown_handles_multiple_entries_without_bleeding():
    two_entries = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):** 90
**Good Example (yes/no):** yes
**Why this answer is good:** Unambiguous single-fact retrieval, directly stated.

---

## Q3_GQ_0002

**Quadrant:** Q3_Direct_Table
**Document:** MSFT_2024
**Query:** What was total operating expenses?
**Ground truth:** $52,866 million

**Score (0-100):** 70
**Good Example (yes/no):** no
**Why this answer is good:** Clean cell extraction, single correct value.

---
"""
    parsed = parse_label_markdown(two_entries)
    assert parsed == [
        {
            "query_id": "Q1_GQ_0001",
            "human_score": 90,
            "human_reasoning": "Unambiguous single-fact retrieval, directly stated.",
            "is_good": True,
        },
        {
            "query_id": "Q3_GQ_0002",
            "human_score": 70,
            "human_reasoning": "Clean cell extraction, single correct value.",
            "is_good": False,
        },
    ]


def test_parse_label_markdown_raises_on_non_integer_score():
    bad = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):** high
**Good Example (yes/no):** yes
**Why this answer is good:** Unambiguous single-fact retrieval, directly stated.

---
"""
    with pytest.raises(ValueError, match="Q1_GQ_0001"):
        parse_label_markdown(bad)


def test_parse_label_markdown_raises_on_unrecognized_is_good_value():
    bad = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):** 90
**Good Example (yes/no):** maybe
**Why this answer is good:** Unambiguous single-fact retrieval, directly stated.

---
"""
    with pytest.raises(ValueError, match="Q1_GQ_0001"):
        parse_label_markdown(bad)


def test_parse_label_markdown_handles_entry_missing_good_example_line():
    """An entry missing only the Good Example line must still parse, not
    silently vanish from the results."""
    old_format = """## Q1_GQ_0001

**Quadrant:** Q1_Direct_Text
**Document:** AAPL_2023
**Query:** What was total revenue?
**Ground truth:** $100 million

**Score (0-100):** 90
**Why this answer is good:** Unambiguous single-fact retrieval, directly stated.

---
"""
    parsed = parse_label_markdown(old_format)
    assert parsed == [{
        "query_id": "Q1_GQ_0001",
        "human_score": 90,
        "human_reasoning": "Unambiguous single-fact retrieval, directly stated.",
        "is_good": None,
    }]
