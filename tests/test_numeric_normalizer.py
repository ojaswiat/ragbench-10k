"""normalize_numeric: currency and separator stripping, suffix expansion.

Exact Match needs "$394.3B" and "394,300 million" to reduce to the same
Decimal, so a Q3 table answer scores correctly however the figure is spelt.
"""
from decimal import Decimal

import pytest

from judge.numeric_normalizer import normalize_numeric


def test_plain_integer():
    assert normalize_numeric("42") == Decimal("42")


def test_plain_decimal():
    assert normalize_numeric("3.14") == Decimal("3.14")


def test_strips_dollar_and_thousands_separators():
    assert normalize_numeric("$1,234.56") == Decimal("1234.56")


def test_strips_pound_and_euro():
    assert normalize_numeric("£1,000") == Decimal("1000")
    assert normalize_numeric("€2,500.50") == Decimal("2500.50")


def test_letter_suffix_billions():
    assert normalize_numeric("$394.3B") == Decimal("394300000000")


def test_word_suffix_million():
    assert normalize_numeric("394,300 million") == Decimal("394300000000")


def test_dollar_billions_equals_word_millions():
    """The two spellings collapse to one value."""
    assert normalize_numeric("$394.3B") == normalize_numeric("394,300 million")


def test_letter_suffixes_k_m_b_t():
    assert normalize_numeric("5K") == Decimal("5000")
    assert normalize_numeric("5M") == Decimal("5000000")
    assert normalize_numeric("5B") == Decimal("5000000000")
    assert normalize_numeric("5T") == Decimal("5000000000000")


def test_word_suffixes_thousand_through_trillion():
    assert normalize_numeric("5 thousand") == Decimal("5000")
    assert normalize_numeric("5 million") == Decimal("5000000")
    assert normalize_numeric("5 billion") == Decimal("5000000000")
    assert normalize_numeric("5 trillion") == Decimal("5000000000000")


def test_case_insensitive_suffix():
    assert normalize_numeric("1.5b") == normalize_numeric("1.5B")
    assert normalize_numeric("2 Million") == Decimal("2000000")


def test_negative_value():
    assert normalize_numeric("-3.2M") == Decimal("-3200000")


def test_space_separated_letter_suffix():
    assert normalize_numeric("5 K") == Decimal("5000")


def test_surrounding_whitespace():
    assert normalize_numeric("  $10  ") == Decimal("10")


@pytest.mark.parametrize("text", ["", "   ", "hello", "n/a", "4.2%", "twelve", "$", "B"])
def test_non_numeric_returns_none(text):
    assert normalize_numeric(text) is None
