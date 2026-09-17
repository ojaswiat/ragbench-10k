"""Reduces a financial figure written as free text to a single Decimal.

"$394.3B" and "394,300 million" collapse to the same Decimal, so the
comparison in judge.metrics.exact_match is scale- and notation-independent.
Anything that is not a bare number, with optional currency mark, thousands
separators, sign, and one magnitude suffix, returns None so the caller can
fall back to a string comparison.
"""
from decimal import Decimal, InvalidOperation

# Only the currency marks and separators handled here, kept deliberately
# small so a stray symbol makes a string non-numeric rather than being
# silently discarded.
_STRIP = str.maketrans("", "", "$£€,")

_WORD_MULTIPLIERS = {
    "thousand": 3,
    "million": 6,
    "billion": 9,
    "trillion": 12,
}
_LETTER_MULTIPLIERS = {"k": 3, "m": 6, "b": 9, "t": 12}


def _split_multiplier(numeric_text: str) -> tuple[str, int]:
    """Return (numeric core, power-of-ten exponent) after peeling one suffix.

    Word suffixes are checked before letter suffixes so "trillion" is never
    mistaken for a bare "t". A letter suffix only counts when a digit, dot or
    space sits before it, so a non-numeric token like "b" on its own does not
    masquerade as a magnitude.
    """
    for word, exp in _WORD_MULTIPLIERS.items():
        if numeric_text.endswith(word):
            return numeric_text[: -len(word)].strip(), exp
    if len(numeric_text) > 1 and numeric_text[-1] in _LETTER_MULTIPLIERS and (numeric_text[-2].isdigit() or numeric_text[-2] in ". "):
        return numeric_text[:-1].strip(), _LETTER_MULTIPLIERS[numeric_text[-1]]
    return numeric_text, 0


def normalize_numeric(text: str) -> Decimal | None:
    cleaned = text.strip().lower().translate(_STRIP).strip()
    if not cleaned:
        return None

    core, exp = _split_multiplier(cleaned)
    try:
        value = Decimal(core)
    except InvalidOperation:
        return None
    return value * (Decimal(10) ** exp)


__all__ = ["normalize_numeric"]
