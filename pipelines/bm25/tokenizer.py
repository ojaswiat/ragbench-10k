"""Custom BM25 tokenizer: preserves numbers, decimals, percentages and
currency amounts, strips table-markdown pipes, applies no stemming.

Called identically at index time and query time, so scores are never
computed against mismatched vocabularies.
"""
import re

_TOKEN_PATTERN = re.compile(r"[$£€]?\d+(?:,\d{3})*(?:\.\d+)?%?|[A-Za-z]+")


def tokenize(text: str) -> list[str]:
    text = text.replace("|", " ")
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
