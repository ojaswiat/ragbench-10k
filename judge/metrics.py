"""Deterministic, LLM-free scoring metrics for a single results row.

Citation matching runs here in code, never delegated to the Judge, which
keeps the metric vector reproducible from the stored output alone. Node-id
arguments are the plain list[str] that database_manager returns.
"""
import re
import string
from decimal import Decimal

from judge.numeric_normalizer import normalize_numeric

# Same marker shape the Answerer emits and parses; stripped here before any
# text-overlap metric so citation scaffolding never counts as answer content.
_CITATION_PATTERN = re.compile(r"\[\[node:[\w\-]+\]\]")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Implicit quadrants have no single canonical figure or phrase to match
# against, so Exact Match is undefined for them (stored as NULL).
_EM_UNDEFINED_QUADRANTS = frozenset({"Q2_Implicit_Text", "Q4_Implicit_Table"})

# A currency-marked or suffixed number as it appears mid-sentence. Each hit is
# handed to normalize_numeric, which owns the actual parsing; this only has to
# be generous enough to bracket the span (optional currency, thousands
# separators, decimals, and one magnitude word/letter).
_NUMBER_SPAN = re.compile(
    r"[$£€]?\s*-?\d[\d,]*(?:\.\d+)?\s*(?:thousand|million|billion|trillion|[kmbt])?\b",
    re.IGNORECASE,
)


def citation_audit(cited_node_ids: list[str], gt_citations: list[str]) -> bool:
    """True when every cited node id is one of the ground-truth citations.

    An empty citation list is vacuously valid: an answer that cited nothing
    introduced no wrong citation. An id cited when there are no valid
    citations at all is a violation.
    """
    return set(cited_node_ids) <= set(gt_citations)


def precision_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Ground-truth nodes among the top-k retrieved, divided by k.

    The denominator is k itself, not the number actually retrieved, so one
    hit in a k=5 slot scores 0.20 rather than 1.0; a pipeline that pads its
    top-k with irrelevant nodes is penalised for it.
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & set(ground_truth))
    return hits / k


def recall_at_k(retrieved: list[str], ground_truth: list[str]) -> float:
    """Fraction of ground-truth nodes that appear anywhere in retrieved.

    No k argument: the retriever has already returned its k nodes, so recall
    is measured over the whole returned set. An empty ground-truth set scores
    0.0 rather than a vacuous 1.0, so it never inflates the evidence_hit
    signal derived from recall > 0.
    """
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    return len(set(retrieved) & gt_set) / len(gt_set)


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens, citation markers and punctuation removed."""
    stripped = _CITATION_PATTERN.sub(" ", text)
    return stripped.lower().translate(_PUNCT_TABLE).split()


def token_f1(output_text: str, gt_text: str) -> float:
    """SQuAD-style token-overlap F1 between the answer and the ground truth.

    Multiplicity is honoured (a token repeated in both sides counts as many
    times as the smaller count). Two empty token sets score 1.0 (they agree);
    exactly one empty side scores 0.0.
    """
    pred = _tokens(output_text)
    truth = _tokens(gt_text)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0

    common = 0
    remaining = list(truth)
    for tok in pred:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0

    precision = common / len(pred)
    recall = common / len(truth)
    return 2 * precision * recall / (precision + recall)


def _within_tolerance(candidate: Decimal, target: Decimal, tolerance: float) -> bool:
    """Relative comparison so notation-equal figures match within epsilon.

    The gap is measured against the larger magnitude, so the tolerance means
    the same thing whichever side is bigger; a zero target demands an exact
    zero rather than dividing by nothing.
    """
    scale = max(abs(candidate), abs(target))
    if scale == 0:
        return candidate == target
    return abs(candidate - target) <= Decimal(str(tolerance)) * scale


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True when `needle` appears as a contiguous run of tokens in `haystack`."""
    if not needle:
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


def exact_match(
    output_text: str,
    gt_text: str,
    *,
    quadrant: str | None = None,
    numeric_tolerance: float = 0.01,
) -> bool | None:
    """Whether the ground-truth answer can be found inside the model output.

    Returns None for the implicit quadrants Q2/Q4 (EM is Q1/Q3-only). For a
    numeric ground truth, any figure in the output that normalises within
    `numeric_tolerance` (relative) of it counts, so "$394.3B" matches out of a
    full sentence and "394,300 million" is treated as equal. For a text
    ground truth, the answer's tokens must appear contiguously in the output
    (case- and punctuation-insensitive).
    """
    if quadrant in _EM_UNDEFINED_QUADRANTS:
        return None

    cleaned = _CITATION_PATTERN.sub(" ", output_text)
    gt_value = normalize_numeric(gt_text)

    if gt_value is not None:
        for span in _NUMBER_SPAN.findall(cleaned):
            candidate = normalize_numeric(span)
            if candidate is not None and _within_tolerance(candidate, gt_value, numeric_tolerance):
                return True
        return False

    return _contains_subsequence(_tokens(cleaned), _tokens(gt_text))


__all__ = [
    "citation_audit",
    "precision_at_k",
    "recall_at_k",
    "token_f1",
    "exact_match",
]
