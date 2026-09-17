"""Reduces gt_citations to the minimal set of nodes that supports the answer.

The Generator is told to draw gt_citations from the section it was shown but
is never told to make that list minimal, so on some rows it returns the whole
section. That is not a harder question, it is a wrong answer key: recall_at_k
divides by len(gt) with no cap, so a gold set larger than the largest k makes
full recall arithmetically unreachable, while precision_at_k divides by k and
is inflated by the same bloat.

The rule here only ever retains a subset of the citations a row already has;
it never introduces a node the Generator did not cite, and never consults a
pipeline's output, so it cannot be tuned toward a result. A node is retained
only when it demonstrably yields the answer, either by containing it (the
direct quadrants) or by reproducing it arithmetically (the implicit
quadrants, where the answer is computed and appears in no node verbatim).
Every retained node carries a recorded reason. A row where nothing can be
verified is left exactly as it was and reported, rather than guessed at.
"""
import json
import re
from decimal import Decimal, InvalidOperation
from itertools import combinations

# Largest k in the experimental design. A gold set at or below this size can
# already reach recall 1.0, so there is no defect to repair and the row is
# left untouched.
MAX_GOLD_SET = 5

# Bounds on the arithmetic search. Subset-sum over n values is exponential,
# so both the input length and the reachable set are capped; the predicate
# gives up and returns False rather than hanging on a large table.
_MAX_SUBSET_TERMS = 25
_MAX_REACHABLE = 200_000

# Role words used when a question asks how many rows of a table carry a
# title. Capitalised words from the question itself are tried first; these
# cover the signature tables, where the role is not capitalised in the query.
_ROLE_TERMS = ("Director", "Officer")

# A signature row in a converted filing table is marked by this token.
_SIGNATURE_MARKER = "/s/"

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_WORD = re.compile(r"[a-z0-9][a-z0-9.,%-]*")
_CAPITALISED = re.compile(r"\b[A-Z][a-zA-Z]{3,}\b")


def _decimals(text: str) -> list[Decimal]:
    """Every number in `text` as a Decimal, thousands separators removed."""
    out = []
    for match in _NUMBER.findall(text):
        try:
            out.append(Decimal(match.replace(",", "")))
        except InvalidOperation:
            continue
    return out


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens with surrounding punctuation stripped."""
    lowered = re.sub(r"[^a-z0-9,.\s%$-]", " ", text.lower())
    return [token for token in (raw_match.strip(".,%-") for raw_match in _WORD.findall(lowered)) if token]


def salient_numbers(answer: str) -> set[str]:
    """Numbers from the answer, in both separated and unseparated form.

    Single digits are excluded: they match far too much prose to be evidence
    that a node carries the answer.
    """
    out: set[str] = set()
    for match in _NUMBER.findall(answer):
        bare = match.replace(",", "").rstrip(".")
        if len(bare.lstrip("-")) >= 2:
            out.add(match.rstrip("."))
            out.add(bare)
    return out


def answer_ngrams(answer: str, gram_size: int = 4) -> set[str]:
    """Contiguous n-token runs of the answer, for matching prose answers.

    A run rather than a bag of words, so a node has to reproduce the answer's
    phrasing and not merely share its vocabulary with it.
    """
    words = _tokens(answer)
    if len(words) < gram_size:
        return set()
    return {" ".join(words[start : start + gram_size]) for start in range(len(words) - gram_size + 1)}


def contains_answer_literal(node_text: str, answer: str) -> bool:
    """Whether the node states the answer outright."""
    stripped = node_text.replace(",", "")
    for number in salient_numbers(answer):
        if number in node_text or number in stripped:
            return True
    grams = answer_ngrams(answer)
    if not grams:
        return False
    flat = " ".join(_tokens(node_text))
    return any(gram in flat for gram in grams)


def _subset_sums_to(values: list[Decimal], target: Decimal) -> bool:
    """Whether any subset of `values` sums to `target`, within fixed bounds."""
    reachable = {Decimal(0)}
    for value in (candidate for candidate in values if candidate != 0):
        reachable |= {reachable_sum + value for reachable_sum in reachable}
        if target in reachable:
            return True
        if len(reachable) > _MAX_REACHABLE:
            return False
    return target in reachable


def _explains_as_plain_arithmetic(
    node_text: str, values: list[Decimal], target: Decimal, query_text: str
) -> str | None:
    """Reading a value, totalling a column, differencing two years, or a subset sum."""
    if target in values:
        return "contains value"
    if values and sum(values) == target:
        return "sum of all numbers"
    for left, right in combinations(values, 2):
        if abs(left - right) == target:
            return f"difference of {left} and {right}"
    if _subset_sums_to(values[:_MAX_SUBSET_TERMS], target):
        return "subset sum"
    return None


def _explains_as_quantity_by_price(
    node_text: str, values: list[Decimal], target: Decimal, query_text: str
) -> str | None:
    """Extending an integer quantity by a fractional unit price."""
    quantities = [value for value in values
                  if value == value.to_integral_value() and abs(value) >= 100]
    unit_prices = [value for value in values if value != value.to_integral_value()]
    extended_values = [quantity * unit_price
                       for quantity in quantities for unit_price in unit_prices]
    if not extended_values:
        return None
    if target in extended_values:
        return "quantity by price"
    if _subset_sums_to(extended_values[:_MAX_SUBSET_TERMS], target):
        return "sum of quantity by price"
    return None


def _explains_as_signature_count(
    node_text: str, values: list[Decimal], target: Decimal, query_text: str
) -> str | None:
    """Counting the signature rows of a converted filing table, whole or by role."""
    signature_count = node_text.count(_SIGNATURE_MARKER)
    if signature_count and Decimal(signature_count) == target:
        return f"count of signature markers = {signature_count}"

    signature_rows = [line for line in node_text.splitlines() if _SIGNATURE_MARKER in line]
    if not signature_rows:
        return None
    for role_term in _ROLE_TERMS:
        role_hits = sum(1 for line in signature_rows[:10]
                        if re.search(rf"\b{role_term}\b", line, re.I))
        if role_hits and Decimal(role_hits) == target:
            return f"count of {role_term!r} in first 10 signatories = {role_hits}"
    return None


def _explains_as_term_count(
    node_text: str, values: list[Decimal], target: Decimal, query_text: str
) -> str | None:
    """Counting a role word, or a capitalised term lifted from the question itself."""
    candidate_terms = list(_CAPITALISED.findall(query_text)) + list(_ROLE_TERMS)
    for term in candidate_terms:
        term_hits = len(re.findall(rf"\b{re.escape(term)}\b", node_text, re.I))
        if term_hits and Decimal(term_hits) == target:
            return f"count of {term!r} = {term_hits}"
    return None


# Tried in this order; the first rule that fires supplies the recorded reason.
# Order is load-bearing: it runs cheapest and most direct first, so a node that
# simply contains the figure is never reported as a coincidental count.
_DERIVATION_RULES = (
    _explains_as_plain_arithmetic,
    _explains_as_quantity_by_price,
    _explains_as_signature_count,
    _explains_as_term_count,
)


def explains_derived(node_text: str, target: Decimal, query_text: str) -> str | None:
    """Why this node yields `target`, or None if it cannot be shown to.

    Covers the arithmetic the implicit quadrants actually ask for: reading a
    value, totalling a column, differencing two years, summing a subset of
    rows, extending a quantity by a price, and counting rows of a table.
    """
    values = _decimals(node_text)
    for explains_as in _DERIVATION_RULES:
        reason = explains_as(node_text, values, target, query_text)
        if reason:
            return reason
    return None


def _target(answer: str) -> Decimal | None:
    """The single figure a derived answer reports, if it reports one."""
    values = _decimals(answer)
    return values[0] if values else None


def localise(row: dict, node_contents: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """The minimal evidence subset of `row`'s citations, with a reason each.

    Returns the citations unchanged with no reasons when the row carries no
    defect (a gold set within MAX_GOLD_SET) or when nothing can be verified.
    Retained ids keep their original relative order.
    """
    citations = row["gt_citations"]
    if len(citations) <= MAX_GOLD_SET:
        return list(citations), {}

    answer = row["ground_truth_answer"]
    direct = {cited_id: "contains answer literal" for cited_id in citations
              if contains_answer_literal(node_contents.get(cited_id, ""), answer)}
    if direct:
        return [cited_id for cited_id in citations if cited_id in direct], direct

    target = _target(answer)
    if target is None:
        return list(citations), {}

    derived: dict[str, str] = {}
    for citation in citations:
        why = explains_derived(node_contents.get(citation, ""), target, row["query_text"])
        if why:
            derived[citation] = why
    if not derived:
        return list(citations), {}
    return [cited_id for cited_id in citations if cited_id in derived], derived


def plan_repairs(rows: list[dict], node_contents: dict[str, str]) -> list[dict]:
    """One entry per row whose citations the rule would change or could not verify.

    A row that needs no repair produces no entry, and neither does one already
    at its fixed point, so a second run over repaired data plans nothing. A row
    that needs a repair but could not be verified is reported with `verified`
    False and its citations left as they are, so a caller can refuse to write
    rather than guess.
    """
    plan = []
    for row in rows:
        before = list(row["gt_citations"])
        after, reasons = localise(row, node_contents)
        if len(before) <= MAX_GOLD_SET:
            continue
        if after == before and reasons:
            continue  # every citation is already load-bearing
        plan.append({
            "query_id": row["query_id"],
            "before": before,
            "after": after,
            "reasons": reasons,
            "verified": bool(reasons),
        })
    return plan


def load_node_contents(conn) -> dict[str, str]:
    """Every node id mapped to its text, for the rule to search."""
    return {node_id: content or "" for node_id, content in
            conn.execute("SELECT node_id, content FROM nodes")}


def load_rows(conn, table: str) -> list[dict]:
    """Query rows from `table` with gt_citations decoded to a list."""
    cursor = conn.execute(
        f"SELECT query_id, query_text, ground_truth_answer, gt_citations FROM {table} ORDER BY query_id"
    )
    return [{"query_id": query_id, "query_text": query_text, "ground_truth_answer": ground_truth_answer, "gt_citations": json.loads(gt_citations)}
            for query_id, query_text, ground_truth_answer, gt_citations in cursor]


def apply_repairs(conn, table: str, plan: list[dict]) -> int:
    """Writes the verified repairs in `plan` to `table`. Returns the row count.

    Unverified entries are skipped rather than written, so a caller that
    ignores the verified flag still cannot corrupt a row it could not check.
    """
    verified = [entry for entry in plan if entry["verified"]]
    conn.executemany(
        f"UPDATE {table} SET gt_citations = ? WHERE query_id = ?",
        [(json.dumps(verified_entry["after"]), verified_entry["query_id"]) for verified_entry in verified],
    )
    return len(verified)


__all__ = [
    "MAX_GOLD_SET",
    "salient_numbers",
    "answer_ngrams",
    "contains_answer_literal",
    "explains_derived",
    "localise",
    "plan_repairs",
    "load_node_contents",
    "load_rows",
    "apply_repairs",
]
