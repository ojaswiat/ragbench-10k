"""Scores each JEQ gate output 1-5 against its ground truth and writes the
deterministic metrics in the same pass, so a scored row is complete in one
UPDATE.

The prompt carries only the five golden_queries exemplars that share the
target row's quadrant, never all 20. The citation check is deterministic
code, never delegated to the model.
"""
import asyncio
import hashlib
import json
import logging
import re

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from core import database_manager as dbm
import llm_client.config as config
from judge.metrics import citation_audit, exact_match, precision_at_k, recall_at_k, token_f1
from llm_client.llm_factory import LLMFactory
from core.loop_template import LOCAL_TEST_THROTTLE, apply_throttle

logger = logging.getLogger(__name__)

_STAGE = "judge"

# In-run retry for the Judge call. The Judge is pinned to a single provider
# host for determinism, so its dominant failure is a transient stall on that
# host rather than a bad request; the 60-row gate lost 15 rows to exactly
# that. Three attempts with 2s then 4s of backoff costs seconds on a healthy
# run and saves a whole re-run on an unhealthy one.
_DEFAULT_ATTEMPTS = 3
_RETRY_BASE_DELAY_SEC = 2.0

# Scores content only. Citation validity is deliberately absent: Guardrails
# 4b assigns that to citation_audit() in judge/metrics.py, and asking the
# Judge for it as well would both duplicate a deterministic check and let a
# model's guess override it. Each band is stated explicitly because the
# calibration exemplars price them; an unstated criterion would leave the
# Judge inferring a deduction it was never told about. The prose here is the
# compressed form of the expanded scoring guide a reference scorer works
# from; the two must agree.
_RUBRIC = (
    "You are grading how well a candidate answer responds to a question about a "
    "SEC 10-K filing. You are given the question and the ground-truth answer. "
    "Score the candidate answer from 1 to 5. A 5 fully matches the ground-truth "
    "answer and is complete and precise, in a form that directly answers the question "
    "asked. A 4 is correct on the value but falls short on completeness, precision or "
    "framing. A 3 is partially correct: one half of a two-part answer, with the other "
    "half absent. A 2 reaches the right area of the filing but reports the wrong "
    "figure, the wrong ordinal or the wrong scope. A 1 is wrong, unsupported, "
    "fabricated, or a refusal to answer. Judge only correctness against the ground "
    "truth provided; do not use outside knowledge, and do not search. Respond with "
    "ONLY a JSON object, no markdown fences: "
    '{"score": <integer 1-5>, "justification": "<one sentence>"}.'
)


# Band edges for the 0-100 to 1-5 rescale, as (inclusive floor, band).
# Explicit thresholds rather than round(score / 20) because arithmetic
# rounding is banker's rounding in Python and sends the two "half the
# question answered" exemplars at 50 down to band 2, where they read as
# wrong-value answers rather than the partially-correct answers they are.
# These edges also keep every one of the five bands populated by at least
# one exemplar, which round(score / 20) does not.
_BAND_FLOORS = ((90, 5), (75, 4), (50, 3), (25, 2), (0, 1))


def _rescale_to_1_5(human_score_0_100: int) -> int:
    """Rescales a 0-100 human score to the Judge's 1-5 scale, clamped."""
    score = max(0, min(100, human_score_0_100))
    for floor, band in _BAND_FLOORS:
        if score >= floor:
            return band
    return 1


def _format_exemplar(exemplar: dict) -> str:
    return (
        f"Question: {exemplar['query_text']}\n"
        f"Ground-truth answer: {exemplar['ground_truth_answer']}\n"
        f"Candidate answer: {exemplar['example_output']}\n"
        f"Correct score (1-5): {_rescale_to_1_5(exemplar['human_score'])}\n"
        f"Reason: {exemplar['human_reasoning']}"
    )


def build_prefix(quadrant: str, exemplars: list[dict]) -> str:
    """Builds the cacheable system message for one quadrant: rubric plus exemplars.

    Same string for every row of that quadrant, so a prompt-caching provider
    charges the exemplar tokens once per quadrant, not once per row.
    """
    blocks = "\n\n".join(_format_exemplar(exemplar) for exemplar in exemplars)
    return (
        f"{_RUBRIC}\n\n"
        f"Here are graded examples for {quadrant} questions. Score the new "
        f"answer on the same standard.\n\n"
        f"{blocks}"
    )


def _build_target_message(row: dict) -> str:
    return (
        f"Question: {row['query_text']}\n"
        f"Ground-truth answer: {row['ground_truth_answer']}\n"
        f"Candidate answer: {row['pipeline_output']}\n"
        "Score the candidate answer now."
    )


def parse_judge_score(raw_text: str) -> int:
    """Pull a 1-5 integer out of the Judge's reply, clamped into range.

    Prefers the JSON object the rubric asks for; falls back to the first bare
    1-5 integer so a model that adds prose around the number is still usable.
    """
    try:
        obj = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict):
        for key in ("score", "judge_score"):
            if key in obj:
                return max(1, min(5, round(float(obj[key]))))

    match = re.search(r"\b([1-5])\b", raw_text)
    if match:
        return int(match.group(1))
    raise ValueError(f"could not parse a 1-5 judge score from: {raw_text!r}")


def compute_deterministic_metrics(row: dict) -> dict:
    """Computes the code-computed metric columns for one gate row (no LLM).

    evidence_hit is derived from recall: retrieval surfaced at least one
    ground-truth citation. exact_match stays None for the implicit quadrants
    Q2/Q4, stored as NULL.
    """
    retrieved = row["retrieved_node_ids"]
    cited = row["cited_node_ids"]
    gt_citations = row["gt_citations"]
    recall = recall_at_k(retrieved, gt_citations)
    exact_match_score = exact_match(row["pipeline_output"], row["ground_truth_answer"], quadrant=row["quadrant"])
    return {
        "precision_at_k": precision_at_k(retrieved, gt_citations, row["k_value"]),
        "recall_at_k": recall,
        "evidence_hit": 1 if recall > 0 else 0,
        "citation_match": 1 if citation_audit(cited, gt_citations) else 0,
        "token_f1": token_f1(row["pipeline_output"], row["ground_truth_answer"]),
        "exact_match": None if exact_match_score is None else int(exact_match_score),
    }


class Judge:
    """Wraps the Qwen judge client; enforces the fixed model and temperature.

    The routed model and temperature=0 are fixed constraints, so a caller
    asking for anything else is rejected rather than silently overridden; a
    gate scored under the wrong model would be meaningless.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        routed_model = config.MODEL_ROUTING[_STAGE]["model"]
        model = model or routed_model
        if model != routed_model:
            raise ValueError(
                f"Judge model is fixed to {routed_model!r} by "
                f"config.MODEL_ROUTING[{_STAGE!r}]; got {model!r}"
            )
        if temperature != 0.0:
            raise ValueError(f"All benchmark LLM calls run at temperature=0; got {temperature!r}")
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = LLMFactory.get_client_for_stage(_STAGE)
        return self._client

    async def score_row(self, row: dict, system_prompt: str) -> int:
        client = self._get_client()
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=_build_target_message(row)),
        ]
        response = await client.achat(messages)
        return parse_judge_score(response.message.content or "")


def rubric_fingerprint(system_prompt: str, model: str) -> str:
    """A short digest of everything that determines a judge score.

    The system prompt already contains the rubric text, the quadrant's five
    exemplars, and their rescaled band labels, so hashing it with the model
    name covers every input that could change a score. Stored beside the
    score so a later run can tell a score produced by the current rubric from
    one produced by a retired rubric, which is a distinction "has a score"
    cannot make: when the scale moved from 1-10 to 1-5, retired scores in the
    1-5 range remained individually plausible and would have been silently
    kept, mixing two rubrics in one reported figure.
    """
    digest = hashlib.sha256(f"{model}\n{system_prompt}".encode()).hexdigest()
    return digest[:16]


async def _score_with_retry(judge: "Judge", row: dict, prompt: str, attempts: int) -> int:
    """Scores one row, retrying transient failures with exponential backoff.

    The Judge is pinned to one provider host for determinism, so a stalled or
    rate-limited host is the expected failure rather than an exceptional one;
    retrying in-run costs one call where failing out costs a whole re-run.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await judge.score_row(row, prompt)
        except Exception:
            if attempt == attempts:
                raise
            await asyncio.sleep(_RETRY_BASE_DELAY_SEC * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


async def judge_rows(
    db_path: str,
    source_set: str = "JEQ",
    judge: Judge | None = None,
    *,
    resume: bool = True,
    attempts: int = _DEFAULT_ATTEMPTS,
) -> dict:
    """Scores one source set: deterministic metrics plus judge_score, one write each.

    Reads the source set's rows joined to their ground truth, builds each
    quadrant's prompt prefix once, then scores and writes the full metric
    vector per row. Honours LOCAL_TEST_THROTTLE: under throttle only the
    first THROTTLE_LIMIT rows run.

    With `resume` set, a row is skipped only when it already carries a score
    *and* the fingerprint of the rubric that produced it matches the current
    one. A row scored under a retired rubric is re-judged rather than kept,
    which is the whole point: a resume keyed on "has a score" cannot see the
    difference and would leave two rubrics mixed in one set of results.
    """
    if judge is None:
        judge = Judge()

    rows = await dbm.get_judging_rows(db_path, source_set)

    # Build each quadrant's cacheable prefix once, up front, so concurrent
    # scoring never races on the exemplar reads.
    prefixes: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for quadrant in {row["quadrant"] for row in rows}:
        exemplars = await dbm.get_golden_queries_by_quadrant(db_path, quadrant)
        prefixes[quadrant] = build_prefix(quadrant, exemplars)
        fingerprints[quadrant] = rubric_fingerprint(prefixes[quadrant], judge.model)

    total = len(rows)
    if resume:
        rows = [row for row in rows
                if row["judge_score"] is None
                or row.get("judge_fingerprint") != fingerprints[row["quadrant"]]]
    skipped = total - len(rows)
    rows = apply_throttle(rows)

    logger.info(
        "judging %d of %d %s row(s); %d already scored under the current rubric%s",
        len(rows), total, source_set, skipped,
        " (LOCAL_TEST_THROTTLE)" if LOCAL_TEST_THROTTLE else "",
    )

    judged = 0
    failures: list[dict] = []

    async def _score_and_write(row: dict) -> None:
        nonlocal judged
        quadrant = row["quadrant"]
        metrics = compute_deterministic_metrics(row)
        metrics["judge_score"] = await _score_with_retry(judge, row, prefixes[quadrant], attempts)
        # Written in the same UPDATE as the score: a score whose fingerprint
        # failed to land would look stale forever and be re-judged on every run.
        metrics["judge_fingerprint"] = fingerprints[quadrant]
        await dbm.update_result_scores(db_path, row["result_id"], metrics)
        judged += 1

    results = await asyncio.gather(
        *(_score_and_write(row) for row in rows), return_exceptions=True
    )
    for row, outcome in zip(rows, results):
        if isinstance(outcome, Exception):
            logger.exception("judging failed for result_id=%s", row["result_id"], exc_info=outcome)
            # repr, not str: an asyncio.TimeoutError from the client's
            # request timeout carries no message, so str() records a
            # timeout as an empty string.
            failures.append({"result_id": row["result_id"], "error": repr(outcome)})

    return {"judged": judged, "attempted": len(rows), "skipped": skipped,
            "total": total, "failures": failures}


async def judge_jeq_rows(db_path: str, judge: Judge | None = None) -> dict:
    """Scores the JEQ gate rows. Thin wrapper over judge_rows for the gate path."""
    return await judge_rows(db_path, "JEQ", judge)


__all__ = [
    "Judge",
    "build_prefix",
    "compute_deterministic_metrics",
    "judge_jeq_rows",
    "judge_rows",
    "parse_judge_score",
    "rubric_fingerprint",
]
