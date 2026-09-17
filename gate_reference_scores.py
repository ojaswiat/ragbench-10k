"""Reference scores for the validation gate, written without the Judge's.

On this run the agent supplies the reference score in place of a human, so
the figure the gate computes is cross-model concordance rather than an
external human-agreement rate. Independence is the one property it still
carries, and this module enforces two things structurally rather than by
the scorer's discipline:

* Evidence parity -- the rendered fields (quadrant, query_text,
  ground_truth_answer, pipeline_output) are exactly what
  async_judge._build_target_message() sends the Judge, plus quadrant, which
  the Judge effectively also sees through its quadrant-matched calibration
  exemplars. gt_citations and cited_node_ids are withheld because
  Guardrails Sec 4b forbids the Judge from grading citations; showing them
  to the scorer but not the Judge would make any disagreement partly
  measure that evidence gap rather than the Judge's quality.
* Pipeline blindness -- result_id encodes the pipeline
  (f"R_{source_set}_{query_id}_{pipeline}_K{k_value}"), and the pipeline
  column names it directly, so both are dropped from the rendering and
  replaced with an opaque token ("row_0000", ...). A scorer who can tell an
  answer came from P3 -- the pipeline this dissertation predicts will
  perform worst -- could shade the score toward the expected result,
  contaminating both the gate and the headline pipeline comparison the
  project exists to produce. apply_scores() re-derives the same token
  order to translate a token-keyed score mapping back to result_id, so the
  real identifier never has to leave this module.

  A token's *position*, not just its text, can leak the pipeline:
  get_jeq_judging_rows() orders by result_id, and result_id sorts
  alphabetically by pipeline name within each query, so assigning tokens
  in that natural order reproduces a perfect repeating cycle (every third
  row is the same pipeline). Rows are shuffled with a fixed seed before
  tokens are assigned, so the token order carries no pipeline signal while
  still being exactly reproducible between a render call and the apply
  call that follows it.
"""
import asyncio
import logging
import random

import database_manager as dbm

logger = logging.getLogger(__name__)

# Fixed so the shuffled token order is reproducible across the render/apply
# pair that make up one scoring session, without persisting anything to
# disk. Its only job is to decorrelate token position from the natural
# (pipeline-sorted) result_id order -- the value itself carries no other
# meaning and must not change once a scoring session has started.
_SHUFFLE_SEED = 20260910

# Exactly what async_judge._build_target_message() renders for the Judge
# (query_text, ground_truth_answer, pipeline_output), plus quadrant, which
# the Judge receives indirectly via its quadrant-matched calibration
# exemplars. Deliberately excludes judge_score, pipeline, result_id,
# gt_citations and cited_node_ids -- see the module docstring.
_VISIBLE_FIELDS = (
    "quadrant",
    "query_text",
    "ground_truth_answer",
    "pipeline_output",
)


def _token_for(index: int) -> str:
    """An opaque stand-in for result_id, keyed by its position after shuffling.

    Deliberately not a function of result_id, query_id, pipeline or
    k_value: a hash of those would still be a function of the pipeline
    name, and a scorer only needs two tokens to repeat in a pattern to
    start reconstructing groupings.
    """
    return f"row_{index:04d}"


def _shuffled(rows: list[dict]) -> list[dict]:
    """rows, reordered with the fixed seed so pipeline never lines up with
    token position (see module docstring). random.Random(seed).shuffle is
    deterministic for a given input order and seed, so calling this on the
    same underlying row list -- always fetched in the same
    ORDER BY result_id -- reproduces the same permutation every time,
    letting apply_scores() re-derive it without storing anything."""
    shuffled = list(rows)
    random.Random(_SHUFFLE_SEED).shuffle(shuffled)
    return shuffled


async def render_rows_for_scoring(db_path: str) -> list[dict]:
    """The gate rows, shuffled and reduced to the fields a scorer may see.

    Each row carries an opaque "token" in place of result_id; apply_scores()
    resolves it back to the real row. The shuffle means consecutive rendered
    rows are not the same query three times running, and that token
    position carries no pipeline signal.
    """
    rows = _shuffled(await dbm.get_jeq_judging_rows(db_path))
    return [
        {"token": _token_for(index), **{field: row[field] for field in _VISIBLE_FIELDS if field in row}}
        for index, row in enumerate(rows)
    ]


async def _token_map(db_path: str) -> dict[str, str]:
    """Rebuilds the token -> result_id mapping render_rows_for_scoring used.

    Re-running get_jeq_judging_rows and the same fixed-seed shuffle (with no
    rows added or removed in between the two calls) reproduces the same
    token assignment without ever having stored or exposed result_id in the
    rendered rows.
    """
    rows = _shuffled(await dbm.get_jeq_judging_rows(db_path))
    return {_token_for(index): row["result_id"] for index, row in enumerate(rows)}


async def apply_scores(db_path: str, scores: dict[str, int]) -> int:
    """Write one reference score per token; returns the count written.

    scores is keyed by the opaque token render_rows_for_scoring handed out,
    not by result_id -- the caller never has the latter. An unknown token
    fails loudly rather than silently leaving a row unscored and shrinking
    the gate's denominator.
    """
    token_map = await _token_map(db_path)
    for token, score in scores.items():
        try:
            result_id = token_map[token]
        except KeyError as exc:
            raise ValueError(f"no results row for token {token!r}") from exc
        await dbm.update_result_human_score(db_path, result_id, score)
    return len(scores)


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(asyncio.run(render_rows_for_scoring("benchmark.db")), indent=2))
