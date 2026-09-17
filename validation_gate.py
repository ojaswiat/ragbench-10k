"""Validation-gate orchestrator: runs the 20 JEQ questions through P1/P2/P3
at a single K=5, reusing loop_executor for retrieval and the shared
Answerer, then runs the Judge over those 60 rows.

It adds no retrieval, answering, or scoring logic of its own; both callees
already carry LOCAL_TEST_THROTTLE, so a throttled dry run caps at
THROTTLE_LIMIT rows through each stage. Human scoring and the
Agreement-Rate gate live in score_gate_outputs.py; this module only
produces the machine side those steps compare against.
"""
import argparse
import asyncio
import logging

import loop_executor
from judge.async_judge import judge_jeq_rows
from pipelines.answerer import Answerer
from pipelines.base import Retriever

# The gate is defined at exactly one K: it validates the Judge against the
# human standard, not the K sweep, so all 60 rows are produced at K=5.
JEQ_K_VALUE = 5

logger = logging.getLogger(__name__)


async def run_validation_gate(
    db_path: str,
    retrievers: dict[str, Retriever] | None = None,
    answerer: Answerer | None = None,
    judge=None,
) -> dict:
    if retrievers is None:
        retrievers = loop_executor.build_retrievers(list(loop_executor.PIPELINES))

    generation = await loop_executor.main(
        db_path=db_path,
        source_set="JEQ",
        retrievers=retrievers,
        answerer=answerer,
        k_values=(JEQ_K_VALUE,),
    )
    judging = await judge_jeq_rows(db_path, judge=judge)

    logger.info(
        "validation gate: generated %s cell(s), judged %s row(s)",
        generation.get("completed"),
        judging.get("judged"),
    )
    return {"generation": generation, "judging": judging}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="benchmark.db")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    summary = asyncio.run(run_validation_gate(db_path=args.db_path))
    print(summary)
