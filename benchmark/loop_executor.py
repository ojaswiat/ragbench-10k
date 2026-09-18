"""Runs a single benchmark cell, one (source_set, query_id, pipeline,
k_value) run, and writes one `results` row.

Resumability rests on the `results.UNIQUE(source_set, query_id, pipeline,
k_value)` constraint: a run reads back the keys already committed and skips
them, so an interrupted batch never re-spends quota on a recorded cell.
Retrievers are injected rather than imported here, since constructing all
three eagerly would make a single-pipeline run pay for all of them.
"""
import argparse
import asyncio
import logging

from core import database_manager as dbm
import llm_client.config as config
from core.loop_template import apply_throttle
from pipelines.answerer import Answerer
from pipelines.base import Retriever

K_VALUES: tuple[int, ...] = (2, 3, 5)
PIPELINES: tuple[str, ...] = ("P1_vector", "P2_bm25", "P3_structural")

logger = logging.getLogger(__name__)


def result_id(source_set: str, query_id: str, pipeline: str, k_value: int) -> str:
    """Derived from the natural key rather than a running counter.

    result_id is the table's PRIMARY KEY while
    (source_set, query_id, pipeline, k_value) is its UNIQUE key; making the
    former a function of the latter means the two can never disagree, so a
    re-run of an already-written cell fails loudly on the primary key
    instead of quietly inserting a second row.
    """
    return f"R_{source_set}_{query_id}_{pipeline}_K{k_value}"


def build_cells(
    queries: list[dict],
    pipelines: tuple[str, ...],
    k_values: tuple[int, ...],
    completed: set[tuple[str, str, int]],
) -> list[dict]:
    """Outstanding cells, query-major so all three pipelines answer a given
    query before the run moves on, preserving the same-query-across-pipelines
    ordering the comparison depends on even if the run is cut short."""
    cells = []
    for query in queries:
        for pipeline in pipelines:
            for k_value in k_values:
                if (query["query_id"], pipeline, k_value) in completed:
                    continue
                cells.append({"query": query, "pipeline": pipeline, "k_value": k_value})
    return cells


async def run_cell(
    db_path: str,
    source_set: str,
    cell: dict,
    retriever: Retriever,
    answerer: Answerer,
) -> dict:
    query = cell["query"]
    pipeline = cell["pipeline"]
    k_value = cell["k_value"]

    # Retriever.retrieve() is sync by contract (pipelines/base.py) and P2's
    # implementation drives its own DB read with asyncio.run(), which raises
    # if a loop is already running on this thread. Off-loading to a worker
    # thread satisfies every implementation at once: P2 gets a thread with
    # no running loop, and P1/P3's CPU-bound embedding and rerank work stops
    # blocking this coroutine.
    nodes = await asyncio.to_thread(retriever.retrieve, query["query_text"], query["document_id"], k_value)

    result = await answerer.answer(query["query_text"], nodes)

    row = {
        "result_id": result_id(source_set, query["query_id"], pipeline, k_value),
        "source_set": source_set,
        "query_id": query["query_id"],
        "pipeline": pipeline,
        "k_value": k_value,
        "retrieved_node_ids": [node.node.node_id for node in nodes],
        "pipeline_output": result.raw_text,
        "cited_node_ids": result.cited_node_ids,
        "latency_sec": result.latency_sec,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
    await dbm.upsert_result(db_path, row)
    return row


async def main(
    db_path: str = "benchmark.db",
    source_set: str = "PQ",
    retrievers: dict[str, Retriever] | None = None,
    answerer: Answerer | None = None,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict:
    if not retrievers:
        raise ValueError(
            "loop_executor.main requires at least one retriever, keyed by "
            f"pipeline name (one of {list(PIPELINES)})"
        )
    unknown = set(retrievers) - set(PIPELINES)
    if unknown:
        raise ValueError(f"Unknown pipeline name(s): {sorted(unknown)}")

    if answerer is None:
        answerer = Answerer()

    await dbm.init_db(db_path)
    queries = await dbm.get_queries(db_path, source_set)
    completed = await dbm.get_completed_keys(db_path, source_set)

    pipelines = tuple(pipeline for pipeline in PIPELINES if pipeline in retrievers)
    cells = build_cells(queries, pipelines, k_values, completed)
    total_outstanding = len(cells)

    # The throttle caps cells rather than queries: a cell is the unit that
    # spends quota, so one query under throttle would still fan out to
    # pipelines x K values worth of LLM calls.
    cells = apply_throttle(cells)

    logger.info(
        "source_set=%s: %d cell(s) outstanding, running %d%s",
        source_set,
        total_outstanding,
        len(cells),
        " (LOCAL_TEST_THROTTLE)" if config.LOCAL_TEST_THROTTLE else "",
    )

    completed_count = 0
    failures: list[dict] = []
    for cell in cells:
        try:
            await run_cell(db_path, source_set, cell, retrievers[cell["pipeline"]], answerer)
            completed_count += 1
        except Exception as exc:
            # One cell's failure must not discard the run's prior progress.
            # The failed key stays absent from `results`, so the next run
            # picks it up again through the ordinary resume path.
            logger.exception(
                "cell failed: query_id=%s pipeline=%s k=%d",
                cell["query"]["query_id"], cell["pipeline"], cell["k_value"],
            )
            failures.append({
                "query_id": cell["query"]["query_id"],
                "pipeline": cell["pipeline"],
                "k_value": cell["k_value"],
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })

    return {
        "source_set": source_set,
        "outstanding": total_outstanding,
        "attempted": len(cells),
        "completed": completed_count,
        "failures": failures,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="benchmark.db")
    parser.add_argument("--source-set", default="PQ", choices=["PQ", "JEQ"])
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(PIPELINES),
        choices=list(PIPELINES),
    )
    return parser.parse_args(argv)


def build_retrievers(names: list[str]) -> dict[str, Retriever]:
    """Construct only the requested retrievers, importing each module
    lazily so an unrequested pipeline's dependencies are never loaded."""
    retrievers: dict[str, Retriever] = {}
    for name in names:
        match name:
            case "P1_vector":
                from pipelines.vector.p1_vector import P1VectorRetriever

                retrievers[name] = P1VectorRetriever()
            case "P2_bm25":
                from pipelines.bm25.p2_bm25 import P2BM25Retriever

                retrievers[name] = P2BM25Retriever()
            case "P3_structural":
                from pipelines.structural.p3_structural import P3StructuralRetriever

                retrievers[name] = P3StructuralRetriever()
            case _:
                raise ValueError(f"Unknown pipeline name: {name!r}")
    return retrievers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    summary = asyncio.run(
        main(
            db_path=args.db_path,
            source_set=args.source_set,
            retrievers=build_retrievers(args.pipelines),
        )
    )
    print(summary)
