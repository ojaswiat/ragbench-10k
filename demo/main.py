"""Single entry point for the demo.

    cd project && uv run python -m demo.main

Runs one query per difficulty quadrant through all three retrieval paradigms
at K=3 and prints the ranked evidence each one returns, scored against the
ground-truth citations already stored with the query. Everything runs locally:
no API key is read and no free-tier quota is spent.
"""
import argparse
import asyncio

import database_manager as dbm

from demo import report
from demo.compare import aggregate, build_retrievers, load_demo_queries, run_query
from demo.config import DB_PATH, DEMO_DOCUMENT, K, PIPELINES, RULE
from demo.indexes import ensure_indexes


async def main(document_id: str = DEMO_DOCUMENT, k: int = K) -> int:
    report.banner("RAG RETRIEVAL BENCHMARK   ·   LIVE DEMO",
                  "Three paradigms, one filing, one query per quadrant")

    nodes = await dbm.get_nodes_by_document(DB_PATH, document_id)
    queries = await load_demo_queries(document_id=document_id)
    if not queries:
        print(f"  No queries found for {document_id}.")
        return 1

    print(f"  Filing         {document_id}   ·   {len(nodes):,} nodes")
    print(f"  Queries        {len(queries)}   ·   one per difficulty quadrant")
    print(f"  K              {k}")
    print("  API calls      0   ·   retrieval is local for all three pipelines")

    report.print_step("STEP 1   ·   INDEXES")
    await ensure_indexes(document_id)

    report.print_step("STEP 2   ·   RETRIEVERS")
    retrievers = build_retrievers()

    report.print_step(f"STEP 3   ·   RETRIEVAL, K={k}")
    print("  Same query to each pipeline. Only the retrieval strategy differs.")

    outcomes = []
    for index, query in enumerate(queries, start=1):
        outcome = await run_query(retrievers, query, k)
        outcomes.append(outcome)
        report.query_block(index, len(queries), outcome, document_id, PIPELINES)

    report.summary_block(aggregate(outcomes, PIPELINES), PIPELINES, len(outcomes))
    report.how_to_read()
    report.caveat(len(outcomes), document_id)
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", default=DEMO_DOCUMENT,
                        help="filing to demo; needs a P3 summary tree already built")
    parser.add_argument("--k", type=int, default=K, help="retrieval depth")
    args = parser.parse_args()
    try:
        return asyncio.run(main(args.document_id, args.k))
    except FileNotFoundError as exc:
        print()
        print(RULE)
        print("  DEMO COULD NOT RUN")
        print(RULE)
        print(f"  {exc}")
        print()
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
