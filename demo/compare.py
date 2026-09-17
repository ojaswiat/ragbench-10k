"""Runs one query through all three retrieval paradigms and scores the overlap.

Only retrieval is exercised. No Answerer, no Judge, no API call: the two
metrics reported here (precision@k, recall@k) are set arithmetic against the
ground-truth citations already stored with the query, so the whole comparison
runs offline and deterministically.
"""
import ast
import asyncio
import json
import time

import database_manager as dbm
from judge.metrics import precision_at_k, recall_at_k
from pipelines.bm25.p2_bm25 import P2BM25Retriever
from pipelines.structural.p3_structural import P3StructuralRetriever
from pipelines.vector.p1_vector import P1VectorRetriever

from demo.config import DB_PATH, DEMO_DOCUMENT, K, QUADRANTS


async def load_demo_queries(
    db_path: str = DB_PATH,
    document_id: str = DEMO_DOCUMENT,
    quadrants: tuple[str, ...] = QUADRANTS,
) -> list[dict]:
    """One query per quadrant, lowest query_id first so the pick is stable."""
    rows = await dbm.get_queries(db_path, "PQ")
    chosen = []
    for quadrant in quadrants:
        matches = sorted(
            (row for row in rows
             if row["document_id"] == document_id and row["quadrant"] == quadrant),
            key=lambda row: row["query_id"],
        )
        if matches:
            chosen.append(matches[0])
    return chosen


def build_retrievers(log=print) -> dict:
    """P1 is constructed last: it loads two ONNX models and is the slow one."""
    log("P2 statistical : loading BM25 corpus")
    p2_bm25 = P2BM25Retriever()
    log("P3 structural  : loading summary tree, MockLLM bound (no tokens spent)")
    p3_structural = P3StructuralRetriever()
    log("P1 semantic    : loading bge-small embedder and bge-reranker cross-encoder")
    p1_vector = P1VectorRetriever()
    return {"P1_vector": p1_vector, "P2_bm25": p2_bm25, "P3_structural": p3_structural}


def _gt_list(query: dict) -> list[str]:
    """gt_citations is a TEXT column holding a list literal.

    Rows written by different generation runs use either JSON (double quotes)
    or a Python repr (single quotes), so both are accepted here rather than
    assuming one and failing on the other.
    """
    raw = query["gt_citations"]
    if not isinstance(raw, str):
        return list(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return list(ast.literal_eval(raw))


async def run_query(retrievers: dict, query: dict, k: int = K) -> dict:
    # Each retrieve() is synchronous and P2's opens its own event loop
    # internally, so calling it directly from this coroutine raises
    # "asyncio.run() cannot be called from a running event loop".
    # to_thread gives every retriever a loop-free thread to run on.
    ground_truth = _gt_list(query)
    outcome = {
        "query_id": query["query_id"],
        "quadrant": query["quadrant"],
        "query_text": query["query_text"],
        "ground_truth": ground_truth,
        "pipelines": {},
    }
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        nodes = await asyncio.to_thread(
            retriever.retrieve, query["query_text"], query["document_id"], k)
        elapsed = time.perf_counter() - start
        node_ids = [scored_node.node.node_id for scored_node in nodes]
        outcome["pipelines"][name] = {
            "node_ids": node_ids,
            "hits": [nid in ground_truth for nid in node_ids],
            "precision": precision_at_k(node_ids, ground_truth, k),
            "recall": recall_at_k(node_ids, ground_truth),
            "latency_sec": elapsed,
        }
    return outcome


def aggregate(outcomes: list[dict], pipelines: tuple[str, ...]) -> dict:
    """Mean of each metric across the demo queries. An illustration only:
    the sample here is a handful of queries on one filing, not the benchmark."""
    summary = {}
    for name in pipelines:
        runs = [outcome["pipelines"][name] for outcome in outcomes if name in outcome["pipelines"]]
        if not runs:
            continue
        run_count = len(runs)
        summary[name] = {
            "precision": sum(run["precision"] for run in runs) / run_count,
            "recall": sum(run["recall"] for run in runs) / run_count,
            "latency_sec": sum(run["latency_sec"] for run in runs) / run_count,
            "n": run_count,
        }
    return summary
