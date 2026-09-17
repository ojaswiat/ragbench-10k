"""Where each pipeline looked, measured against where the evidence actually was.

The aggregate metrics say how often a pipeline found the supporting node. They
do not say how it failed, and for the structural pipeline that distinction
carries the whole argument: a retriever that returns valid but consistently
misplaced nodes is behaving as designed and badly suited to the task, whereas
one that returns the same few nodes for every question is simply broken.

Two measurements separate those cases.

Diversity is distinct node ids over total returned slots across the benchmark.
A pipeline that ignores the query converges on the same handful of nodes and
scores low; one that responds to the query scores high. It is measured corpus
wide rather than within a single result row, because no pipeline returns the
same node twice in one row and a within-row figure is therefore always 1.000
and says nothing.

Miss distance is the number of nodes between a returned node and the nearest
node of ground-truth evidence. Because the filings run from under 700 nodes to
over 3,600, it is also reported as a share of the filing it was measured in;
an absolute distance averaged across the corpus is dominated by the longest
documents and is not comparable between them.

Node position comes from the node id. `ingest/node_builder.build_nodes` walks
the parsed filing top to bottom and stamps an incrementing counter into each
id, so the integer in `AAPL_2023_n0412` is that node's position in the
document. This is the only ordering the database carries, since
`nodes.source_page_num` was never populated.
"""
import argparse
import json
import sqlite3
import statistics

PIPELINES = ("P1_vector", "P2_bm25", "P3_structural")


def node_position(node_id: str) -> int:
    """The node's 1-based position in its filing, read from its id."""
    return int(node_id.rsplit("_n", 1)[1])


def document_sizes(conn: sqlite3.Connection) -> dict[str, int]:
    return {row[0]: row[1] for row in conn.execute(
        "SELECT document_id, COUNT(*) FROM nodes GROUP BY document_id")}


def gold_positions(conn: sqlite3.Connection, table: str = "queries") -> dict[str, dict]:
    """Ground-truth evidence positions per question, with its filing."""
    out = {}
    for row in conn.execute(f"SELECT query_id, document_id, gt_citations FROM {table}"):
        query_id, document_id, citations = row
        out[query_id] = {
            "document_id": document_id,
            "positions": [node_position(node_id) for node_id in json.loads(citations or "[]")],
        }
    return out


def diagnose(conn: sqlite3.Connection, pipeline: str, source_set: str = "PQ") -> dict:
    """Diversity and miss distance for one pipeline."""
    sizes = document_sizes(conn)
    gold = gold_positions(conn)

    slots = 0
    distinct: set[str] = set()
    retrieved_share, gold_share = [], []
    miss_nodes, miss_share = [], []
    on_evidence = 0

    cursor = conn.execute(
        "SELECT query_id, retrieved_node_ids FROM results "
        "WHERE source_set = ? AND pipeline = ?", (source_set, pipeline))
    for query_id, retrieved in cursor:
        node_ids = json.loads(retrieved or "[]")
        slots += len(node_ids)
        distinct.update(node_ids)

        truth = gold.get(query_id)
        if not truth or not truth["positions"]:
            continue
        size = sizes[truth["document_id"]]
        gold_share.extend(position / size for position in truth["positions"])
        for node_id in node_ids:
            position = node_position(node_id)
            retrieved_share.append(position / size)
            distance = min(abs(position - gold_position) for gold_position in truth["positions"])
            miss_nodes.append(distance)
            miss_share.append(distance / size)
            if distance == 0:
                on_evidence += 1

    median = lambda values: statistics.median(values) if values else None
    return {
        "pipeline": pipeline,
        "slots": slots,
        "distinct_nodes": len(distinct),
        "diversity": len(distinct) / slots if slots else None,
        "median_retrieved_share": median(retrieved_share),
        "median_gold_share": median(gold_share),
        "median_miss_nodes": median(miss_nodes),
        "median_miss_share": median(miss_share),
        "on_evidence_share": on_evidence / len(miss_nodes) if miss_nodes else None,
    }


def diagnose_all(conn: sqlite3.Connection, source_set: str = "PQ") -> dict[str, dict]:
    return {pipeline: diagnose(conn, pipeline, source_set) for pipeline in PIPELINES}


def print_report(report: dict[str, dict]) -> None:
    print(f"\nRETRIEVAL DIAGNOSTICS\n{'pipeline':<16}{'slots':>7}{'distinct':>10}"
          f"{'diversity':>11}{'position':>10}{'miss':>8}{'miss %':>9}{'on gold':>9}")
    for pipeline, diagnostic in report.items():
        print(f"{pipeline:<16}{diagnostic['slots']:>7}{diagnostic['distinct_nodes']:>10}"
              f"{diagnostic['diversity']:>11.3f}{diagnostic['median_retrieved_share']:>10.3f}"
              f"{diagnostic['median_miss_nodes']:>8.0f}{diagnostic['median_miss_share'] * 100:>8.1f}%"
              f"{diagnostic['on_evidence_share'] * 100:>8.1f}%")
    any_row = next(iter(report.values()))
    print(f"\n  diversity : distinct node ids over returned slots, corpus wide\n"
          f"  position  : median returned node as a share of its filing "
          f"(gold sits at {any_row['median_gold_share']:.3f})\n"
          f"  miss      : median nodes to the nearest ground-truth node, and that\n"
          f"              distance as a share of the filing it was measured in\n"
          f"  on gold   : share of returned slots landing exactly on evidence\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="benchmark.db")
    parser.add_argument("--source-set", default="PQ", choices=["PQ", "JEQ"])
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    report = diagnose_all(conn, args.source_set)
    print_report(report)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
