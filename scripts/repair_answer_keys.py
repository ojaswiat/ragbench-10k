"""Repairs the benchmark answer keys, then refreshes anything scored against them.

Two corrections, both to the answer key rather than to any question. First,
gt_citations is reduced to the minimal evidence set wherever the Generator
returned a whole section (see evidence_localisation). Second, a small table
of individually reviewed answer corrections is applied, for rows where the
Generator extracted a cell literally and produced a key that grades the right
answer as wrong.

Neither correction touches question text, so the terrain the three pipelines
are compared on is unchanged; only the ruler they are measured with moves.
Because retrieval metrics already stored in `results` were computed against
the old keys, the run finishes by recomputing them, which is pure arithmetic
over rows already fetched and costs no LLM call. Judge and reference scores
are never touched: neither reads gt_citations.

The run is a dry run unless --apply is passed, and refuses to write at all if
any row needing repair could not be verified.
"""
import argparse
import json
import sqlite3
import sys

from analysis import evidence_localisation as el
from judge.async_judge import compute_deterministic_metrics

# Tables holding answer keys the rule applies to. golden_queries is excluded
# deliberately: those 20 rows are the Judge's calibration exemplars, are never
# run through a pipeline, and so are never scored for retrieval.
REPAIR_TABLES = ("queries", "judge_validation")

# Metric columns derived from gt_citations or the ground-truth answer text.
# judge_score and human_score are not here and are never rewritten.
DERIVED_COLUMNS = ("precision_at_k", "recall_at_k", "evidence_hit",
                   "citation_match", "token_f1", "exact_match")

# Rows whose stored answer was a literal cell extraction that graded
# backwards, rewritten to the reading the filing actually supports. Each
# entry records the citations that jointly support the corrected answer.
ANSWER_KEY_CORRECTIONS = {
    "QT3_PQ_015": {
        "table": "queries",
        "ground_truth_answer": (
            "Owned. The primary manufacturing facilities table marks Gigafactory Shanghai "
            "with a footnote rather than an ownership word; the footnote states that Tesla "
            "owns the building and the land use rights, that the land use rights have an "
            "initial term of 50 years, and that they are treated as operating lease "
            "right-of-use assets."
        ),
        "gt_citations": ["TSLA_2024_n0371", "TSLA_2024_n0372"],
        "reason": (
            "The stored answer was the raw footnote marker '*', which marked a pipeline "
            "that resolved the footnote wrong and one that echoed the cell right."
        ),
    },
}


def plan_all_repairs(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """The per-table citation repair plan, computed but not written."""
    node_contents = el.load_node_contents(conn)
    return {table: el.plan_repairs(el.load_rows(conn, table), node_contents)
            for table in REPAIR_TABLES}


def recompute_metrics(conn: sqlite3.Connection, source_set: str, table: str) -> int:
    """Recomputes the derived metric columns for one source set. Returns rows written."""
    conn.row_factory = sqlite3.Row
    keys = {answer_key_row["query_id"]: answer_key_row for answer_key_row in conn.execute(
        f"SELECT query_id, quadrant, ground_truth_answer, gt_citations FROM {table}")}

    updates = []
    for row in conn.execute("SELECT * FROM results WHERE source_set = ?", (source_set,)):
        key = keys[row["query_id"]]
        metrics = compute_deterministic_metrics({
            "retrieved_node_ids": json.loads(row["retrieved_node_ids"]),
            "cited_node_ids": json.loads(row["cited_node_ids"]),
            "gt_citations": json.loads(key["gt_citations"]),
            "pipeline_output": row["pipeline_output"],
            "ground_truth_answer": key["ground_truth_answer"],
            "quadrant": key["quadrant"],
            "k_value": row["k_value"],
        })
        updates.append([metrics[column] for column in DERIVED_COLUMNS] + [row["result_id"]])

    conn.executemany(
        f"UPDATE results SET {', '.join(column + ' = ?' for column in DERIVED_COLUMNS)} WHERE result_id = ?",
        updates,
    )
    return len(updates)


def _report(plans: dict[str, list[dict]]) -> list[str]:
    """Prints the plan and returns the ids of any row that could not be verified."""
    unverified = []
    for table, entries in plans.items():
        needed = len(entries)
        verified_entries = [plan_entry for plan_entry in entries if plan_entry["verified"]]
        unverified += [plan_entry["query_id"] for plan_entry in entries if not plan_entry["verified"]]
        before = sum(len(plan_entry["before"]) for plan_entry in entries)
        after = sum(len(plan_entry["after"]) for plan_entry in entries)
        print(f"{table:<18} rows needing repair: {needed:>3}   verified: {len(verified_entries):>3}   "
              f"citations {before} -> {after}")
        for entry in entries:
            mark = " " if entry["verified"] else "!"
            reason = next(iter(entry["reasons"].values()), "UNVERIFIED")
            print(f"  {mark} {entry['query_id']:<14}{len(entry['before']):>4} -> "
                  f"{len(entry['after']):<4} {reason}")
    return unverified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="benchmark.db")
    parser.add_argument("--apply", action="store_true", help="write the repairs; otherwise dry run")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    plans = plan_all_repairs(conn)
    unverified = _report(plans)

    for query_id, correction in ANSWER_KEY_CORRECTIONS.items():
        print(f"\nanswer key {query_id}: {correction['reason']}")

    if unverified:
        print(f"\nrefusing to write: {len(unverified)} row(s) could not be verified: {unverified}")
        return 1
    if not args.apply:
        print("\ndry run. re-run with --apply to write.")
        return 0

    with conn:
        written = sum(el.apply_repairs(conn, table, entries) for table, entries in plans.items())
        for query_id, correction in ANSWER_KEY_CORRECTIONS.items():
            cursor = conn.execute(
                f"UPDATE {correction['table']} SET ground_truth_answer = ?, gt_citations = ? "
                "WHERE query_id = ?",
                (correction["ground_truth_answer"], json.dumps(correction["gt_citations"]), query_id),
            )
            # A mistyped query_id would update nothing and report success, so
            # the correction is only credible if it actually landed on a row.
            if cursor.rowcount != 1:
                raise ValueError(
                    f"answer key correction for {query_id!r} matched {cursor.rowcount} rows "
                    f"in {correction['table']!r}, expected exactly 1"
                )
        refreshed = recompute_metrics(conn, "JEQ", "judge_validation")

    print(f"\nwrote {written} citation repair(s), {len(ANSWER_KEY_CORRECTIONS)} answer key(s), "
          f"refreshed {refreshed} result row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
