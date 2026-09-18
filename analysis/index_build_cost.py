"""Index build cost: the third leg of the efficiency pillar.

Latency and tokens describe what a query costs once the index exists.
They say nothing about what it cost to create that index, which is a
one-off per corpus but is the dominant difference between these three
pipelines: two build locally with no LLM, one drives an LLM over every
node in the filing.

The two halves of this module reflect how the two kinds of cost become
knowable. P3's cost can only be captured while the build runs, because
its wall clock includes provider queueing that cannot be replayed, so
`build_summary_index` writes an append-only attempt log and this module
reads it back. P1 and P2 are local, deterministic and cheap enough to
re-run, so their cost is measured on demand by rebuilding into a scratch
directory and timing it.

Both halves land in one committed snapshot, because the P3 log lives
under the git-ignored `logs/` tree and would otherwise leave the
efficiency pillar unreproducible from a fresh clone.

Wall clock is reported as measured, including the time P3 spent waiting
on its provider. That wait is a real cost of the design, not measurement
noise, and removing it would report a build that nobody can actually
run.
"""
import argparse
import asyncio
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

COST_LOG_PATH = Path("logs/index_build_costs.json")
SNAPSHOT_PATH = Path("data/index_build_costs.json")
MANIFEST_PATH = Path("data/filings_manifest.json")
DB_PATH = "benchmark.db"

# Where each pipeline's finished index lives, for the on-disk size column.
STORAGE_DIRS = {
    "P1_vector": Path("storage/chroma"),
    "P2_bm25": Path("storage/bm25"),
    "P3_structural": Path("storage/summary_index"),
}

# Only P3 calls an LLM to build. This is the guardrail that separates the
# structural pipeline from the other two, so it is recorded as data rather
# than left implicit in the token totals.
USES_LLM = {"P1_vector": False, "P2_bm25": False, "P3_structural": True}


# --- P3: read back what the build recorded -----------------------------------

def load_attempts(path: Path = COST_LOG_PATH) -> list[dict]:
    """Every build attempt P3 logged, in the order it made them."""
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return json.load(handle)


def _authoritative_attempt_per_filing(built_attempts: list[dict]) -> dict[str, dict]:
    """The attempt that produced the tree on disk, per filing.

    The last attempt that actually called the model wins, because that is the
    build the benchmark queried. Falling back to the first attempt covers a
    filing whose token counts were never recorded at all.
    """
    chosen: dict[str, dict] = {}
    for attempt in built_attempts:
        document_id = attempt["document_id"]
        if attempt.get("input_tokens", 0) > 0 or document_id not in chosen:
            chosen[document_id] = attempt
        if chosen[document_id].get("input_tokens", 0) == 0:
            chosen[document_id] = attempt
    return chosen


def _token_totals(per_filing: list[dict]) -> tuple[int, int, int, list[str]]:
    """Token totals over the filings that recorded them, and the ids of those that did not.

    Two filings were built before the build logged token counts, so their
    tokens are absent rather than zero; naming them lets the total be reported
    as a floor instead of counting a missing measurement as free.
    """
    attempts_with_tokens = [attempt for attempt in per_filing
                            if attempt.get("input_tokens", 0) > 0]
    filings_missing_tokens = sorted(attempt["document_id"] for attempt in per_filing
                                    if attempt.get("input_tokens", 0) == 0)
    return (sum(attempt["input_tokens"] for attempt in attempts_with_tokens),
            sum(attempt["output_tokens"] for attempt in attempts_with_tokens),
            len(attempts_with_tokens),
            filings_missing_tokens)


def summarise_attempts(attempts: list[dict]) -> dict:
    """Per-filing cost from an attempt log, plus what the later attempts cost.

    A filing can appear several times: a genuine rebuild, or a resumed run
    that re-walked already-persisted work. The attempt reported per filing is
    the last one that actually called the model, because that is the build
    that produced the tree on disk and the tree the benchmark queried. Every
    other attempt is reported separately rather than dropped, since the time
    was really spent, but it is not the cost of building the corpus once.

    Taking the plain last attempt instead would be wrong in a way that is easy
    to miss: a resumed run re-walks persisted work without calling the model,
    so it records real wall clock against zero tokens, and summing those would
    report most of the corpus as having cost nothing to summarise.

    Token counts are summed only over the attempts that actually recorded
    them. Two filings were built before the build logged its token counts, so
    their tokens are absent rather than zero; they are named in the result so
    the total can be reported as a floor over the filings that have one,
    instead of quietly counting a missing measurement as free.

    Entries marked `skipped` are cache hits from a later re-run touching an
    already-built filing. They cost nothing and are not attempts.
    """
    built_attempts = [attempt for attempt in attempts if not attempt.get("skipped")]
    chosen = _authoritative_attempt_per_filing(built_attempts)

    per_filing = sorted(chosen.values(), key=lambda attempt: attempt["document_id"])
    filing_seconds = [attempt["wall_clock_sec"] for attempt in per_filing]
    input_tokens, output_tokens, filings_with_tokens, filings_missing_tokens = (
        _token_totals(per_filing))
    later_attempts = [attempt for attempt in built_attempts
                      if attempt is not chosen[attempt["document_id"]]]
    return {
        "filings": len(per_filing),
        "attempts": len(built_attempts),
        "wall_clock_sec": sum(filing_seconds),
        "median_filing_sec": statistics.median(filing_seconds) if filing_seconds else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "filings_with_tokens": filings_with_tokens,
        "filings_missing_tokens": filings_missing_tokens,
        "later_attempt_sec": sum(attempt["wall_clock_sec"] for attempt in later_attempts),
        "per_filing_sec": {attempt["document_id"]: attempt["wall_clock_sec"]
                           for attempt in per_filing},
    }


# --- P1 and P2: measure by rebuilding ----------------------------------------

def _builder(pipeline: str):
    if pipeline == "P1_vector":
        from pipelines.vector import build_vector_index
        return build_vector_index
    if pipeline == "P2_bm25":
        from pipelines.bm25 import build_bm25_index
        return build_bm25_index
    raise ValueError(
        f"{pipeline!r} has no local builder to measure. P3_structural's cost "
        "comes from its attempt log, since its wall clock cannot be replayed."
    )


def document_ids(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    with open(manifest_path) as handle:
        return [entry["document_id"] for entry in json.load(handle)]


async def measure_pipeline(
    pipeline: str,
    scratch_root: Path,
    db_path: str = DB_PATH,
    documents: list[str] | None = None,
) -> dict:
    """Times a full rebuild of one local pipeline into a scratch directory.

    The scratch root is required and is emptied first, so the measurement
    can never read a cached index (which would time a no-op) or disturb the
    live index the benchmark ran against.
    """
    module = _builder(pipeline)
    scratch_root = Path(scratch_root)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True)

    per_filing: dict[str, float] = {}
    for document_id in documents if documents is not None else document_ids():
        started = time.perf_counter()
        result = await module.build_index_for_document(
            document_id, db_path=db_path, storage_root=scratch_root
        )
        elapsed = time.perf_counter() - started
        if result["skipped"]:
            raise RuntimeError(
                f"{pipeline} reported {document_id!r} as already indexed inside a "
                f"scratch root that was emptied before the run. The measurement "
                f"would report a cache hit as a build cost."
            )
        per_filing[document_id] = elapsed

    return summarise_measured(per_filing)


def summarise_measured(per_filing_sec: dict[str, float]) -> dict:
    """The same block shape as `summarise_attempts`, for a local rebuild.

    A local build is one pass with no retries and no model call, so several
    fields are constant by construction rather than measured. Zero tokens is
    a fact about the design, not a missing measurement, which is why no
    filing is listed as having unrecorded tokens.
    """
    seconds = list(per_filing_sec.values())
    return {
        "filings": len(per_filing_sec),
        "attempts": len(per_filing_sec),
        "wall_clock_sec": sum(seconds),
        "median_filing_sec": statistics.median(seconds) if seconds else None,
        "input_tokens": 0,
        "output_tokens": 0,
        "filings_with_tokens": len(per_filing_sec),
        "filings_missing_tokens": [],
        "later_attempt_sec": 0.0,
        "per_filing_sec": dict(per_filing_sec),
    }


# --- the committed snapshot ---------------------------------------------------

def _directory_bytes(path: Path) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict | None:
    """The committed build-cost snapshot, or None when it has not been taken."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def build_snapshot(pipelines: dict[str, dict], node_count: int) -> dict:
    """Assembles the snapshot written to `data/index_build_costs.json`."""
    for name, block in pipelines.items():
        block["uses_llm"] = USES_LLM[name]
        block["storage_bytes"] = _directory_bytes(STORAGE_DIRS[name])
        block["sec_per_1k_nodes"] = (
            block["wall_clock_sec"] / node_count * 1000 if node_count else None
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_nodes": node_count,
        "pipelines": pipelines,
    }


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--cost-log", default=str(COST_LOG_PATH))
    parser.add_argument("--scratch", default="storage/_build_cost_scratch",
                        help="scratch root for the P1/P2 rebuilds, emptied first")
    parser.add_argument("--out", default=str(SNAPSHOT_PATH))
    args = parser.parse_args(argv)

    import sqlite3
    with sqlite3.connect(args.db) as conn:
        node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    pipelines = {}
    for pipeline in ("P1_vector", "P2_bm25"):
        print(f"measuring {pipeline} ...", flush=True)
        pipelines[pipeline] = await measure_pipeline(
            pipeline, Path(args.scratch) / pipeline, db_path=args.db
        )
        print(f"  {pipelines[pipeline]['wall_clock_sec']:.1f}s over "
              f"{pipelines[pipeline]['filings']} filings", flush=True)

    pipelines["P3_structural"] = summarise_attempts(load_attempts(Path(args.cost_log)))

    snapshot = build_snapshot(pipelines, node_count)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(snapshot, handle, indent=2)
    shutil.rmtree(args.scratch, ignore_errors=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
