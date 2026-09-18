"""index_build_cost: the one-off cost of building each pipeline's index.

The risks worth testing here are all about honesty of the number rather
than arithmetic. A cache hit timed as a build would report the structural
pipeline as nearly free; a filing rebuilt three times would look like one
cheap build if only the last attempt were counted and the rest vanished;
and a snapshot that silently fails to load would drop the whole third leg
of the efficiency pillar out of the report without anyone noticing.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from analysis import aggregate_results as agg
from analysis import index_build_cost as ibc


def _attempt(document_id, seconds, inp=10, out=5):
    return {"document_id": document_id, "skipped": False,
            "wall_clock_sec": seconds, "input_tokens": inp, "output_tokens": out}


# --- reading the structural pipeline's attempt log -----------------------------

def test_a_cache_hit_is_not_a_build():
    """`skipped` entries are later runs finding the index already there."""
    summary = ibc.summarise_attempts([
        _attempt("AAPL_2023", 100.0),
        {"document_id": "AAPL_2023", "skipped": True},
        {"document_id": "AAPL_2023", "skipped": True},
    ])
    assert summary["filings"] == 1
    assert summary["attempts"] == 1
    assert summary["wall_clock_sec"] == 100.0


def test_a_rebuilt_filing_reports_the_build_that_produced_the_index():
    """A real rebuild supersedes the first build: it is the tree on disk."""
    summary = ibc.summarise_attempts([
        _attempt("AAPL_2023", 100.0, inp=1, out=1),
        _attempt("AAPL_2023", 300.0, inp=7, out=3),
    ])
    assert summary["filings"] == 1
    assert summary["wall_clock_sec"] == 300.0
    assert (summary["input_tokens"], summary["output_tokens"]) == (7, 3)
    assert summary["later_attempt_sec"] == 100.0


def test_a_resumed_run_never_becomes_the_reported_build():
    """A resume re-walks persisted work: real wall clock, no model call.

    Counting it as the build would report the summarisation as free.
    """
    summary = ibc.summarise_attempts([
        _attempt("AAPL_2023", 1200.0, inp=90_000, out=30_000),
        _attempt("AAPL_2023", 1100.0, inp=0, out=0),
    ])
    assert summary["input_tokens"] == 90_000
    assert summary["wall_clock_sec"] == 1200.0
    assert summary["filings_missing_tokens"] == []


def test_the_later_attempts_are_reported_not_dropped_silently():
    """Time spent rebuilding and resuming was still really spent."""
    summary = ibc.summarise_attempts([
        _attempt("AAPL_2023", 100.0),
        _attempt("AAPL_2023", 300.0),
        _attempt("MSFT_2023", 50.0),
    ])
    assert summary["attempts"] == 3
    assert summary["wall_clock_sec"] == 350.0
    assert summary["later_attempt_sec"] == 100.0


def test_a_filing_whose_tokens_were_never_logged_is_named_not_counted_as_zero():
    """Absent is not free, and a silent zero would understate the build."""
    summary = ibc.summarise_attempts([
        _attempt("JNJ_2023", 3300.0, inp=191_464, out=194_334),
        _attempt("JPM_2023", 7500.0, inp=0, out=0),
    ])
    assert summary["filings"] == 2
    assert summary["filings_with_tokens"] == 1
    assert summary["filings_missing_tokens"] == ["JPM_2023"]
    assert summary["input_tokens"] == 191_464
    # The unmeasured filing still contributes its wall clock, which was logged.
    assert summary["wall_clock_sec"] == 10_800.0


def test_the_median_describes_one_filing_not_the_corpus():
    summary = ibc.summarise_attempts([
        _attempt("A", 10.0), _attempt("B", 20.0), _attempt("C", 300.0),
    ])
    assert summary["median_filing_sec"] == 20.0
    assert summary["wall_clock_sec"] == 330.0


def test_an_empty_log_summarises_to_nothing_rather_than_failing():
    summary = ibc.summarise_attempts([])
    assert summary["filings"] == 0
    assert summary["wall_clock_sec"] == 0
    assert summary["median_filing_sec"] is None
    assert summary["filings_missing_tokens"] == []


def test_a_missing_log_reads_as_no_attempts(tmp_path):
    assert ibc.load_attempts(tmp_path / "absent.json") == []


def test_per_filing_seconds_are_kept_for_the_write_up(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps([_attempt("AAPL_2023", 120.0), _attempt("MSFT_2023", 240.0)]))
    summary = ibc.summarise_attempts(ibc.load_attempts(path))
    assert summary["per_filing_sec"] == {"AAPL_2023": 120.0, "MSFT_2023": 240.0}


# --- measuring the two local pipelines ----------------------------------------

class _FakeBuilder:
    """Stands in for build_vector_index / build_bm25_index."""

    def __init__(self, skipped=False):
        self.skipped = skipped
        self.roots = []

    async def build_index_for_document(self, document_id, db_path, storage_root):
        self.roots.append(Path(storage_root))
        (Path(storage_root) / f"{document_id}.marker").write_text("x")
        return {"document_id": document_id, "skipped": self.skipped, "node_count": 1}


@pytest.mark.asyncio
async def test_a_measured_build_reports_every_filing(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "_builder", lambda name: _FakeBuilder())
    result = await ibc.measure_pipeline(
        "P1_vector", tmp_path / "scratch", db_path="x", documents=["A", "B"])
    assert result["filings"] == 2
    assert result["wall_clock_sec"] >= 0
    assert set(result["per_filing_sec"]) == {"A", "B"}


@pytest.mark.asyncio
async def test_timing_a_cache_hit_fails_loudly(tmp_path, monkeypatch):
    """Otherwise a skipped build would be published as a near-zero cost."""
    monkeypatch.setattr(ibc, "_builder", lambda name: _FakeBuilder(skipped=True))
    with pytest.raises(RuntimeError, match="already indexed"):
        await ibc.measure_pipeline(
            "P1_vector", tmp_path / "scratch", db_path="x", documents=["A"])


@pytest.mark.asyncio
async def test_the_scratch_root_is_emptied_before_measuring(tmp_path, monkeypatch):
    """A leftover index from an earlier measurement would be timed as a hit."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stale = scratch / "stale.marker"
    stale.write_text("from a previous run")
    monkeypatch.setattr(ibc, "_builder", lambda name: _FakeBuilder())
    await ibc.measure_pipeline("P1_vector", scratch, db_path="x", documents=["A"])
    assert not stale.exists()


@pytest.mark.asyncio
async def test_the_live_index_is_never_the_measurement_target(tmp_path, monkeypatch):
    builder = _FakeBuilder()
    monkeypatch.setattr(ibc, "_builder", lambda name: builder)
    scratch = tmp_path / "scratch"
    await ibc.measure_pipeline("P2_bm25", scratch, db_path="x", documents=["A", "B"])
    assert builder.roots == [scratch, scratch]
    assert ibc.STORAGE_DIRS["P2_bm25"] not in builder.roots


def test_the_structural_pipeline_cannot_be_remeasured_by_rebuilding():
    """Its wall clock includes provider queueing that a replay would not see."""
    with pytest.raises(ValueError, match="attempt log"):
        ibc._builder("P3_structural")


def test_an_unknown_pipeline_is_rejected():
    with pytest.raises(ValueError):
        ibc._builder("P4_imaginary")


# --- the committed snapshot ---------------------------------------------------

def test_the_snapshot_records_which_pipeline_needed_an_llm(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot(
        {pipeline: ibc.summarise_attempts([_attempt("A", 10.0)]) for pipeline in ibc.USES_LLM}, 1000)
    flags = {pipeline: build["uses_llm"] for pipeline, build in snapshot["pipelines"].items()}
    assert flags == {"P1_vector": False, "P2_bm25": False, "P3_structural": True}


def test_cost_is_normalised_by_corpus_size(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P1_vector": ibc.summarise_attempts([_attempt("A", 200.0)])}, 2000)
    assert snapshot["pipelines"]["P1_vector"]["sec_per_1k_nodes"] == 100.0


def test_an_empty_corpus_does_not_divide_by_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P1_vector": ibc.summarise_attempts([])}, 0)
    assert snapshot["pipelines"]["P1_vector"]["sec_per_1k_nodes"] is None


def test_storage_size_is_none_when_the_index_is_not_on_this_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / "absent" for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P2_bm25": ibc.summarise_attempts([])}, 10)
    assert snapshot["pipelines"]["P2_bm25"]["storage_bytes"] is None


def test_storage_size_sums_the_index_directory(tmp_path, monkeypatch):
    root = tmp_path / "bm25"
    (root / "nested").mkdir(parents=True)
    (root / "a.pkl").write_bytes(b"x" * 100)
    (root / "nested" / "b.pkl").write_bytes(b"y" * 50)
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: root for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P2_bm25": ibc.summarise_attempts([])}, 10)
    assert snapshot["pipelines"]["P2_bm25"]["storage_bytes"] == 150


def test_a_missing_snapshot_reads_as_none_rather_than_raising(tmp_path):
    assert ibc.load_snapshot(tmp_path / "absent.json") is None


def test_a_written_snapshot_reads_back_identically(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P1_vector": ibc.summarise_attempts([_attempt("A", 5.0)])}, 10)
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snapshot))
    assert ibc.load_snapshot(path) == snapshot


# --- reaching the aggregation -------------------------------------------------

def _seed(path: Path) -> str:
    db_path = path / "mini.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, quadrant TEXT, document_id TEXT);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT, query_id TEXT,
            pipeline TEXT, k_value INTEGER, precision_at_k REAL, recall_at_k REAL,
            evidence_hit INTEGER, citation_match INTEGER, token_f1 REAL, exact_match INTEGER,
            judge_score INTEGER, latency_sec REAL, input_tokens INTEGER, output_tokens INTEGER);
        INSERT INTO queries VALUES ('Q1', 'Q1_Direct_Text', 'AAPL_2023');
        INSERT INTO results VALUES ('R1','PQ','Q1','P1_vector',2,0.5,0.5,1,1,0.5,1,5,1.0,10,5);
    """)
    conn.commit()
    conn.close()
    return str(db_path)


def test_the_aggregation_carries_the_build_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({"P1_vector": ibc.summarise_attempts([_attempt("A", 5.0)])}, 10)
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snapshot))
    conn = sqlite3.connect(_seed(tmp_path))
    report = agg.build_report(conn, "PQ", build_cost_path=path)
    assert report["index_build"]["pipelines"]["P1_vector"]["wall_clock_sec"] == 5.0


def test_the_aggregation_still_builds_without_a_snapshot(tmp_path):
    """Build cost is taken separately, so its absence must not break the run."""
    conn = sqlite3.connect(_seed(tmp_path))
    report = agg.build_report(conn, "PQ", build_cost_path=tmp_path / "absent.json")
    assert report["index_build"] is None
    assert report["cells"] == 1


def test_the_printed_report_says_so_when_no_snapshot_exists(capsys):
    agg.print_index_build(None)
    assert "no snapshot" in capsys.readouterr().out


def test_the_printed_report_marks_the_llm_driven_build(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    blocks = {"P1_vector": ibc.summarise_attempts([_attempt("A", 5.0, inp=0, out=0)]),
              "P3_structural": ibc.summarise_attempts([_attempt("A", 7200.0, inp=99, out=88)])}
    agg.print_index_build(ibc.build_snapshot(blocks, 1000))
    out = capsys.readouterr().out
    assert "2.00h" in out
    assert "99" in out and "88" in out
