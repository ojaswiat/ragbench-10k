"""retrieval_diagnostics: where a pipeline looked against where the evidence was.

These numbers exist to distinguish a retriever that is badly suited to the
task from one that is broken, so the tests pin the two properties that
distinction rests on: that diversity is counted across the benchmark rather
than within a single row, where it is always 1.000 and says nothing, and that
miss distance is normalised by the filing it was measured in, since the corpus
mixes documents of 682 and 3,647 nodes.
"""
import json
import sqlite3

import pytest

import retrieval_diagnostics as rd


def _seed(directory, rows, gold=None, sizes=(("A", 10), ("B", 100))):
    """A corpus of two filings of deliberately different lengths."""
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(directory / "mini.db")
    conn.executescript("""
        CREATE TABLE nodes (node_id TEXT PRIMARY KEY, document_id TEXT);
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, document_id TEXT,
                              gt_citations TEXT);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT,
                              query_id TEXT, pipeline TEXT, retrieved_node_ids TEXT);
    """)
    for document_id, count in sizes:
        for index in range(1, count + 1):
            conn.execute("INSERT INTO nodes VALUES (?,?)",
                         (f"{document_id}_n{index:04d}", document_id))
    gold = gold or {"QA": ("A", ["A_n0005"]), "QB": ("B", ["B_n0050"])}
    for query_id, (document_id, citations) in gold.items():
        conn.execute("INSERT INTO queries VALUES (?,?,?)",
                     (query_id, document_id, json.dumps(citations)))
    for index, (query_id, pipeline, retrieved) in enumerate(rows):
        conn.execute("INSERT INTO results VALUES (?,?,?,?,?)",
                     (f"R{index}", "PQ", query_id, pipeline, json.dumps(retrieved)))
    conn.commit()
    return conn


# --- position comes from the node id -----------------------------------------

@pytest.mark.parametrize("node_id,expected", [
    ("AAPL_2023_n0001", 1), ("AAPL_2023_n0412", 412), ("JPM_2023_n3647", 3647),
])
def test_the_node_id_carries_its_document_position(node_id, expected):
    assert rd.node_position(node_id) == expected


def test_a_node_id_without_a_position_is_not_silently_accepted():
    with pytest.raises(ValueError):
        rd.node_position("AAPL_2023_nXXXX")


# --- diversity ----------------------------------------------------------------

def test_diversity_counts_distinct_nodes_across_the_benchmark(tmp_path):
    """Four slots over four different nodes is full diversity."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005", "A_n0007"]),
                            ("QB", "P1_vector", ["B_n0050", "B_n0060"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["slots"] == 4
    assert diagnostic["distinct_nodes"] == 4
    assert diagnostic["diversity"] == 1.0


def test_a_pipeline_returning_the_same_nodes_for_every_question_scores_low(tmp_path):
    """This is the shape that would mean the retriever ignores the query."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0001", "A_n0002"]),
                            ("QA", "P1_vector", ["A_n0001", "A_n0002"]),
                            ("QA", "P1_vector", ["A_n0001", "A_n0002"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["slots"] == 6
    assert diagnostic["distinct_nodes"] == 2
    assert diagnostic["diversity"] == pytest.approx(1 / 3)


def test_diversity_is_not_the_within_row_figure(tmp_path):
    """Within one row it is always 1.000, which would hide the real behaviour."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0001", "A_n0002"]),
                            ("QA", "P1_vector", ["A_n0001", "A_n0002"])])
    assert rd.diagnose(conn, "P1_vector")["diversity"] < 1.0


# --- miss distance ------------------------------------------------------------

def test_miss_distance_is_to_the_nearest_evidence_node(tmp_path):
    """Gold sits at 5; returning 7 is a miss of 2."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0007"])])
    assert rd.diagnose(conn, "P1_vector")["median_miss_nodes"] == 2


def test_landing_on_the_evidence_is_a_miss_of_zero(tmp_path):
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["median_miss_nodes"] == 0
    assert diagnostic["on_evidence_share"] == 1.0


def test_the_nearest_of_several_evidence_nodes_wins(tmp_path):
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0008"])],
                 gold={"QA": ("A", ["A_n0002", "A_n0009"])})
    assert rd.diagnose(conn, "P1_vector")["median_miss_nodes"] == 1


def test_miss_distance_is_also_reported_as_a_share_of_its_own_filing(tmp_path):
    """A hit in the short filing and a 10-node miss in the long one."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005"]),
                            ("QB", "P1_vector", ["B_n0060"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["median_miss_nodes"] == pytest.approx(5)
    assert diagnostic["median_miss_share"] == pytest.approx(0.05)


def test_the_same_absolute_miss_is_worse_in_a_shorter_filing(tmp_path):
    """Guards the normalisation: without it both would read identically."""
    short = _seed(tmp_path / "short", [("QA", "P1_vector", ["A_n0010"])])
    long = _seed(tmp_path / "long", [("QB", "P1_vector", ["B_n0055"])])
    short_filing = rd.diagnose(short, "P1_vector")
    long_filing = rd.diagnose(long, "P1_vector")
    assert short_filing["median_miss_nodes"] == long_filing["median_miss_nodes"] == 5
    assert short_filing["median_miss_share"] > long_filing["median_miss_share"]


# --- position -----------------------------------------------------------------

def test_position_is_reported_relative_so_filings_can_be_compared(tmp_path):
    """Node 5 of 10 and node 50 of 100 sit at the same place in their document."""
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005"]),
                            ("QB", "P1_vector", ["B_n0050"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["median_retrieved_share"] == pytest.approx(0.5)
    assert diagnostic["median_gold_share"] == pytest.approx(0.5)


def test_a_late_biased_retriever_shows_a_higher_relative_position(tmp_path):
    conn = _seed(tmp_path, [("QB", "P1_vector", ["B_n0090"])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["median_retrieved_share"] == pytest.approx(0.9)
    assert diagnostic["median_gold_share"] == pytest.approx(0.5)


# --- degenerate input ---------------------------------------------------------

def test_a_question_with_no_evidence_is_skipped_not_counted_as_a_hit(tmp_path):
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005"])],
                 gold={"QA": ("A", [])})
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["slots"] == 1
    assert diagnostic["median_miss_nodes"] is None
    assert diagnostic["on_evidence_share"] is None


def test_an_empty_retrieval_contributes_no_slots(tmp_path):
    conn = _seed(tmp_path, [("QA", "P1_vector", [])])
    diagnostic = rd.diagnose(conn, "P1_vector")
    assert diagnostic["slots"] == 0
    assert diagnostic["diversity"] is None


def test_a_pipeline_with_no_rows_does_not_raise(tmp_path):
    conn = _seed(tmp_path, [("QA", "P1_vector", ["A_n0005"])])
    assert rd.diagnose(conn, "P3_structural")["slots"] == 0


def test_all_three_pipelines_are_reported(tmp_path):
    conn = _seed(tmp_path, [("QA", pipeline_name, ["A_n0005"]) for pipeline_name in rd.PIPELINES])
    assert set(rd.diagnose_all(conn)) == set(rd.PIPELINES)


def test_the_printed_report_names_every_pipeline(tmp_path, capsys):
    conn = _seed(tmp_path, [("QA", pipeline_name, ["A_n0005", "A_n0007"]) for pipeline_name in rd.PIPELINES])
    rd.print_report(rd.diagnose_all(conn))
    out = capsys.readouterr().out
    for pipeline in rd.PIPELINES:
        assert pipeline in out
