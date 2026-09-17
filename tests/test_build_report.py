"""build_report: assembling the standalone results page.

The report is a deliverable, and its failure mode is quiet: an f-string
that never gets substituted, a figure that silently drops out because its
filename no longer parses into the key the template asks for, or a table
that renders empty because the data it needs was not loaded. None of those
raise on their own, so they are asserted here.
"""
import json
import sqlite3
import zlib
from pathlib import Path

import pytest

import build_report as br
import index_build_cost as ibc

# The smallest bytes that are a valid PNG: an 8x8 greyscale image. The
# report only base64-embeds these, so the pixels never matter, but a real
# header keeps the fixture honest about what it stands in for.
def _png_bytes() -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (len(payload).to_bytes(4, "big") + body
                + zlib.crc32(body).to_bytes(4, "big"))
    ihdr = (8).to_bytes(4, "big") + (8).to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    raw = b"".join(b"\x00" + b"\xff" * 8 for _ in range(8))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


FIGURE_SLUGS = ("answer-quality", "retrieval-quality", "effect-of-k",
                "by-quadrant", "score-distribution", "paired-comparisons")


@pytest.fixture
def figures(tmp_path) -> Path:
    directory = tmp_path / "figs"
    directory.mkdir()
    for number, slug in enumerate(FIGURE_SLUGS, start=1):
        (directory / f"figure-5-{number}-{slug}.png").write_bytes(_png_bytes())
    return directory


def _retrieved(pipeline: str, index: int) -> str:
    if pipeline == "P1_vector":
        return f"AAPL_2023_n{index + 1:04d}"
    if pipeline == "P2_bm25":
        return f"AAPL_2023_n{index % 4 + 5:04d}"
    return "AAPL_2023_n0020"


@pytest.fixture
def benchmark_db(tmp_path) -> str:
    path = tmp_path / "mini.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE nodes (node_id TEXT PRIMARY KEY, document_id TEXT, content TEXT);
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, quadrant TEXT, document_id TEXT,
                              gt_citations TEXT);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT, query_id TEXT,
            pipeline TEXT, k_value INTEGER, retrieved_node_ids TEXT,
            precision_at_k REAL, recall_at_k REAL,
            evidence_hit INTEGER, citation_match INTEGER, token_f1 REAL, exact_match INTEGER,
            judge_score INTEGER, human_score INTEGER, latency_sec REAL,
            input_tokens INTEGER, output_tokens INTEGER);
    """)
    for index in range(1, 21):
        conn.execute("INSERT INTO nodes VALUES (?,?,?)",
                     (f"AAPL_2023_n{index:04d}", "AAPL_2023", "text"))
    quadrants = ["Q1_Direct_Text", "Q2_Implicit_Text", "Q3_Direct_Table", "Q4_Implicit_Table"]
    for quadrant_index, quadrant in enumerate(quadrants):
        for question_index in range(3):
            query_id = f"Q{quadrant_index}_{question_index}"
            conn.execute("INSERT INTO queries VALUES (?,?,?,?)",
                         (query_id, quadrant, "AAPL_2023",
                          json.dumps(["AAPL_2023_n0008"])))
            for pipeline_index, pipeline in enumerate(("P1_vector", "P2_bm25", "P3_structural")):
                for pipeline_name in (2, 3, 5):
                    conn.execute(
                        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"R_{query_id}_{pipeline}_{pipeline_name}", "PQ", query_id, pipeline, pipeline_name,
                          # Each pipeline gets a different retrieval shape so the
                         # three diversity figures differ, and a test asserting
                         # one cannot be satisfied by another's value:
                         # P1 query-specific, P2 rotating over four, P3 fixed.
                         json.dumps([_retrieved(pipeline, quadrant_index * 3 + question_index)]),
                         0.2, 0.5, 1, 1, 0.4, 1, max(1, 5 - pipeline_index), None, 3.0, 100, 20))
    for index in range(4):
        conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (f"J{index}", "JEQ", f"JE{index}", "P1_vector", 5,
                      json.dumps(["AAPL_2023_n0008"]),
                      0.2, 0.5, 1, 1, 0.4, 1, 4, 4 if index < 3 else 5, 3.0, 100, 20))
    conn.commit()
    conn.close()
    return str(path)


# --- the whole page -----------------------------------------------------------

def test_the_page_assembles_with_every_table_populated(benchmark_db, figures, tmp_path, monkeypatch):
    import json
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(_snapshot(tmp_path, monkeypatch)))
    html = br.build_html(benchmark_db, figures, build_cost_path=path)
    for caption in ("Table 5.1", "Table 5.2", "Table 5.3"):
        assert caption in html
    assert "10.0 h" in html


def test_no_placeholder_survives_substitution(benchmark_db, figures, tmp_path):
    """An unsubstituted field would render as literal braces in the page."""
    body = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / 'absent.json').split("<style>")[1].split("</style>")[1]
    assert "{" not in body and "}" not in body


def test_every_figure_is_embedded_rather_than_linked(benchmark_db, figures, tmp_path):
    html = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / 'absent.json')
    assert html.count("data:image/png;base64,") == len(FIGURE_SLUGS)
    assert 'src="figure' not in html


def test_a_renamed_figure_fails_loudly_instead_of_dropping_out(benchmark_db, figures, tmp_path):
    """The slug is the template's key; a filename change must not be silent."""
    target = figures / "figure-5-3-effect-of-k.png"
    target.rename(figures / "figure-5-3-effect-of-retrieval-depth.png")
    with pytest.raises(KeyError):
        br.build_html(benchmark_db, figures, build_cost_path=tmp_path / 'absent.json')


def test_the_figure_key_is_the_slug_not_the_number(benchmark_db, figures, tmp_path):
    """The key is the slug alone; one field too few leaves the number on it."""
    html = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / 'absent.json')
    assert "Figure 5.1" in html and "Figure 5.6" in html


def test_the_gate_percentage_comes_from_the_database(benchmark_db, figures, tmp_path):
    """3 of 4 JEQ rows agree in the fixture, so the page must say 75.0%."""
    assert "75.0%" in br.build_html(benchmark_db, figures, build_cost_path=tmp_path / 'absent.json')


def test_the_structural_diagnosis_quotes_computed_figures_not_literals(benchmark_db, figures, tmp_path):
    """The paragraph explaining the P3 result once carried hardcoded numbers.

    Asserting the rendered text against a fresh computation is what makes a
    reintroduced literal fail rather than merely go stale.
    """
    import sqlite3 as _sqlite3
    import retrieval_diagnostics as diag
    expected = diag.diagnose_all(_sqlite3.connect(benchmark_db))
    html = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / "absent.json")
    for pipeline in ("P1_vector", "P2_bm25", "P3_structural"):
        assert f"{expected[pipeline]['diversity']:.3f}" in html, pipeline
    p3_stats = expected["P3_structural"]
    assert f"{p3_stats['median_miss_nodes']:.0f} nodes away" in html
    assert f"{p3_stats['median_miss_share'] * 100:.1f}%" in html


def test_the_diagnosis_follows_the_data_when_the_retrieval_changes(benchmark_db, figures, tmp_path):
    """Point every pipeline straight at the evidence and the miss must vanish."""
    import json as _json
    conn = sqlite3.connect(benchmark_db)
    conn.execute("UPDATE results SET retrieved_node_ids = ?",
                 (_json.dumps(["AAPL_2023_n0008"]),))
    conn.commit()
    conn.close()
    html = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / "absent.json")
    assert "0 nodes away" in html
    assert "100.0% of the nodes it returns are evidence nodes" in html


# --- build cost ---------------------------------------------------------------

def _snapshot(tmp_path, monkeypatch, structural_sec=36000.0, local_sec=120.0):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    def attempts(seconds, inp=0, out=0, extra=()):
        rows = [{"document_id": "A", "skipped": False, "wall_clock_sec": seconds,
                 "input_tokens": inp, "output_tokens": out}]
        return list(extra) + rows
    return ibc.build_snapshot({
        "P1_vector": ibc.summarise_attempts(attempts(local_sec)),
        "P2_bm25": ibc.summarise_attempts(attempts(local_sec / 2)),
        "P3_structural": ibc.summarise_attempts(attempts(structural_sec, 500, 400)),
    }, 1000)


def test_the_build_cost_table_marks_the_pipelines_that_spend_no_tokens(tmp_path, monkeypatch):
    rows = br.build_cost_table({"index_build": _snapshot(tmp_path, monkeypatch)})
    assert rows.count("none") == 2
    assert "500 in / 400 out" in rows


def test_the_note_flags_filings_whose_build_tokens_were_never_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    snapshot = ibc.build_snapshot({
        "P1_vector": ibc.summarise_attempts([{"document_id": "A", "skipped": False,
                                              "wall_clock_sec": 60.0, "input_tokens": 0,
                                              "output_tokens": 0}]),
        "P3_structural": ibc.summarise_attempts([
            {"document_id": "JNJ_2023", "skipped": False, "wall_clock_sec": 3300.0,
             "input_tokens": 191464, "output_tokens": 194334},
            {"document_id": "JPM_2023", "skipped": False, "wall_clock_sec": 7500.0,
             "input_tokens": 0, "output_tokens": 0}]),
    }, 1000)
    note = br.build_cost_note({"index_build": snapshot})
    assert "floor" in note and "JPM_2023" in note
    assert "1 of 2 filings" in note


def test_the_build_cost_note_states_the_ratio_against_the_cheapest_local_index(tmp_path, monkeypatch):
    """36000s structural against a 60s BM25 build is a factor of 600."""
    note = br.build_cost_note({"index_build": _snapshot(tmp_path, monkeypatch)})
    assert "600" in note
    assert "900" in note  # 500 + 400 build tokens


def test_later_build_attempts_are_named_in_the_note(tmp_path, monkeypatch):
    monkeypatch.setattr(ibc, "STORAGE_DIRS", {pipeline_name: tmp_path / pipeline_name for pipeline_name in ibc.STORAGE_DIRS})
    retried = [{"document_id": "A", "skipped": False, "wall_clock_sec": seconds,
                "input_tokens": 0, "output_tokens": 0} for seconds in (7200.0, 3600.0)]
    snapshot = ibc.build_snapshot({
        "P1_vector": ibc.summarise_attempts([{"document_id": "A", "skipped": False,
                                              "wall_clock_sec": 60.0, "input_tokens": 0,
                                              "output_tokens": 0}]),
        "P3_structural": ibc.summarise_attempts(retried),
    }, 1000)
    note = br.build_cost_note({"index_build": snapshot})
    assert "rebuilds and resumed runs" in note
    assert "3.0 h" in note  # a 2.0h first build plus a 1.0h later attempt


def test_a_missing_snapshot_is_declared_not_hidden():
    note = br.build_cost_note({"index_build": None})
    assert "has not been taken" in note
    assert br.build_cost_table({"index_build": None}) == ""


def test_the_page_says_so_when_the_build_cost_is_unavailable(benchmark_db, figures, tmp_path):
    html = br.build_html(benchmark_db, figures, build_cost_path=tmp_path / "absent.json")
    assert "has not been taken" in html
    assert "Table 5.3" in html


# --- formatting ---------------------------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0.4, "0.4 s"), (59.9, "59.9 s"), (60.0, "1.0 min"),
    (3599.0, "60.0 min"), (3600.0, "1.0 h"), (37584.0, "10.4 h"),
])
def test_durations_switch_unit_with_magnitude(seconds, expected):
    assert br._duration(seconds) == expected


def test_an_absent_duration_renders_as_a_dash():
    assert br._duration(None) == "-"


def test_sizes_render_in_megabytes():
    assert br._format_bytes(142 * 1024 ** 2) == "142 MB"
    assert br._format_bytes(None) == "-"
