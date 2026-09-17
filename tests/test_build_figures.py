"""build_figures: rendering the results figures from the database.

A chart is hard to assert on pixel by pixel, so these tests cover what can
actually regress silently: that every figure renders and is non-trivial, that
each pipeline keeps one colour drawn from the documented palette, and that a
pipeline missing from the data fails loudly instead of producing a chart with
a series quietly absent.
"""
import re
import sqlite3
from pathlib import Path

import pytest

import build_figures as bf

PALETTE = Path(__file__).resolve().parents[1] / "docs" / "design" / "palette.md"


def _seed(path: Path) -> str:
    """A miniature but structurally complete benchmark database."""
    db_path = path / "mini.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, quadrant TEXT, document_id TEXT);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT, query_id TEXT,
            pipeline TEXT, k_value INTEGER, precision_at_k REAL, recall_at_k REAL,
            evidence_hit INTEGER, citation_match INTEGER, token_f1 REAL, exact_match INTEGER,
            judge_score INTEGER, latency_sec REAL, input_tokens INTEGER, output_tokens INTEGER);
    """)
    quadrants = ["Q1_Direct_Text", "Q2_Implicit_Text", "Q3_Direct_Table", "Q4_Implicit_Table"]
    for quadrant_index, quadrant in enumerate(quadrants):
        for question_index in range(3):
            query_id = f"Q{quadrant_index}_{question_index}"
            conn.execute("INSERT INTO queries VALUES (?,?,?)", (query_id, quadrant, "AAPL_2023"))
            for pipeline_index, pipeline in enumerate(bf.PIPELINE_COLOURS):
                for k_value in (2, 3, 5):
                    score = 5 - pipeline_index - (k_value == 2)
                    conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        f"R_{query_id}_{pipeline}_{k_value}", "PQ", query_id, pipeline, k_value,
                        0.2, 0.5, 1, 1, 0.4, 1, max(1, score), 3.0, 100, 20))
    conn.commit()
    conn.close()
    return str(db_path)


def test_every_figure_renders_and_is_not_blank(tmp_path):
    paths = bf.build_all(_seed(tmp_path), tmp_path / "figs")
    assert len(paths) == 6
    for path in paths:
        assert path.exists(), path
        # A blank axes still produces a file; a real chart is far larger.
        assert path.stat().st_size > 10_000, f"{path.name} looks empty"


def test_figures_are_numbered_for_the_results_chapter(tmp_path):
    paths = bf.build_all(_seed(tmp_path), tmp_path / "figs")
    numbers = sorted(re.search(r"figure-5-(\d)", figure_path.name).group(1) for figure_path in paths)
    assert numbers == ["1", "2", "3", "4", "5", "6"]


def test_rerendering_is_idempotent(tmp_path):
    out = tmp_path / "figs"
    db_path = _seed(tmp_path)
    first = bf.build_all(db_path, out)
    sizes = [figure_path.stat().st_size for figure_path in first]
    second = bf.build_all(db_path, out)
    assert [figure_path.name for figure_path in first] == [figure_path.name for figure_path in second]
    assert all(figure_path.stat().st_size == expected_size for figure_path, expected_size in zip(second, sizes))


# --- the house style ----------------------------------------------------------

def test_every_pipeline_colour_is_taken_from_the_documented_palette():
    """Guards against a colour drifting away from Oxford Ink over time."""
    documented = set(hex_colour.upper() for hex_colour in re.findall(r"#[0-9A-Fa-f]{6}", PALETTE.read_text()))
    for pipeline, colour in bf.PIPELINE_COLOURS.items():
        assert colour.upper() in documented, f"{pipeline} uses {colour}, not in palette.md"


def test_each_pipeline_has_its_own_colour():
    assert len(set(bf.PIPELINE_COLOURS.values())) == len(bf.PIPELINE_COLOURS)


def test_the_accent_and_sequential_ramp_come_from_the_palette():
    documented = set(hex_colour.upper() for hex_colour in re.findall(r"#[0-9A-Fa-f]{6}", PALETTE.read_text()))
    assert bf.GOLD.upper() in documented
    for colour in bf.SEQUENTIAL:
        assert colour.upper() in documented


def test_the_sequential_ramp_covers_all_five_score_bands():
    assert len(bf.SEQUENTIAL) == 5


def test_every_pipeline_in_the_aggregation_has_a_colour_and_a_label():
    for pipeline in bf.agg.PIPELINES:
        assert pipeline in bf.PIPELINE_COLOURS
        assert pipeline in bf.PIPELINE_LABELS


def test_every_quadrant_has_a_display_label():
    for quadrant in bf.agg.QUADRANTS:
        assert quadrant in bf.QUADRANT_LABELS


def test_house_style_pins_calibri_ahead_of_the_matplotlib_default():
    bf.apply_house_style()
    import matplotlib.pyplot as plt
    fonts = plt.rcParams["font.sans-serif"]
    assert fonts[0] == "Calibri"
    assert "DejaVu Sans" in fonts, "a fallback must exist for machines without Calibri"


# --- failure behaviour --------------------------------------------------------

def test_a_missing_pipeline_fails_loudly_rather_than_dropping_a_series(tmp_path):
    db_path = _seed(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM results WHERE pipeline = 'P3_structural'")
    conn.commit()
    conn.close()
    with pytest.raises(KeyError):
        bf.build_all(db_path, tmp_path / "figs")
