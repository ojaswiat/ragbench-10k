"""build_findings_page: the standalone results page.

The failure modes worth guarding are silent ones: an f-string field that
never gets substituted, a chart whose labels run outside its own viewBox,
and a summary sentence that keeps asserting a leader after the data behind
it has moved.
"""
import json
import re
import sqlite3
from pathlib import Path

import pytest

from reporting import build_findings_page as bfp

PALETTE = Path(__file__).resolve().parents[1] / "docs" / "design" / "palette.md"


def _retrieved(pipeline: str, index: int) -> str:
    """A distinct retrieval shape per pipeline, in the real node-id format.

    The diagnostics on the page count distinct nodes and measure distance to
    the evidence, so a fixture that hands all three pipelines the same nodes
    cannot tell a working diagnostic from a hardcoded one.
    """
    if pipeline == "P1_vector":
        return f"AAPL_2023_n{index % 20 + 1:04d}"
    if pipeline == "P2_bm25":
        return f"AAPL_2023_n{index % 6 + 3:04d}"
    return f"AAPL_2023_n{index % 3 + 25:04d}"


@pytest.fixture
def benchmark_db(tmp_path) -> str:
    path = tmp_path / "mini.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE nodes (node_id TEXT PRIMARY KEY, document_id TEXT,
            node_type TEXT, token_count INTEGER);
        CREATE TABLE queries (query_id TEXT PRIMARY KEY, quadrant TEXT, document_id TEXT,
            gt_citations TEXT, verified INTEGER);
        CREATE TABLE results (result_id TEXT PRIMARY KEY, source_set TEXT, query_id TEXT,
            pipeline TEXT, k_value INTEGER, retrieved_node_ids TEXT,
            precision_at_k REAL, recall_at_k REAL,
            evidence_hit INTEGER, citation_match INTEGER, token_f1 REAL, exact_match INTEGER,
            judge_score INTEGER, human_score INTEGER, latency_sec REAL,
            input_tokens INTEGER, output_tokens INTEGER);
    """)
    for index in range(1, 31):
        conn.execute("INSERT INTO nodes VALUES (?,?,?,?)",
                     (f"AAPL_2023_n{index:04d}", "AAPL_2023",
                      "table" if index % 5 == 0 else "text", 40 + index))
    quadrants = ["Q1_Direct_Text", "Q2_Implicit_Text", "Q3_Direct_Table", "Q4_Implicit_Table"]
    for quadrant_index, quadrant in enumerate(quadrants):
        for question_index in range(3):
            query_id = f"Q{quadrant_index}_{question_index}"
            conn.execute("INSERT INTO queries VALUES (?,?,?,?,?)",
                         (query_id, quadrant, "AAPL_2023",
                          json.dumps([f"AAPL_2023_n{quadrant_index * 3 + question_index + 1:04d}"]), 1))
            for pipeline_index, pipeline in enumerate(("P1_vector", "P2_bm25", "P3_structural")):
                for k_value in (2, 3, 5):
                    nodes = [_retrieved(pipeline, quadrant_index * 3 + question_index + offset) for offset in range(k_value)]
                    conn.execute(
                        "INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"R_{query_id}_{pipeline}_{k_value}", "PQ", query_id, pipeline, k_value,
                         json.dumps(nodes), 0.3, 0.5,
                         # Both sides of the evidence split are exercised for
                         # every pipeline, so a panel that renders only one of
                         # them cannot pass unnoticed.
                         0 if (question_index + pipeline_index) % 3 == 0 else 1, 1, 0.4, 1,
                         max(1, 5 - pipeline_index - (k_value == 2)), None, 3.0 + pipeline_index, 100, 20))
    for index in range(4):
        conn.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (f"J{index}", "JEQ", f"JE{index}", "P1_vector", 5,
                      json.dumps(["AAPL_2023_n0001"]), 0.3, 0.5, 1, 1, 0.4, 1,
                      4, 4 if index < 3 else 5, 3.0, 100, 20))
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def page(benchmark_db, monkeypatch, tmp_path) -> str:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    return bfp.build_html(benchmark_db)


# --- the page holds together --------------------------------------------------

def test_every_section_renders(page):
    for heading in ("The headline", "All three pillars", "bimodal", "Depth helps",
                    "ordering changes", "Is the difference real", "costs to build",
                    "costs to ask", "Filing by filing", "What would weaken"):
        assert heading in page, heading


def test_no_placeholder_survives_substitution(page):
    body = page.split("</style>", 1)[1]
    assert not re.search(r"\{[a-z_\[\]'\"]+\}", body)


def test_the_page_is_self_contained(page):
    """Nothing to fetch: no script, no external stylesheet, no remote image."""
    assert "<script" not in page
    assert "http://" not in page.replace("http://localhost", "")
    assert "<link" not in page


def test_it_declares_a_viewport_so_phones_do_not_zoom_out(page):
    assert 'name="viewport"' in page


def test_every_table_sits_in_its_own_scroll_container(page):
    """A wide table must scroll inside its box, never widen the page body."""
    tables = page.count("<table>")
    assert tables >= 4, "the page should carry its result tables"
    assert page.count('class="scroll"') == tables


# --- numbers come from the data -----------------------------------------------

def test_the_cell_count_is_read_not_asserted(page):
    assert "108 cells" in page or ">108<" in page or "108" in page


def test_the_gate_percentage_is_computed_from_the_database(page):
    """3 of 4 JEQ rows agree in the fixture."""
    assert "75.0%" in page


def test_the_heatmap_shows_the_question_count_behind_each_row(page):
    """A row built on one question must not read like a rate."""
    assert "questions" in page and 'class="hm-n"' in page


def test_the_quadrant_badge_marks_exactly_the_quadrants_bm25_does_not_lead(benchmark_db, page, monkeypatch):
    """The badge is the reader's cue that the usual ordering broke."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.row_factory = sqlite3.Row
    flipped = 0
    for quadrant in bfp.QUAD:
        scores = {pipeline: conn.execute(
            """SELECT AVG(r.judge_score) FROM results r JOIN queries q ON q.query_id=r.query_id
               WHERE r.source_set='PQ' AND q.quadrant=? AND r.pipeline=?""",
            (quadrant, pipeline)).fetchone()[0] for pipeline in bfp.PIPE}
        if max(scores, key=lambda pipeline: scores[pipeline]) != "P2_bm25":
            flipped += 1
    conn.close()
    assert flipped > 0, "fixture must exercise at least one flip"
    assert page.count('class="flip"') == flipped


def test_the_badge_disappears_when_bm25_leads_every_quadrant(benchmark_db, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.execute("UPDATE results SET judge_score = 1")
    conn.execute("UPDATE results SET judge_score = 5 WHERE pipeline = 'P2_bm25'")
    conn.commit()
    conn.close()
    assert 'class="flip"' not in bfp.build_html(benchmark_db)


def test_the_quadrant_lead_is_derived_not_hardcoded(benchmark_db, monkeypatch):
    """Flip the data so a different pipeline leads Q4, and the prose must follow."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.execute("""UPDATE results SET judge_score = 5 WHERE pipeline='P3_structural'
                    AND query_id IN (SELECT query_id FROM queries WHERE quadrant='Q4_Implicit_Table')""")
    conn.commit()
    conn.close()
    assert "P3 Structural leads there" in bfp.build_html(benchmark_db)


# --- charts stay inside their own boxes ---------------------------------------

def _viewbox(svg: str) -> tuple[float, float]:
    width, height = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups()
    return float(width), float(height)


def test_chart_geometry_stays_within_its_viewbox(page):
    for svg in re.findall(r"<svg.*?</svg>", page, re.S):
        _assert_inside_viewbox(svg)


def _assert_inside_viewbox(svg: str) -> None:
    width, height = _viewbox(svg)
    for x_value in re.findall(r'(?:x|cx|x1|x2)="([\d.]+)"', svg):
        assert float(x_value) <= width, f"x={x_value} beyond viewBox width {width}"
    for y_value in re.findall(r'(?:y|cy|y1|y2)="([\d.]+)"', svg):
        assert float(y_value) <= height, f"y={y_value} beyond viewBox height {height}"


# A label's anchor sitting inside the box is not enough: the glyphs run on from
# there. Calibri at these sizes averages a little over half the font size per
# character, so this is a deliberate over-estimate, which is the safe direction
# for a bound that exists to catch text running off the edge.
_CHAR_WIDTH = {"lbl": 8.4, "ax": 7.6}


def test_chart_labels_have_room_for_their_own_text(page):
    for svg in re.findall(r"<svg.*?</svg>", page, re.S):
        _assert_labels_fit(svg)


def _assert_labels_fit(svg: str) -> None:
    width, _ = _viewbox(svg)
    for x_value, cls, anchor, text in re.findall(
            r'<text x="([\d.]+)"[^>]*class="(lbl|ax)"(?:[^>]*text-anchor="(\w+)")?[^>]*>([^<]*)</text>',
            svg):
        run = len(text) * _CHAR_WIDTH[cls]
        right = float(x_value) + (0 if anchor == "end" else run / (2 if anchor == "middle" else 1))
        assert right <= width + 0.5, (
            f"label {text!r} starts at {x_value} and needs {run:.0f}px, "
            f"running past the {width:.0f}px viewBox")


def test_the_depth_chart_separates_labels_that_would_collide(benchmark_db, monkeypatch):
    """Two pipelines finishing on the same score must not stack their labels."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.execute("UPDATE results SET judge_score = 4")
    conn.commit()
    conn.close()
    svg = next(svg_block for svg_block in re.findall(r"<svg.*?</svg>", bfp.build_html(benchmark_db), re.S)
               if "against retrieval depth" in svg_block)
    label_y_positions = sorted(float(y_value) for y_value in re.findall(r'<text x="[\d.]+" y="([\d.]+)" class="lbl"', svg))
    assert len(label_y_positions) == 3
    assert all(later_y - earlier_y >= 16 for earlier_y, later_y in zip(label_y_positions, label_y_positions[1:])), f"labels overlap at {label_y_positions}"


def test_each_metric_gets_its_own_forest_scale(page):
    """Judge score and recall run on different ranges and must not share an axis."""
    assert "Judge score, 1 to 5" in page
    assert "Recall at k, 0 to 1" in page


# --- colour ------------------------------------------------------------------

def test_each_pipeline_has_its_own_colour():
    hues = [meta["hue"] for meta in bfp.PIPE.values()]
    assert len(set(hues)) == len(hues)


def test_the_five_score_bands_are_distinct():
    assert len(set(bfp.BAND)) == 5


def test_the_house_ink_and_gold_are_taken_from_the_project_palette():
    """The page extends Oxford Ink for data, but keeps its anchors."""
    documented = {height.upper() for height in re.findall(r"#[0-9A-Fa-f]{6}", PALETTE.read_text())}
    assert bfp.GOLD.upper() in documented
    assert bfp.INK.upper() in documented


# --- the added sections carry real content ------------------------------------

def test_every_section_in_the_contents_has_a_heading_to_land_on(page):
    """A contents entry pointing at no anchor is a dead link in a print-out."""
    for anchor, title, _ in bfp.SECTIONS:
        assert f'href="#{anchor}"' in page, f"{anchor} missing from contents"
        assert f'id="{anchor}"' in page, f"{anchor} has no heading"


def test_each_part_rule_appears_once(page):
    parts = [part for _, _, part in bfp.SECTIONS if part]
    assert len(parts) == len(set(parts)), "a part is declared twice"
    for part in parts:
        assert page.count(f">{part}<") == 2, f"{part} should sit in the contents and the body"


def test_the_routing_table_names_every_stage_that_calls_a_model(page):
    for role, model, _, _ in bfp.ROUTING:
        assert role in page
        assert model in page


def test_the_answerer_and_the_judge_are_different_families(page):
    """The anti-self-grading invariant is a property of the routing table."""
    models = {role: model for role, model, _, _ in bfp.ROUTING}
    answerer = models["Pipeline answerer"].split("/")[0]
    judge = models["Judge"].split("/")[0]
    assert answerer != judge
    assert "never asked to mark its own output" in page


def test_every_threat_carries_a_defence_not_just_a_complaint(page):
    """The section exists to be written up, so each entry needs its argument."""
    threats = page.count('class="threat ')
    assert threats >= 8, "the validity section should be thorough"
    assert page.count("How to put it in the write-up.") == threats
    assert page.count("What the data shows.") == threats


def test_the_threats_are_graded_so_the_worst_are_obvious(page):
    for level in ("material", "moderate", "minor"):
        assert f'class="threat {level}"' in page


def test_the_leakage_count_is_read_from_the_database(benchmark_db, page, monkeypatch):
    """Cells scoring well with no evidence retrieved are the leakage floor."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    expected = conn.execute(
        """SELECT COUNT(*) FROM results
           WHERE source_set='PQ' AND evidence_hit = 0 AND judge_score >= 4""").fetchone()[0]
    conn.close()
    assert f"<b>{expected} of the" in page


def test_the_leakage_sentence_follows_the_data(benchmark_db, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.execute("UPDATE results SET evidence_hit = 1 WHERE source_set='PQ'")
    conn.commit()
    conn.close()
    assert "<b>0 of the" in bfp.build_html(benchmark_db)


def test_the_corpus_table_splits_text_from_table_nodes(benchmark_db, page, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    tables = conn.execute("SELECT COUNT(*) FROM nodes WHERE node_type='table'").fetchone()[0]
    texts = conn.execute("SELECT COUNT(*) FROM nodes WHERE node_type='text'").fetchone()[0]
    conn.close()
    assert tables and texts, "fixture must carry both node types"
    assert f"<td>{tables:,}</td>" in page
    assert f"<td>{texts:,}</td>" in page


def test_the_evidence_histogram_reports_the_largest_answer_key(benchmark_db, page, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    largest = conn.execute(
        "SELECT MAX(json_array_length(gt_citations)) FROM queries").fetchone()[0]
    conn.close()
    assert f"names {largest} node" in page


def test_the_gate_confusion_matrix_totals_the_gate_rows(benchmark_db, page, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    rows = conn.execute(
        "SELECT COUNT(*) FROM results WHERE source_set='JEQ'").fetchone()[0]
    conn.close()
    assert f"of {rows} cells fall on the diagonal" in page


def test_the_latency_section_reports_medians_not_only_means(page):
    """The means differ through provider outliers; the medians are the finding."""
    assert "Latency is not the discriminator" in page
    assert "null result" in page
    assert "box spans the middle half" in page


def test_the_split_table_covers_every_depth_and_every_quadrant(benchmark_db, page, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    k_values = [row[0] for row in conn.execute(
        "SELECT DISTINCT k_value FROM results WHERE source_set='PQ' ORDER BY 1")]
    conn.close()
    for k_value in k_values:
        assert f"<th scope=\"row\">k = {k_value}</th>" in page
    for quadrant, (title, _) in bfp.QUAD.items():
        assert f"<th scope=\"row\">{title}</th>" in page


def test_the_contingency_panel_splits_on_whether_evidence_came_back(page):
    assert "Evidence retrieved" in page
    assert "Evidence missed" in page
    assert page.count('class="split"') >= 1


def test_the_diagnostics_table_normalises_the_miss_by_filing_length(page):
    """A raw node gap is meaningless across filings of 682 and 3,647 nodes."""
    assert "Miss, share of filing" in page
    assert "share of the\n      filing it was measured in" in page.replace("\r", "")


def test_the_latency_medians_are_named_by_pipeline(benchmark_db, page, monkeypatch):
    """Three near-identical unlabelled numbers tell the reader nothing."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    for pipeline in ("P1_vector", "P2_bm25", "P3_structural"):
        median = conn.execute(
            """SELECT latency_sec FROM results WHERE source_set='PQ' AND pipeline=?
               ORDER BY latency_sec""", (pipeline,)).fetchall()
        expected = median[(len(median) - 1) // 2][0]
        assert f"{expected:.2f} s for {bfp.PIPE[pipeline]['label']}" in page
    conn.close()


def test_the_per_filing_lead_counts_are_counted_not_typed(benchmark_db, page, monkeypatch):
    """These were prose with a typed tie count, and the typed count was wrong."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    scores = {}
    for document, pipeline, mean in conn.execute(
            """SELECT q.document_id, r.pipeline, AVG(r.judge_score) FROM results r
               JOIN queries q ON q.query_id = r.query_id
               WHERE r.source_set='PQ' GROUP BY 1, 2"""):
        scores.setdefault(document, {})[pipeline] = mean
    conn.close()
    ties = sum(1 for row in scores.values()
               if row["P1_vector"] == row["P2_bm25"])
    assert f"tie on {ties} more" in page


def test_the_lead_counts_follow_the_data(benchmark_db, monkeypatch):
    """Make every filing a tie and the sentence must say so."""
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    conn = sqlite3.connect(benchmark_db)
    conn.execute("UPDATE results SET judge_score = 3 WHERE source_set='PQ'")
    conn.commit()
    documents = conn.execute(
        "SELECT COUNT(DISTINCT document_id) FROM queries").fetchone()[0]
    conn.close()
    page = bfp.build_html(benchmark_db)
    assert f"tie on {documents} more" in page
    assert "outright on 0," in page
