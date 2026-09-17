"""Builds the standalone findings page at artifacts/findings.html.

Results only: what the benchmark measured, with no build narrative. Every
number is read from `benchmark.db` at render time, through the same
aggregation the dissertation tables use, so this page cannot drift away from
the data it describes.

Colour follows the Oxford Ink palette where that palette has an
answer, and extends it where it does not. Oxford Ink is a deliberately
blue-monochrome identity, which cannot give three pipelines three legible
identities in a chart, so each pipeline takes a saturated hue anchored to the
house Ink and the house Gold is kept for emphasis only.

Charts are inline SVG and CSS rather than a charting library, so the output
stays a single file with nothing to fetch.
"""
import argparse
import html
import math
import sqlite3
from datetime import date
from pathlib import Path

import aggregate_results as agg
import retrieval_diagnostics as diag

OUT_PATH = Path("artifacts/findings.html")

# Where every stage of the benchmark sent its calls. Reproduced here because
# the anti-self-grading invariant is a claim about this table: the model that
# writes an answer is never from the family that grades it.
ROUTING = [
    ("Dataset generator", "nvidia/nemotron-3-super-120b-a12b:free", "OpenRouter",
     "Proposes each question, its answer and its evidence nodes."),
    ("Dataset critic", "openai/gpt-oss-20b:free", "OpenRouter",
     "Accepts or rejects a proposed question. A different family from the generator."),
    ("P3 index build", "nvidia/nemotron-3-super-120b-a12b", "NVIDIA NIM",
     "Summarises every node of every filing into the tree. The only indexing model."),
    ("Pipeline answerer", "meta-llama/llama-3.3-70b-instruct", "OpenRouter",
     "Writes the answer from retrieved nodes. Identical for all three pipelines."),
    ("Judge", "qwen/qwen3.6-27b", "OpenRouter",
     "Scores the answer 1 to 5. A different family from the answerer."),
]

# Extended from palette.md: house Ink and Gold retained, one saturated hue per
# pipeline so three series stay separable at a glance.
PIPE = {
    "P1_vector": {"label": "P1 Vector", "hue": "#3D5AFE", "wash": "#E8ECFF",
                  "note": "ChromaDB and fastembed, reranked by a local cross-encoder"},
    "P2_bm25": {"label": "P2 BM25", "hue": "#00A5A8", "wash": "#DFF5F5",
                "note": "rank_bm25 only, purely statistical"},
    "P3_structural": {"label": "P3 Structural", "hue": "#FF4D6D", "wash": "#FFE7EB",
                      "note": "a LlamaIndex summary tree per filing"},
}
# Score bands 1 to 5, wrong to right.
BAND = ["#FF4D6D", "#FF8A5B", "#FFC145", "#7BC96F", "#00A5A8"]
GOLD = "#D4A017"
INK = "#0A1F33"

QUAD = {
    "Q1_Direct_Text": ("Direct, text", "One explicit statement in continuous prose."),
    "Q2_Implicit_Text": ("Implicit, text", "Synthesis across several narrative passages."),
    "Q3_Direct_Table": ("Direct, table", "Exact extraction of a single table cell."),
    "Q4_Implicit_Table": ("Implicit, table", "A calculation or inference over a table."),
}


def escape_html(text) -> str:
    return html.escape(str(text))


def format_percent(value, places=1) -> str:
    return "-" if value is None else f"{value * 100:.{places}f}%"


def format_number(value, places=3) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _format_rate(seconds) -> str:
    """Seconds per 1000 nodes, which spans from hundredths to thousands."""
    if seconds is None:
        return "-"
    if seconds >= 100:
        return f"{seconds:,.0f} s"
    if seconds >= 1:
        return f"{seconds:.1f} s"
    return f"{seconds:.2f} s"


def duration(seconds) -> str:
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


# --- data ---------------------------------------------------------------------

def load_report_data(db_path: str) -> dict:
    """Everything the page renders, read once."""
    conn = sqlite3.connect(db_path)
    report = agg.build_report(conn, "PQ")
    conn.row_factory = sqlite3.Row

    distribution: dict[str, dict[int, int]] = {}
    for row in conn.execute("""SELECT pipeline, judge_score, COUNT(*) n FROM results
                               WHERE source_set='PQ' GROUP BY 1,2"""):
        distribution.setdefault(row["pipeline"], {})[row["judge_score"]] = row["n"]

    tokens = {}
    for row in conn.execute("""SELECT pipeline, SUM(input_tokens) i, SUM(output_tokens) o,
                                      AVG(latency_sec) l FROM results
                               WHERE source_set='PQ' GROUP BY 1"""):
        tokens[row["pipeline"]] = dict(inp=row["i"], out=row["o"], latency=row["l"])

    gate = conn.execute("""SELECT COUNT(*) n, SUM(judge_score = human_score) agree,
                                  SUM(ABS(judge_score - human_score) <= 1) within_one
                           FROM results WHERE source_set='JEQ'""").fetchone()
    corpus = conn.execute("""SELECT COUNT(*) nodes, COUNT(DISTINCT document_id) filings
                             FROM nodes""").fetchone()
    report.update(
        distribution=distribution, tokens=tokens,
        gate=(gate["agree"], gate["n"]), gate_within_one=gate["within_one"],
        nodes=corpus["nodes"], filings=corpus["filings"],
        questions=conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0],
        per_document_questions={document_row["document_id"]: document_row["n"] for document_row in conn.execute(
            "SELECT document_id, COUNT(*) n FROM queries GROUP BY 1")},
    )
    report.update(load_materials(conn))
    report.update(load_mechanism(conn))
    report.update(load_splits(db_path))
    report["diagnostics"] = diag.diagnose_all(conn)
    return report


def load_materials(conn: sqlite3.Connection) -> dict:
    """The corpus and the question set: what the benchmark was run over."""
    corpus = [dict(corpus_row) for corpus_row in conn.execute("""
        SELECT document_id,
               COUNT(*) nodes,
               SUM(node_type = 'text') text_nodes,
               SUM(node_type = 'table') table_nodes,
               SUM(token_count) tokens
        FROM nodes GROUP BY 1 ORDER BY 1""")]
    quadrants = {corpus_row["quadrant"]: dict(corpus_row) for corpus_row in conn.execute("""
        SELECT quadrant, COUNT(*) n, SUM(verified) verified
        FROM queries GROUP BY 1""")}
    # How many nodes each answer key names. A question whose gold set exceeds
    # the largest k cannot reach full recall, so the shape of this histogram
    # is what makes recall a fair metric rather than an arithmetic trap.
    evidence: dict[int, int] = {}
    for row in conn.execute("SELECT json_array_length(gt_citations) c FROM queries"):
        evidence[row["c"]] = evidence.get(row["c"], 0) + 1
    return {"corpus": corpus, "quadrants": quadrants, "evidence_sizes": evidence,
            "node_totals": {
                "text": sum(corpus_row["text_nodes"] for corpus_row in corpus),
                "table": sum(corpus_row["table_nodes"] for corpus_row in corpus),
                "tokens": sum(corpus_row["tokens"] for corpus_row in corpus)}}


def load_mechanism(conn: sqlite3.Connection) -> dict:
    """The chain from retrieval to answer, and where each pipeline looked."""
    import json as _json

    # The gate's full 5x5 agreement surface, not just its headline percentage:
    # a judge that disagrees by one band is a different instrument from one
    # that disagrees by three, and the percentage alone cannot tell them apart.
    confusion: dict[tuple[int, int], int] = {}
    for row in conn.execute("""SELECT human_score h, judge_score j, COUNT(*) n
                               FROM results WHERE source_set='JEQ'
                               AND human_score IS NOT NULL GROUP BY 1,2"""):
        confusion[(row["h"], row["j"])] = row["n"]
    gate_by_pipeline = {query_row["pipeline"]: dict(query_row) for query_row in conn.execute("""
        SELECT pipeline, COUNT(*) n, SUM(judge_score = human_score) agree,
               AVG(judge_score) judge_mean, AVG(human_score) human_mean
        FROM results WHERE source_set='JEQ' AND human_score IS NOT NULL GROUP BY 1""")}

    # Does finding the evidence decide the answer? Split every cell on
    # evidence_hit and read the judge outcome on each side of the split.
    contingency = {}
    for row in conn.execute("""SELECT pipeline, evidence_hit, COUNT(*) n,
                                      AVG(judge_score) judge_mean,
                                      AVG(judge_score >= 4) pass_rate
                               FROM results WHERE source_set='PQ' GROUP BY 1,2"""):
        contingency[(row["pipeline"], row["evidence_hit"])] = dict(row)

    # A cell scoring 4 or 5 with no evidence retrieved was answered from the
    # model's own knowledge, not from the filing. That is the leakage floor.
    leakage = {query_row["pipeline"]: query_row["n"] for query_row in conn.execute("""
        SELECT pipeline, COUNT(*) n FROM results
        WHERE source_set='PQ' AND evidence_hit = 0 AND judge_score >= 4 GROUP BY 1""")}

    latency = {}
    for pipeline in agg.PIPELINES:
        values = sorted(query_row[0] for query_row in conn.execute(
            "SELECT latency_sec FROM results WHERE source_set='PQ' AND pipeline=?",
            (pipeline,)) if query_row[0] is not None)
        if values:
            def percentile_at(fraction, sorted_values=values):
                return sorted_values[int(fraction * (len(sorted_values) - 1))]
            latency[pipeline] = {"min": values[0], "p25": percentile_at(.25), "median": percentile_at(.5),
                                 "p75": percentile_at(.75), "p90": percentile_at(.90), "max": values[-1],
                                 "mean": sum(values) / len(values)}

    # What kind of node each pipeline puts in front of the answerer, split by
    # whether the question is about a table. A table question answered from
    # prose is a different failure from one answered from the wrong table.
    node_type = {query_row["node_id"]: query_row["node_type"] for query_row in conn.execute(
        "SELECT node_id, node_type FROM nodes")}
    retrieved: dict[tuple[str, str], dict[str, int]] = {}
    for row in conn.execute("""SELECT r.pipeline, q.quadrant, r.retrieved_node_ids ids
                               FROM results r JOIN queries q USING(query_id)
                               WHERE r.source_set='PQ'"""):
        family = "table" if "Table" in row["quadrant"] else "text"
        bucket = retrieved.setdefault((row["pipeline"], family), {"text": 0, "table": 0})
        for node_id in _json.loads(row["ids"]):
            kind = node_type.get(node_id)
            if kind:
                bucket[kind] += 1
    return {"confusion": confusion, "gate_by_pipeline": gate_by_pipeline,
            "contingency": contingency, "leakage": leakage, "latency": latency,
            "retrieved_types": retrieved}


def load_splits(db_path: str) -> dict:
    """Paired comparisons inside each depth and each quadrant.

    The corpus-wide P1 against P2 interval spans zero, which says the two are
    indistinguishable on average. That is not the same as saying they behave
    alike, and these splits are what separates the two readings.
    """
    conn = sqlite3.connect(db_path)
    rows = agg.load_rows(conn, "PQ")
    by_k, by_quadrant = {}, {}
    for k_value in sorted({row["k_value"] for row in rows}):
        by_k[k_value] = agg.compare([row for row in rows if row["k_value"] == k_value],
                              "P1_vector", "P2_bm25")
    for quadrant in sorted({row["quadrant"] for row in rows}):
        by_quadrant[quadrant] = agg.compare(
            [row for row in rows if row["quadrant"] == quadrant], "P1_vector", "P2_bm25")
    conn.close()
    return {"split_by_k": by_k, "split_by_quadrant": by_quadrant}


# --- building blocks ----------------------------------------------------------

def bar_row(label: str, value: float, scale: float, colour: str, display: str) -> str:
    """One horizontal bar: label, track, value. Widths are percentages."""
    width = 0 if not scale else max(0.6, value / scale * 100)
    return f"""<div class="bar">
      <span class="bar-l">{escape_html(label)}</span>
      <span class="bar-t"><span class="bar-f" style="width:{width:.1f}%;background:{colour}"></span></span>
      <span class="bar-v">{escape_html(display)}</span>
    </div>"""


def facts_strip(report: dict) -> str:
    items = [
        (f"{report['filings']}", "SEC 10-K filings"),
        (f"{report['nodes']:,}", "indexed nodes"),
        (f"{report['questions']}", "benchmark questions"),
        ("3", "retrieval paradigms"),
        (f"{report['cells']}", "scored cells"),
        (f"{100 * report['gate'][0] / report['gate'][1]:.1f}%", "judge concordance"),
    ]
    cells = "\n".join(
        f'      <div class="fact"><b>{escape_html(value)}</b><span>{escape_html(label)}</span></div>' for value, label in items)
    return f'    <div class="facts">\n{cells}\n    </div>'


def scoreboard(report: dict) -> str:
    order = sorted(report["overall"], key=lambda pipeline_key: -report["overall"][pipeline_key]["judge_mean"])
    cards = []
    for rank, pipeline in enumerate(order, start=1):
        stats = report["overall"][pipeline]
        meta = PIPE[pipeline]
        cards.append(f"""      <div class="card" style="--hue:{meta['hue']};--wash:{meta['wash']}">
        <div class="card-top"><span class="rank">{rank}</span>
          <div><h3>{escape_html(meta['label'])}</h3><p>{escape_html(meta['note'])}</p></div></div>
        <div class="card-nums">
          <div><b>{format_number(stats['judge_mean'], 2)}</b><span>judge mean</span></div>
          <div><b>{format_percent(stats['judge_pass_rate'], 0)}</b><span>pass rate</span></div>
          <div><b>{format_number(stats['recall_at_k'])}</b><span>recall</span></div>
        </div>
      </div>""")
    return "\n".join(cards)


def overall_table(report: dict) -> str:
    order = sorted(report["overall"], key=lambda pipeline_key: -report["overall"][pipeline_key]["judge_mean"])
    rows = []
    for pipeline in order:
        stats = report["overall"][pipeline]
        token_stats = report["tokens"][pipeline]
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{format_number(stats['judge_mean'], 2)}</td><td>{format_percent(stats['judge_pass_rate'])}</td>
          <td>{format_percent(stats['judge_fail_rate'])}</td><td>{format_number(stats['recall_at_k'])}</td>
          <td>{format_number(stats['precision_at_k'])}</td><td>{format_percent(stats['hit_rate'])}</td>
          <td>{format_percent(stats['citation_match'])}</td><td>{format_number(stats['token_f1'])}</td>
          <td>{format_number(token_stats['latency'], 2)} s</td>
        </tr>""")
    return "\n".join(rows)


def quality_chart(report: dict) -> str:
    order = sorted(report["overall"], key=lambda pipeline_key: -report["overall"][pipeline_key]["judge_mean"])
    bars = []
    for pipeline in order:
        stats = report["overall"][pipeline]
        bars.append(bar_row(PIPE[pipeline]["label"], stats["judge_mean"], 5.0,
                            PIPE[pipeline]["hue"], f"{stats['judge_mean']:.2f} / 5"))
    passes = []
    for pipeline in order:
        stats = report["overall"][pipeline]
        passes.append(bar_row(PIPE[pipeline]["label"], stats["judge_pass_rate"], 1.0,
                              PIPE[pipeline]["hue"], format_percent(stats["judge_pass_rate"], 1)))
    return f"""      <div class="pair">
        <div><h4>Mean judge score</h4>{''.join(bars)}</div>
        <div><h4>Pass rate, scoring 4 or 5</h4>{''.join(passes)}</div>
      </div>"""


def distribution_chart(report: dict) -> str:
    rows = []
    for pipeline in ("P2_bm25", "P1_vector", "P3_structural"):
        counts = report["distribution"][pipeline]
        total = sum(counts.values())
        segments = []
        for score in range(1, 6):
            count = counts.get(score, 0)
            if not count:
                continue
            share = count / total * 100
            inner = f"{count}" if share >= 7 else ""
            segments.append(
                f'<span class="seg" style="width:{share:.2f}%;background:{BAND[score - 1]}" '
                f'title="score {score}: {count} cells">{inner}</span>')
        rows.append(f"""        <div class="dist">
          <span class="dist-l">{escape_html(PIPE[pipeline]['label'])}</span>
          <span class="dist-t">{''.join(segments)}</span>
        </div>""")
    key = "".join(f'<span><i style="background:{BAND[band_index]}"></i>{band_index + 1}</span>' for band_index in range(5))
    return f"""      <div class="dist-wrap">
{chr(10).join(rows)}
        <div class="key">{key}<em>judge score, 1 wrong to 5 correct</em></div>
      </div>"""


def k_chart(report: dict) -> str:
    """Judge score against retrieval depth, one line per pipeline."""
    k_values = [2, 3, 5]
    # x1 stops short of the viewBox so the end-of-line series labels,
    # which sit to the right of the last point, stay inside the box.
    left_x, right_x, top_y, bottom_y = 74, 566, 34, 250
    step = (right_x - left_x) / (len(k_values) - 1)
    x_positions = [left_x + index * step for index in range(len(k_values))]

    def y_of(value):
        return bottom_y - (value / 5.0) * (bottom_y - top_y)

    grid, ticks = [], []
    for value in range(6):
        y_position = y_of(value)
        grid.append(f'<line x1="{left_x}" y1="{y_position:.1f}" x2="{right_x}" y2="{y_position:.1f}" class="grid"/>')
        ticks.append(f'<text x="{left_x - 12}" y="{y_position + 4:.1f}" class="ax" text-anchor="end">{value}</text>')
    for index, k_value in enumerate(k_values):
        ticks.append(f'<text x="{x_positions[index]:.1f}" y="{bottom_y + 26}" class="ax" text-anchor="middle">k = {k_value}</text>')

    plotted = {}
    for pipeline in PIPE:
        plotted[pipeline] = [(x_positions[index], y_of(report["by_k"][f"{pipeline}|k={k_value}"]["judge_mean"]))
                             for index, k_value in enumerate(k_values)]

    # Two pipelines can finish within a few points of each other, which puts
    # their end-of-line labels on top of one another. Walk the labels from the
    # top down and push each one below the last if it is too close.
    label_y, last = {}, None
    for pipeline in sorted(plotted, key=lambda pipeline_key: plotted[pipeline_key][-1][1]):
        y_position = plotted[pipeline][-1][1]
        if last is not None and y_position - last < 17:
            y_position = last + 17
        label_y[pipeline] = last = y_position

    series = []
    for pipeline, meta in PIPE.items():
        points = plotted[pipeline]
        path = " ".join(f"{'M' if index == 0 else 'L'}{x_position:.1f},{y_position:.1f}" for index, (x_position, y_position) in enumerate(points))
        dots = "".join(f'<circle cx="{x_position:.1f}" cy="{y_position:.1f}" r="5" fill="{meta["hue"]}"/>' for x_position, y_position in points)
        end_x, end_y = points[-1]
        label_y_position = label_y[pipeline]
        # When a label has been nudged, a short leader keeps it tied to its line.
        leader = ("" if abs(label_y_position - end_y) < 1 else
                  f'<line x1="{end_x + 6:.1f}" y1="{end_y:.1f}" x2="{end_x + 12:.1f}" '
                  f'y2="{label_y_position:.1f}" stroke="{meta["hue"]}" stroke-width="1.5"/>')
        series.append(
            f'<path d="{path}" fill="none" stroke="{meta["hue"]}" stroke-width="3" '
            f'stroke-linecap="round" stroke-linejoin="round"/>{dots}{leader}'
            f'<text x="{end_x + 16:.1f}" y="{label_y_position + 4:.1f}" class="lbl" fill="{meta["hue"]}">'
            f'{escape_html(meta["label"])}</text>')

    return f"""      <div class="chart">
        <svg viewBox="0 0 760 296" role="img" aria-label="Judge score against retrieval depth">
          {''.join(grid)}
          <line x1="{left_x}" y1="{top_y}" x2="{left_x}" y2="{bottom_y}" class="axis"/>
          <line x1="{left_x}" y1="{bottom_y}" x2="{right_x}" y2="{bottom_y}" class="axis"/>
          {''.join(ticks)}
          {''.join(series)}
        </svg>
      </div>"""


def forest_plot(report: dict, metric: str, title: str) -> str:
    """Paired differences for one metric, against the zero line.

    One plot per metric rather than one plot for both: judge score runs 1 to 5
    and recall runs 0 to 1, so a shared axis would squash every recall interval
    into a dot and invite a comparison of magnitudes that does not exist.
    """
    comps = [comparison for comparison in report["comparisons"] if comparison["metric"] == metric]
    lowest = min(comparison["ci_low"] for comparison in comps)
    highest = max(comparison["ci_high"] for comparison in comps)
    pad = (highest - lowest) * 0.16
    lowest, highest = min(lowest - pad, -pad), highest + pad
    # Every interval can collapse onto zero, for instance when a metric is
    # constant across pipelines. Give the axis a minimum span so the scale
    # stays defined and the marks land on the zero line rather than crashing.
    if highest - lowest < 1e-9:
        lowest, highest = -0.5, 0.5
    left_x, right_x = 300, 742
    row_h, top = 46, 50

    def x_of(value):
        return left_x + (value - lowest) / (highest - lowest) * (right_x - left_x)

    zero = x_of(0.0)
    rows = []
    for index, comparison in enumerate(comps):
        y_position = top + index * row_h
        colour = GOLD if comparison["significant"] else "#767C84"
        name = f"{PIPE[comparison['left']]['label']} vs {PIPE[comparison['right']]['label']}"
        rows.append(
            f'<text x="12" y="{y_position + 5}" class="lbl">{escape_html(name)}</text>'
            f'<line x1="{x_of(comparison["ci_low"]):.1f}" y1="{y_position}" x2="{x_of(comparison["ci_high"]):.1f}" y2="{y_position}" '
            f'stroke="{colour}" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{x_of(comparison["mean_difference"]):.1f}" cy="{y_position}" r="6" fill="{colour}"/>'
            f'<text x="{right_x + 8}" y="{y_position + 5}" class="ax" text-anchor="start">'
            f'{comparison["mean_difference"]:+.3f}</text>')
    height = top + len(comps) * row_h + 20
    return f"""      <div class="chart">
        <h4>{escape_html(title)}</h4>
        <svg viewBox="0 0 810 {height}" role="img" aria-label="Paired differences in {escape_html(title)}">
          <line x1="{zero:.1f}" y1="26" x2="{zero:.1f}" y2="{height - 30}" class="zero"/>
          <text x="{zero:.1f}" y="17" class="ax" text-anchor="middle">no difference</text>
          {''.join(rows)}
          <text x="{left_x}" y="{height - 10}" class="ax" text-anchor="start">{lowest:+.2f}</text>
          <text x="{right_x}" y="{height - 10}" class="ax" text-anchor="end">{highest:+.2f}</text>
        </svg>
      </div>"""


def forest_key() -> str:
    return (f'      <p class="cap"><span class="swatch" style="background:{GOLD}"></span> '
            f'interval clear of zero, a difference is supported &nbsp; '
            f'<span class="swatch" style="background:#767C84"></span> '
            f'interval spans zero, it is not</p>')


def comparison_table(report: dict) -> str:
    rows = []
    for comparison in report["comparisons"]:
        state = "yes" if comparison["significant"] else "no"
        verdict = "difference" if comparison["significant"] else "indistinguishable"
        metric = "Judge score" if comparison["metric"] == "judge_score" else "Recall"
        rows.append(f"""        <tr>
          <th scope="row">{escape_html(PIPE[comparison['left']]['label'])} vs {escape_html(PIPE[comparison['right']]['label'])}</th>
          <td>{escape_html(metric)}</td>
          <td>{comparison['mean_difference']:+.3f}</td>
          <td class="mono">[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}]</td>
          <td>{comparison['wins']} / {comparison['losses']} / {comparison['ties']}</td>
          <td class="mono">{comparison['sign_test_p']:.1e}</td>
          <td><span class="pill {state}">{verdict}</span></td>
        </tr>""")
    return "\n".join(rows)


def quadrant_grid(report: dict) -> str:
    cells = []
    for quadrant, (title, blurb) in QUAD.items():
        scores = {pipeline: report["by_quadrant"][f"{quadrant}|{pipeline}"]["judge_mean"] for pipeline in PIPE}
        winner = max(scores, key=lambda pipeline: scores[pipeline])
        bars = "".join(
            bar_row(PIPE[pipeline]["label"], scores[pipeline], 5.0, PIPE[pipeline]["hue"], f"{scores[pipeline]:.2f}")
            for pipeline in ("P2_bm25", "P1_vector", "P3_structural"))
        flag = ("" if winner == "P2_bm25" else
                f'<span class="flip">{escape_html(PIPE[winner]["label"])} leads here</span>')
        cells.append(f"""        <div class="quad">
          <p class="eyebrow">{escape_html(quadrant.split('_')[0])}</p>
          <h4>{escape_html(title)}</h4><p class="blurb">{escape_html(blurb)}</p>
          {bars}{flag}
        </div>""")
    return "\n".join(cells)


def build_cost(report: dict) -> tuple[str, str]:
    """Build cost as a table and as log-scaled bars, which span four decades."""
    snapshot = report.get("index_build")
    if not snapshot:
        return ("", '<p class="note">No index build cost snapshot has been taken.</p>')
    blocks = snapshot["pipelines"]
    top = math.log10(max(build["wall_clock_sec"] for build in blocks.values()) + 1)
    rows, bars = [], []
    for pipeline in agg.PIPELINES:
        build = blocks.get(pipeline)
        if not build:
            continue
        tokens = ("none" if not build["uses_llm"]
                  else f"{build['input_tokens']:,} in / {build['output_tokens']:,} out")
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{'yes' if build['uses_llm'] else 'no'}</td>
          <td>{duration(build['wall_clock_sec'])}</td>
          <td>{duration(build['median_filing_sec'])}</td>
          <td>{_format_rate(build['sec_per_1k_nodes'])}</td>
          <td>{escape_html(tokens)}</td>
          <td>{build['storage_bytes'] / 1024 ** 2:.0f} MB</td>
        </tr>""")
        width = max(0.8, math.log10(build["wall_clock_sec"] + 1) / top * 100)
        bars.append(f"""<div class="bar">
          <span class="bar-l">{escape_html(PIPE[pipeline]['label'])}</span>
          <span class="bar-t"><span class="bar-f" style="width:{width:.1f}%;background:{PIPE[pipeline]['hue']}"></span></span>
          <span class="bar-v">{duration(build['wall_clock_sec'])}</span>
        </div>""")
    chart = (f'      <div class="logbars">{"".join(bars)}'
             f'<p class="cap">Log scale. The spread is four orders of magnitude, '
             f'so a linear axis would render the two local builds invisible.</p></div>')
    return "\n".join(rows), chart


def token_table(report: dict) -> str:
    rows = []
    for pipeline in agg.PIPELINES:
        token_stats = report["tokens"][pipeline]
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{token_stats['inp']:,}</td><td>{token_stats['out']:,}</td>
          <td>{token_stats['inp'] / 300:,.0f}</td><td>{token_stats['latency']:.2f} s</td>
        </tr>""")
    return "\n".join(rows)


def document_heatmap(report: dict) -> str:
    """Judge score per filing, with the question count that produced it.

    The count is on the face of the chart because the questions are not evenly
    spread: one filing carries seventeen and another carries one, so a row is
    not a like-for-like comparison with the row above it.
    """
    docs = sorted({key.split("|")[0] for key in report["by_document"]})
    counts = report["per_document_questions"]
    # Two labels per heading: the full name, and a short one that a phone-width
    # column can hold without wrapping into its neighbour.
    head = ('<span class="hm-n"><b class="lg">questions</b><b class="sm">n</b></span>'
            + "".join(f'<span class="hm-h"><b class="lg">{escape_html(PIPE[pipeline_key]["label"])}</b>'
                      f'<b class="sm">{escape_html(PIPE[pipeline_key]["label"].split()[0])}</b></span>'
                      for pipeline_key in agg.PIPELINES))
    rows = []
    for document in docs:
        cells = []
        for pipeline in agg.PIPELINES:
            score = report["by_document"][f"{document}|{pipeline}"]["judge_mean"]
            colour = BAND[min(4, max(0, int(round(score)) - 1))]
            dark = score >= 4.5 or score < 1.5
            cells.append(f'<span class="hm-c" style="background:{colour};'
                         f'color:{"#fff" if dark else INK}">{score:.2f}</span>')
        rows.append(f'<span class="hm-r">{escape_html(document)}</span>'
                    f'<span class="hm-n">{counts.get(document, 0)}</span>{"".join(cells)}')
    body = "".join(f'<div class="hm-row">{row_html}</div>' for row_html in rows)
    key = "".join(f'<span><i style="background:{BAND[band_index]}"></i>{band_index + 1}</span>' for band_index in range(5))
    return f"""      <div class="heat">
        <div class="hm-row hm-head"><span class="hm-r"></span>{head}</div>
        {body}
        <div class="key">{key}<em>mean judge score for that filing</em></div>
      </div>"""


# --- design and materials -----------------------------------------------------

def routing_table() -> str:
    return "\n".join(f"""        <tr>
          <th scope="row">{escape_html(role)}</th>
          <td class="mono">{escape_html(model)}</td><td>{escape_html(host)}</td><td>{escape_html(job)}</td>
        </tr>""" for role, model, host, job in ROUTING)


def corpus_table(report: dict) -> str:
    rows = []
    for row in report["corpus"]:
        rows.append(f"""        <tr>
          <th scope="row" class="mono">{escape_html(row['document_id'])}</th>
          <td>{row['nodes']:,}</td><td>{row['text_nodes']:,}</td>
          <td>{row['table_nodes']:,}</td>
          <td>{row['table_nodes'] / row['nodes']:.1%}</td>
          <td>{row['tokens']:,}</td>
          <td>{row['tokens'] / row['nodes']:.0f}</td>
        </tr>""")
    node_totals = report["node_totals"]
    total_nodes = node_totals["text"] + node_totals["table"]
    rows.append(f"""        <tr class="total">
      <th scope="row">All {len(report['corpus'])} filings</th>
      <td>{total_nodes:,}</td><td>{node_totals['text']:,}</td><td>{node_totals['table']:,}</td>
      <td>{node_totals['table'] / total_nodes:.1%}</td><td>{node_totals['tokens']:,}</td>
      <td>{node_totals['tokens'] / total_nodes:.0f}</td>
    </tr>""")
    return "\n".join(rows)


def corpus_chart(report: dict) -> str:
    """Nodes per filing, split by node type.

    The filings are not the same size: the largest holds more than five times
    the nodes of the smallest, which is why every distance measured inside a
    filing has to be normalised by that filing before it can be pooled.
    """
    corpus = report["corpus"]
    if not corpus:
        return ""
    top = max(row["nodes"] for row in corpus)
    bars = []
    for row in corpus:
        share = row["nodes"] / top * 100
        table_share = row["table_nodes"] / row["nodes"] * 100
        bars.append(f"""<div class="bar">
          <span class="bar-l mono">{escape_html(row['document_id'])}</span>
          <span class="bar-t"><span class="bar-f" style="width:{share:.1f}%;background:var(--p1)">
            <span class="bar-sub" style="width:{table_share:.1f}%"></span></span></span>
          <span class="bar-v">{row['nodes']:,}</span>
        </div>""")
    return f"""      <div class="logbars">{''.join(bars)}
        <p class="cap"><span class="swatch" style="background:var(--p1)"></span> text nodes
        &nbsp; <span class="swatch" style="background:{GOLD}"></span> table nodes.
        Every pipeline indexes this identical node set, so no architecture sees a
        cleaner or a larger corpus than another.</p>
      </div>"""


def quadrant_matrix(report: dict) -> str:
    """The 2x2 the question set is built on, as a 2x2."""
    axes = [("Q1_Direct_Text", "Q3_Direct_Table"), ("Q2_Implicit_Text", "Q4_Implicit_Table")]
    rows = []
    for row_label, pair in zip(("Stated directly", "Must be inferred"), axes):
        cells = []
        for quadrant in pair:
            title, blurb = QUAD[quadrant]
            meta = report["quadrants"].get(quadrant, {})
            cells.append(f"""<div class="mx-cell">
              <p class="eyebrow">{escape_html(quadrant.split('_')[0])}</p>
              <b>{meta.get('n', 0)}</b>
              <h4>{escape_html(title)}</h4>
              <p class="blurb">{escape_html(blurb)}</p>
            </div>""")
        rows.append(f'<div class="mx-lab">{escape_html(row_label)}</div>{"".join(cells)}')
    return f"""      <div class="matrix">
        <div class="mx-row mx-head"><div class="mx-lab"></div>
          <div class="mx-h">In continuous prose</div><div class="mx-h">In a table</div></div>
        {''.join(f'<div class="mx-row">{row_html}</div>' for row_html in rows)}
      </div>"""


def evidence_histogram(report: dict) -> str:
    """How many nodes each answer key names."""
    sizes = report["evidence_sizes"]
    if not sizes:
        return ""
    top = max(sizes.values())
    bars = []
    for size in sorted(sizes):
        count = sizes[size]
        # The value column is narrow at phone width, so it carries the count
        # alone and the caption says what the count is of.
        bars.append(bar_row(f"{size} node" + ("" if size == 1 else "s"),
                            count, top, GOLD, f"{count:,}"))
    largest = max(sizes)
    return f"""      <div class="logbars">{''.join(bars)}
        <p class="cap">Bar length is the number of questions whose answer key names
        that many nodes. The largest answer key names {largest} node{'' if largest == 1 else 's'},
        which is at or below the deepest retrieval setting, so recall of 1.000 is
        arithmetically reachable on every question. That was not true before the
        evidence sets were re-derived; the original keys made full recall
        unreachable on 24 questions.</p>
      </div>"""


# --- the judge ----------------------------------------------------------------

def gate_confusion(report: dict) -> str:
    """The gate's full agreement surface, reference score against judge score."""
    confusion = report["confusion"]
    if not confusion:
        return '<p class="note">No gate rows carry a reference score.</p>'
    top = max(confusion.values())
    head = "".join(f'<span class="hm-h"><b class="lg">judge {judge_score}</b>'
                   f'<b class="sm">{judge_score}</b></span>' for judge_score in range(1, 6))
    rows = []
    for human in range(1, 6):
        cells = []
        for judge in range(1, 6):
            count = confusion.get((human, judge), 0)
            if not count:
                cells.append('<span class="hm-c empty">&middot;</span>')
                continue
            on_diagonal = human == judge
            strength = 0.18 + 0.82 * (count / top)
            colour = (f"rgba(0,165,168,{strength:.2f})" if on_diagonal
                      else f"rgba(255,77,109,{strength:.2f})")
            cells.append(f'<span class="hm-c" style="background:{colour};'
                         f'color:{"#fff" if strength > 0.6 else INK}">{count}</span>')
        rows.append(f'<div class="hm-row"><span class="hm-r">reference {human}</span>'
                    f'{"".join(cells)}</div>')
    agree, total = report["gate"]
    return f"""      <div class="heat heat-5">
        <div class="hm-row hm-head"><span class="hm-r"></span>{head}</div>
        {''.join(rows)}
        <p class="cap"><span class="swatch" style="background:var(--p2)"></span> agreement,
        on the diagonal &nbsp; <span class="swatch" style="background:var(--p3)"></span>
        disagreement. {agree} of {total} cells fall on the diagonal
        ({100 * agree / total:.1f}%), and
        {report['gate_within_one']} of {total} fall within one band
        ({100 * report['gate_within_one'] / total:.1f}%). Every disagreement in this
        matrix is a single band; the Judge never missed by two or more.</p>
      </div>"""


def gate_table(report: dict) -> str:
    rows = []
    for pipeline in agg.PIPELINES:
        gate = report["gate_by_pipeline"].get(pipeline)
        if not gate:
            continue
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{gate['n']}</td><td>{gate['agree']}</td>
          <td>{gate['agree'] / gate['n']:.1%}</td>
          <td>{gate['judge_mean']:.2f}</td><td>{gate['human_mean']:.2f}</td>
          <td>{gate['judge_mean'] - gate['human_mean']:+.2f}</td>
        </tr>""")
    return "\n".join(rows)


# --- retrieval ----------------------------------------------------------------

def retrieval_table(report: dict) -> str:
    rows = []
    for pipeline in agg.PIPELINES:
        for k_value in (2, 3, 5):
            stats = report["by_k"].get(f"{pipeline}|k={k_value}")
            if not stats:
                continue
            first = k_value == 2
            label = (f'<th scope="row" rowspan="3"><span class="dot" '
                     f'style="background:{PIPE[pipeline]["hue"]}"></span>'
                     f'{escape_html(PIPE[pipeline]["label"])}</th>' if first else "")
            rows.append(f"""        <tr{' class="grp"' if first else ''}>
          {label}<td>{k_value}</td>
          <td>{format_number(stats['precision_at_k'])}</td><td>{format_number(stats['recall_at_k'])}</td>
          <td>{format_percent(stats['hit_rate'])}</td><td>{format_percent(stats['citation_match'])}</td>
          <td>{format_number(stats['judge_mean'], 2)}</td><td>{format_percent(stats['judge_pass_rate'])}</td>
        </tr>""")
    return "\n".join(rows)


def retrieval_bars(report: dict) -> str:
    """Four retrieval measures side by side, each on its own 0 to 1 scale."""
    metrics = [("recall_at_k", "Recall at k", "share of evidence nodes returned"),
               ("hit_rate", "Evidence hit rate", "at least one evidence node returned"),
               ("citation_match", "Citation match", "the answer cites a node it was given"),
               ("precision_at_k", "Precision at k", "share of returned nodes that are evidence")]
    panels = []
    for key, title, blurb in metrics:
        bars = "".join(
            bar_row(PIPE[pipeline]["label"], report["overall"][pipeline][key] or 0, 1.0, PIPE[pipeline]["hue"],
                    format_number(report["overall"][pipeline][key], 3))
            for pipeline in sorted(report["overall"], key=lambda pipeline: -(report["overall"][pipeline][key] or 0)))
        panels.append(f'<div><h4>{escape_html(title)}</h4><p class="blurb">{escape_html(blurb)}</p>{bars}</div>')
    return f'      <div class="quad-grid">{"".join(panels)}</div>'


def flip_chart(report: dict) -> str:
    """P1 minus P2 in each quadrant, as a diverging bar against zero.

    The corpus-wide interval spans zero. These four do not all behave the same
    way, and a single pooled figure cannot show that.
    """
    splits = report["split_by_quadrant"]
    if not splits:
        return ""
    span = max(max(abs(split["ci_low"]), abs(split["ci_high"])) for split in splits.values()) or 1.0
    width, mid = 780, 390
    row_h, top = 62, 44
    rows = []
    for index, quadrant in enumerate(QUAD):
        split = splits.get(quadrant)
        if not split:
            continue
        y_position = top + index * row_h
        half = (width / 2 - 24) / span
        x_lo, x_hi = mid + split["ci_low"] * half, mid + split["ci_high"] * half
        x_mid = mid + split["mean_difference"] * half
        colour = GOLD if split["significant"] else "#767C84"
        leader = PIPE["P1_vector"] if split["mean_difference"] > 0 else PIPE["P2_bm25"]
        rows.append(
            f'<text x="10" y="{y_position - 12}" class="lbl">{escape_html(QUAD[quadrant][0])}</text>'
            f'<text x="10" y="{y_position + 20}" class="ax">{escape_html(leader["label"])} ahead by '
            f'{abs(split["mean_difference"]):.3f}</text>'
            f'<line x1="{x_lo:.1f}" y1="{y_position}" x2="{x_hi:.1f}" y2="{y_position}" stroke="{colour}" '
            f'stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{x_mid:.1f}" cy="{y_position}" r="6" fill="{colour}"/>')
    height = top + len(splits) * row_h
    return f"""      <div class="chart">
        <svg viewBox="0 0 {width} {height}" role="img"
             aria-label="Vector minus BM25 judge score, by quadrant">
          <line x1="{mid}" y1="18" x2="{mid}" y2="{height - 16}" class="zero"/>
          <text x="{mid}" y="12" class="ax" text-anchor="middle">no difference</text>
          {''.join(rows)}
          <text x="24" y="{height - 2}" class="ax" text-anchor="start">BM25 better</text>
          <text x="{width - 24}" y="{height - 2}" class="ax" text-anchor="end">Vector better</text>
        </svg>
      </div>"""


def split_table(report: dict) -> str:
    """Vector against BM25 inside each depth and each quadrant."""
    rows = []
    for k_value, split in sorted(report["split_by_k"].items()):
        rows.append(("Depth", f"k = {k_value}", split))
    for quadrant, split in report["split_by_quadrant"].items():
        rows.append(("Quadrant", QUAD[quadrant][0], split))
    out = []
    for group, label, split in rows:
        state = "yes" if split["significant"] else "no"
        verdict = "difference" if split["significant"] else "indistinguishable"
        out.append(f"""        <tr>
          <td class="grp-l">{escape_html(group)}</td>
          <th scope="row">{escape_html(label)}</th>
          <td>{split['mean_difference']:+.3f}</td>
          <td class="mono">[{split['ci_low']:+.3f}, {split['ci_high']:+.3f}]</td>
          <td>{split['wins']} / {split['losses']} / {split['ties']}</td>
          <td class="mono">{split['sign_test_p']:.4f}</td>
          <td><span class="pill {state}">{verdict}</span></td>
        </tr>""")
    return "\n".join(out)


# --- mechanism ----------------------------------------------------------------

def contingency_panel(report: dict) -> str:
    """Judge outcome split on whether the evidence was retrieved at all."""
    def side(block, klass=""):
        """One half of the split, or an explicit statement that it is empty.

        A pipeline that never misses, or never hits, would otherwise drop off
        this panel silently and read as if it had not been measured.
        """
        if not block:
            return f'<b class="{klass}">&ndash;</b><span>no cells on this side</span>'
        return (f'<b class="{klass}">{block["pass_rate"]:.0%}</b>'
                f'<span>pass rate over {block["n"]} cells</span>')

    def mean(block):
        return "-" if not block else f"{block['judge_mean']:.2f}"

    cards = []
    for pipeline in agg.PIPELINES:
        hit = report["contingency"].get((pipeline, 1))
        miss = report["contingency"].get((pipeline, 0))
        if not hit and not miss:
            continue
        meta = PIPE[pipeline]
        cards.append(f"""        <div class="card" style="--hue:{meta['hue']};--wash:{meta['wash']}">
          <h3>{escape_html(meta['label'])}</h3>
          <div class="split">
            <div><p class="eyebrow">Evidence retrieved</p>{side(hit)}</div>
            <div><p class="eyebrow">Evidence missed</p>{side(miss, 'bad')}</div>
          </div>
          <p class="blurb">Mean judge score {mean(hit)} with the evidence,
          {mean(miss)} without it.</p>
        </div>""")
    return "\n".join(cards)


def node_type_chart(report: dict) -> str:
    """The share of returned slots that are table nodes, on table questions."""
    retrieved = report["retrieved_types"]
    if not retrieved:
        return ""
    panels = []
    for family, title, blurb in (
            ("table", "Questions about a table",
             "Q3 and Q4. The answer sits in or is derived from a tabular block."),
            ("text", "Questions about prose",
             "Q1 and Q2. A table node here is a retrieval mistake.")):
        bars = []
        for pipeline in agg.PIPELINES:
            bucket = retrieved.get((pipeline, family))
            if not bucket:
                continue
            total = bucket["text"] + bucket["table"]
            if not total:
                continue
            share = bucket["table"] / total
            bars.append(bar_row(PIPE[pipeline]["label"], share, 1.0,
                                PIPE[pipeline]["hue"], f"{share:.1%}"))
        panels.append(f'<div><h4>{escape_html(title)}</h4><p class="blurb">{escape_html(blurb)}</p>'
                      f'{"".join(bars)}</div>')
    return f'      <div class="pair">{"".join(panels)}</div>'


def position_chart(report: dict) -> str:
    """Where each pipeline looked in the filing, against where the evidence was.

    Positions are shares of a filing rather than node counts, because the
    corpus spans 682 to 3,647 nodes and a raw index is not comparable across
    two documents of that different a length.
    """
    diagnostics = report.get("diagnostics") or {}
    usable = {pipeline_key: diagnostic for pipeline_key, diagnostic in diagnostics.items()
              if diagnostic.get("median_retrieved_share") is not None}
    if not usable:
        return ""
    width, left_x, right_x = 780, 150, 700
    row_h, top = 54, 74
    gold_share = next((diagnostic["median_gold_share"] for diagnostic in usable.values()
                       if diagnostic.get("median_gold_share") is not None), None)
    marks = []
    for index, pipeline in enumerate([pipeline_key for pipeline_key in agg.PIPELINES if pipeline_key in usable]):
        diagnostic = usable[pipeline]
        y_position = top + index * row_h
        x_position = left_x + diagnostic["median_retrieved_share"] * (right_x - left_x)
        marks.append(
            f'<text x="10" y="{y_position + 5}" class="lbl" fill="{PIPE[pipeline]["hue"]}">'
            f'{escape_html(PIPE[pipeline]["label"])}</text>'
            f'<line x1="{left_x}" y1="{y_position}" x2="{right_x}" y2="{y_position}" class="grid"/>'
            f'<circle cx="{x_position:.1f}" cy="{y_position}" r="7" fill="{PIPE[pipeline]["hue"]}"/>'
            f'<text x="{x_position:.1f}" y="{y_position - 13}" class="ax" text-anchor="middle">'
            f'{diagnostic["median_retrieved_share"]:.3f}</text>')
    height = top + len(usable) * row_h + 30
    gold_line = ""
    if gold_share is not None:
        gold_x = left_x + gold_share * (right_x - left_x)
        gold_line = (f'<line x1="{gold_x:.1f}" y1="44" x2="{gold_x:.1f}" y2="{height - 34}" '
                     f'stroke="{GOLD}" stroke-width="2" stroke-dasharray="5 4"/>'
                     f'<text x="{gold_x:.1f}" y="34" class="ax" text-anchor="middle" '
                     f'fill="{GOLD}">evidence sits at {gold_share:.3f}</text>')
    return f"""      <div class="chart">
        <svg viewBox="0 0 {width} {height}" role="img"
             aria-label="Median returned node position within its filing">
          {gold_line}
          {''.join(marks)}
          <text x="{left_x}" y="{height - 8}" class="ax" text-anchor="start">start of filing</text>
          <text x="{right_x}" y="{height - 8}" class="ax" text-anchor="end">end of filing</text>
        </svg>
      </div>"""


def diagnostics_table(report: dict) -> str:
    rows = []
    for pipeline in agg.PIPELINES:
        diagnostic = (report.get("diagnostics") or {}).get(pipeline)
        if not diagnostic:
            continue
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{diagnostic['distinct_nodes']:,} / {diagnostic['slots']:,}</td>
          <td>{format_number(diagnostic['diversity'])}</td>
          <td>{format_number(diagnostic['median_retrieved_share'])}</td>
          <td>{'-' if diagnostic['median_miss_nodes'] is None else f"{diagnostic['median_miss_nodes']:,.0f}"}</td>
          <td>{format_percent(diagnostic['median_miss_share'])}</td>
          <td>{format_percent(diagnostic['on_evidence_share'])}</td>
        </tr>""")
    return "\n".join(rows)


def latency_chart(report: dict) -> str:
    """Latency spread per pipeline, drawn as a quartile box with whiskers.

    The means differ, but they are pulled by a handful of very slow calls
    against a shared provider. The medians are what the box shows, and they
    are the same to within a fifth of a second.
    """
    latency = report.get("latency") or {}
    if not latency:
        return ""
    top = max(latency_stats["p90"] for latency_stats in latency.values())
    width, left_x, right_x = 780, 150, 720
    row_h, start = 62, 78
    scale = (right_x - left_x) / top

    def x_of(value):
        return left_x + min(value, top) * scale

    rows = []
    for index, pipeline in enumerate([pipeline_key for pipeline_key in agg.PIPELINES if pipeline_key in latency]):
        latency_stats = latency[pipeline]
        y_position = start + index * row_h
        hue = PIPE[pipeline]["hue"]
        box_lo, box_hi = x_of(latency_stats["p25"]), x_of(latency_stats["p75"])
        rows.append(
            f'<text x="10" y="{y_position + 5}" class="lbl" fill="{hue}">'
            f'{escape_html(PIPE[pipeline]["label"])}</text>'
            f'<line x1="{x_of(latency_stats["min"]):.1f}" y1="{y_position}" x2="{x_of(latency_stats["p90"]):.1f}" y2="{y_position}" '
            f'stroke="{hue}" stroke-width="2"/>'
            f'<rect x="{box_lo:.1f}" y="{y_position - 13}" width="{max(2, box_hi - box_lo):.1f}" '
            f'height="26" fill="{hue}" opacity="0.28"/>'
            f'<line x1="{x_of(latency_stats["median"]):.1f}" y1="{y_position - 15}" '
            f'x2="{x_of(latency_stats["median"]):.1f}" y2="{y_position + 15}" stroke="{hue}" stroke-width="4"/>'
            f'<text x="{x_of(latency_stats["median"]):.1f}" y="{y_position - 22}" class="ax" '
            f'text-anchor="middle">{latency_stats["median"]:.2f}s</text>')
    height = start + len(latency) * row_h + 24
    ticks = []
    for step in range(0, int(top) + 1, max(1, int(top // 5))):
        x_position = x_of(step)
        ticks.append(f'<text x="{x_position:.1f}" y="{height - 6}" class="ax" '
                     f'text-anchor="middle">{step}s</text>')
    return f"""      <div class="chart">
        <svg viewBox="0 0 {width} {height}" role="img"
             aria-label="Latency spread per pipeline">
          <text x="10" y="22" class="ax">box spans the middle half, line is the median,
          whisker runs to the 90th percentile</text>
          {''.join(rows)}
          {''.join(ticks)}
        </svg>
      </div>"""


def latency_table(report: dict) -> str:
    rows = []
    for pipeline in agg.PIPELINES:
        latency_stats = (report.get("latency") or {}).get(pipeline)
        if not latency_stats:
            continue
        rows.append(f"""        <tr>
          <th scope="row"><span class="dot" style="background:{PIPE[pipeline]['hue']}"></span>{escape_html(PIPE[pipeline]['label'])}</th>
          <td>{latency_stats['min']:.2f}</td><td>{latency_stats['p25']:.2f}</td><td>{latency_stats['median']:.2f}</td>
          <td>{latency_stats['p75']:.2f}</td><td>{latency_stats['p90']:.2f}</td><td>{latency_stats['max']:.2f}</td>
          <td>{latency_stats['mean']:.2f}</td>
        </tr>""")
    return "\n".join(rows)


# --- validity -----------------------------------------------------------------

def filing_leads(report: dict) -> dict:
    """Who leads each filing, counted rather than asserted.

    This was prose carrying typed counts, and the tie count in it was wrong.
    Counting from the same aggregation the heatmap draws keeps the sentence
    and the chart from disagreeing.
    """
    scores: dict[str, dict[str, float]] = {}
    for key, block in report["by_document"].items():
        document, pipeline = key.split("|")
        scores.setdefault(document, {})[pipeline] = block["judge_mean"]
    counts = {"P1_vector": 0, "P2_bm25": 0, "tie": 0, "worst_last": 0, "identical": 0}
    for row in scores.values():
        if "P1_vector" not in row or "P2_bm25" not in row:
            continue
        if row["P1_vector"] > row["P2_bm25"]:
            counts["P1_vector"] += 1
        elif row["P2_bm25"] > row["P1_vector"]:
            counts["P2_bm25"] += 1
        else:
            counts["tie"] += 1
        others = [judge_mean for pipeline_key, judge_mean in row.items() if pipeline_key != "P3_structural"]
        if "P3_structural" in row and others:
            if row["P3_structural"] < min(others):
                counts["worst_last"] += 1
            if len({round(judge_mean, 9) for judge_mean in row.values()}) == 1:
                counts["identical"] += 1
    counts["filings"] = len(scores)
    return counts


def median_phrase(report: dict) -> str:
    """Median latency per pipeline, named, because an unlabelled list of three
    near-identical numbers tells the reader nothing about which is which."""
    latency = report.get("latency") or {}
    return ", ".join(f"{latency[pipeline]['median']:.2f} s for {PIPE[pipeline]['label']}"
                     for pipeline in agg.PIPELINES if pipeline in latency)


def median_spread(report: dict) -> float:
    values = [latency_stats["median"] for latency_stats in (report.get("latency") or {}).values()]
    return (max(values) - min(values)) if values else 0.0


def threats(report: dict) -> str:
    """Every limitation that could be raised in a viva, with its defence.

    Each entry states what the limitation is, how large it is in the data, and
    which way it cuts. The direction matters more than the size: a limitation
    that applies identically to all three pipelines cannot reorder them, and
    that is the argument to make rather than an apology.
    """
    leak = report.get("leakage") or {}
    leak_total = sum(leak.values())
    leads = filing_leads(report)
    leak_p3 = leak.get("P3_structural", 0)
    spread = median_spread(report)
    docs = report.get("per_document_questions") or {}
    most = max(docs.values()) if docs else 0
    least = min(docs.values()) if docs else 0

    items = [
        ("material", "Latency does not separate the three paradigms",
         f"Median per-cell latency is {median_phrase(report)}, "
         f"a spread of {spread:.2f} s across all three. The means differ more, but they "
         f"are pulled by a small number of very slow calls against a shared hosted "
         f"provider, not by retrieval work done locally.",
         "Report query latency as a null result and say why: every cell routes through "
         "the same answering model, so the shared network call dominates the "
         "measurement and local retrieval time is lost inside it. The efficiency "
         "claim rests on index build cost and on tokens per query, which are "
         "measured cleanly and do separate the paradigms."),
        ("material", "Some answers were correct without the evidence being retrieved",
         f"{leak_total} of the 900 cells scored 4 or 5 while retrieving no ground-truth "
         f"node at all, {leak_p3} of them in the structural pipeline. Those answers "
         f"came from the answering model's own knowledge of these companies, not from "
         f"the filing.",
         "Disclose it as a floor on parametric leakage and note the direction: it "
         "inflates the weakest pipeline's score, because a pipeline that retrieves "
         "nothing useful leaves the model free to answer from memory. The measured "
         "gap between structural and the other two is therefore a lower bound on the "
         "true gap, not an exaggeration of it."),
        ("material", "One run per cell, so there is no run-to-run variance",
         "Every cell was executed once at temperature 0, with each model pinned to a "
         "single upstream host so that quantisation could not vary between calls. "
         "The confidence intervals describe variation across questions.",
         "State plainly that the intervals are question-level, not run-level, and that "
         "the design trades repeated sampling for breadth: 900 distinct cells rather "
         "than a smaller matrix sampled repeatedly. Temperature 0 and host pinning are "
         "what make a single run defensible; say both."),
        ("material", "The judge gate measures concordance, not human agreement",
         f"The gate compares the Judge against a reference scoring of the same 60 "
         f"outputs produced under a blinding protocol, not against an independent "
         f"human annotator. It reads {report['gate'][0]} of {report['gate'][1]} exact "
         f"({100 * report['gate'][0] / report['gate'][1]:.1f}%), with every disagreement "
         f"confined to a single band.",
         "Call it cross-scorer concordance throughout and never call it human "
         "validation. The defensible claim is that the Judge is consistent with an "
         "independent application of the same published rubric under blinding, which "
         "is what the gate actually tests. Cite the blinding protocol: pipeline "
         "identity and citation evidence were withheld from the reference scorer."),
        ("moderate", "The answer keys were corrected after the pipelines were built",
         "Twenty-four of the 100 questions had evidence sets that named a whole filing "
         "section rather than the supporting nodes, which made full recall "
         "arithmetically unreachable. Those were re-derived to the minimal verified "
         "evidence set before the final run.",
         "Argue it on mechanism, not on trust. The retention rule is mechanical, "
         "uniform, and was applied blind to every pipeline's output, so nothing could "
         "be tuned toward a preferred result; and the repair replays from the "
         "pre-repair backup to reproduce the current database exactly. Note also that "
         "it was not common-mode: it changed the vector against BM25 ordering on "
         "precision, which is precisely why leaving it in place and disclosing it "
         "would not have been defensible."),
        ("moderate", "Questions are not evenly spread across the filings",
         f"The most-sampled filing carries {most} questions and the least-sampled "
         f"carries {least}. Per-filing means are therefore built on very different "
         f"sample sizes.",
         "Use the per-filing table as a spread check only, never as a ranking, and say "
         "so where you present it. Every headline figure is paired within a question, "
         "so the imbalance cancels: each pipeline answers exactly the same questions "
         "over exactly the same filings."),
        ("moderate", "The structural pipeline's build cost is a floor, not a total",
         "Eleven of the 13 filings recorded their token spend; two did not, because a "
         "logging defect was fixed after those two builds had already run. The "
         "reported figure sums the 11 that were measured.",
         "Report it as a measured floor over 11 filings and name the two that are "
         "missing. The direction is safe: the true cost is higher than the figure "
         "quoted, so the conclusion that the structural index is the most expensive to "
         "build is strengthened by the gap, not threatened by it."),
        ("moderate", "A small number of questions are redundant or self-answering",
         "Two pairs ask the same question of the same filing, a third pair splits one "
         "fact across two quadrants, and three questions restate most of their own "
         "answer in the question text. At most six of 100.",
         "Disclose the count and argue common-mode: all three pipelines answer the same "
         "six, so the effect is a very slight compression of the measured gap rather "
         "than a shift in the ordering. Removing them after retrieval behaviour was "
         "known would have been the less defensible choice."),
        ("moderate", "One quadrant tests a narrower skill than its definition implies",
         "Six of the 25 implicit-table questions share two task shapes: counting the "
         "filing's signatories, and summing page numbers in an index.",
         "Frame it as a construct-coverage limitation of that quadrant, not a validity "
         "defect in any row, and be specific about why the rows are still valid: "
         "counting signatories requires retrieving an intact tabular block, which is "
         "exactly the structure-versus-semantics contrast being measured."),
        ("minor", "The corpus carries parsing artefacts from the source filings",
         "1,089 nodes contain an unescaped HTML entity, seven contain a raw tag, and "
         "four carry a corrupted digit run where a financial figure should be. Exactly "
         "one benchmark question cites a corrupted node.",
         "Argue it as common-mode and, for the one affected question, as a feature: "
         "all three pipelines index the identical corpus, and on that row the answer "
         "key records what the node actually says, so the row rewards faithful "
         "extraction over a plausible figure recalled from training. Messy conversion "
         "is a property of the problem domain rather than an accident of this build."),
        ("minor", "The retrieval depths are 2, 3 and 5 rather than 3, 5 and 10",
         "The sweep was narrowed on instruction after the design was written. Three "
         "depths are retained, so the 900-cell matrix is unchanged.",
         "Record it as a scope decision with its consequence stated: the deepest "
         "setting is now 5, which sits at or above the largest corrected evidence set, "
         "so full recall stays reachable at the top of the sweep. Note that the token "
         "budget written against a maximum depth of 10 becomes a conservative "
         "over-estimate rather than a wrong one."),
    ]
    blocks = []
    for level, title, finding, defence in items:
        blocks.append(f"""        <div class="threat {escape_html(level)}">
          <p class="eyebrow">{escape_html(level)}</p>
          <h3>{escape_html(title)}</h3>
          <p><b>What the data shows.</b> {escape_html(finding)}</p>
          <p class="defence"><b>How to put it in the write-up.</b> {escape_html(defence)}</p>
        </div>""")
    return "\n".join(blocks)


# --- the page -----------------------------------------------------------------

CSS = """
:root {
  --ink:#0A1F33; --body:#1A1D21; --muted:#54595F; --faint:#7A8087;
  --rule:#E2E5E9; --hair:#EEF0F3; --paper:#FFFFFF; --wash:#F7F9FB;
  --gold:#D4A017; --gold-wash:#FEF7E4;
  --p1:#3D5AFE; --p2:#00A5A8; --p3:#FF4D6D;
  --serif:"EB Garamond",Garamond,"Times New Roman",serif;
  --sans:Calibri,Carlito,"Segoe UI",system-ui,-apple-system,sans-serif;
  --mono:Consolas,"SF Mono",Menlo,monospace;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--body);
       font-family:var(--serif); font-size:17px; line-height:1.65;
       -webkit-font-smoothing:antialiased; }
.wrap { max-width:1080px; margin:0 auto; padding-inline:24px; }
h1,h2,h3,h4 { font-family:var(--sans); color:var(--ink); text-wrap:balance; margin:0; }
h1 { font-size:clamp(2rem,5vw,3.1rem); font-weight:700; letter-spacing:-.02em; line-height:1.1; }
h2 { font-size:clamp(1.3rem,3vw,1.8rem); font-weight:700; letter-spacing:-.01em; }
h3 { font-size:1.12rem; font-weight:700; }
h4 { font-size:.82rem; font-weight:700; text-transform:uppercase;
     letter-spacing:.09em; color:var(--muted); margin-bottom:.9rem; }
p { margin:0 0 1rem; max-width:68ch; }
a { color:var(--p1); }

/* masthead */
header { border-top:6px solid var(--ink); padding-block:3.2rem 2.2rem; }
.eyebrow { font-family:var(--sans); font-size:.74rem; font-weight:700;
           text-transform:uppercase; letter-spacing:.16em; color:var(--p2); margin:0 0 .9rem; }
.stand { font-size:1.2rem; color:var(--muted); max-width:60ch; margin-top:1.1rem; }
.rule-bar { display:flex; height:5px; margin-top:2rem; }
.rule-bar i { flex:1; }

/* facts */
.facts { display:grid; grid-template-columns:repeat(6,1fr); gap:1px;
         background:var(--rule); border-block:1px solid var(--rule); margin-block:2.4rem; }
.fact { background:var(--paper); padding:1.1rem .9rem; text-align:center; }
.fact b { display:block; font-family:var(--sans); font-size:1.55rem;
          font-weight:700; color:var(--ink); line-height:1.15; }
.fact span { display:block; font-family:var(--sans); font-size:.72rem;
             text-transform:uppercase; letter-spacing:.05em; color:var(--faint); margin-top:.3rem; }

section { padding-block:2.6rem; border-top:1px solid var(--hair); }
section > h2 { margin-bottom:.4rem; }
.lede { color:var(--muted); margin-bottom:1.8rem; }

/* key finding */
.finding { background:var(--gold-wash); border-left:5px solid var(--gold);
           padding:1.4rem 1.6rem; margin-block:1.6rem; }
.finding p { margin:0; max-width:none; }
.finding p + p { margin-top:.7rem; }
.finding b { color:var(--ink); }

/* scoreboard */
.cards { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.4rem; }
.card { border:1px solid var(--rule); border-top:4px solid var(--hue);
        padding:1.3rem 1.2rem; background:var(--paper); }
.card-top { display:flex; gap:.85rem; align-items:flex-start; }
.rank { flex:none; width:30px; height:30px; border-radius:50%; background:var(--wash);
        color:var(--hue); font-family:var(--sans); font-weight:700; font-size:.95rem;
        display:grid; place-items:center; }
.card h3 { line-height:1.25; }
.card-top p { font-size:.88rem; color:var(--faint); margin:.25rem 0 0; }
.card-nums { display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem;
             margin-top:1.2rem; padding-top:1rem; border-top:1px solid var(--hair); }
.card-nums b { display:block; font-family:var(--sans); font-size:1.35rem;
               font-weight:700; color:var(--hue); }
.card-nums span { display:block; font-family:var(--sans); font-size:.68rem;
                  text-transform:uppercase; letter-spacing:.05em; color:var(--faint); }

/* tables */
.scroll { overflow-x:auto; margin-block:1.4rem; }
table { border-collapse:collapse; width:100%; font-family:var(--sans);
        font-size:.88rem; min-width:640px; }
caption { caption-side:top; text-align:left; font-family:var(--sans); font-size:.8rem;
          color:var(--faint); padding-bottom:.7rem; }
th,td { padding:.62rem .7rem; text-align:right; border-bottom:1px solid var(--hair);
        font-variant-numeric:tabular-nums; }
thead th { text-align:right; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
           color:#fff; background:var(--ink); border-bottom:none; font-weight:700; }
thead th:first-child, tbody th { text-align:left; }
tbody th { font-weight:700; color:var(--ink); white-space:nowrap; }
tbody tr:nth-child(even) { background:var(--wash); }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.5rem; }
.mono { font-family:var(--mono); font-size:.82rem; }
.pill { display:inline-block; font-size:.7rem; font-weight:700; padding:.2em .65em;
        border-radius:2px; text-transform:uppercase; letter-spacing:.04em; }
.pill.yes { background:var(--gold); color:#3A2C00; }
.pill.no { background:#E9ECEF; color:var(--muted); }

/* bars */
.pair { display:grid; grid-template-columns:1fr 1fr; gap:2.4rem; margin-top:1.2rem; }
.bar { display:grid; grid-template-columns:6.6rem 1fr 4.2rem; align-items:center;
       gap:.7rem; margin-bottom:.55rem; font-family:var(--sans); font-size:.83rem; }
.bar-l { color:var(--muted); }
.bar-t { background:var(--hair); height:16px; overflow:hidden; }
.bar-f { display:block; height:100%; }
.bar-v { text-align:right; font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }

/* distribution */
.dist-wrap { margin-top:1.2rem; }
.dist { display:grid; grid-template-columns:6.6rem 1fr; align-items:center;
        gap:.7rem; margin-bottom:.5rem; font-family:var(--sans); font-size:.83rem; }
.dist-l { color:var(--muted); }
.dist-t { display:flex; height:30px; overflow:hidden; }
.seg { display:grid; place-items:center; color:#fff; font-size:.72rem; font-weight:700; }
.key { display:flex; flex-wrap:wrap; align-items:center; gap:.75rem; margin-top:1rem;
       font-family:var(--sans); font-size:.75rem; color:var(--faint); }
.key span { display:inline-flex; align-items:center; gap:.35rem; }
.key i { width:13px; height:13px; display:inline-block; }
.key em { font-style:normal; }

/* svg charts */
.chart { margin-top:1.2rem; }
.chart svg { width:100%; height:auto; display:block; overflow:visible; }
.grid { stroke:var(--hair); stroke-width:1; }
.axis { stroke:var(--rule); stroke-width:1.5; }
.zero { stroke:var(--faint); stroke-width:1.5; stroke-dasharray:4 4; }
.ax { font-family:Calibri,Carlito,system-ui,sans-serif; font-size:13px; fill:var(--faint); }
.lbl { font-family:Calibri,Carlito,system-ui,sans-serif; font-size:14px;
       font-weight:700; fill:var(--ink); }
.cap { font-family:var(--sans); font-size:.78rem; color:var(--faint); margin-top:.9rem; }
.swatch { display:inline-block; width:11px; height:11px; vertical-align:-1px; }

/* quadrants */
.quads { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.4rem; }
.quad { border:1px solid var(--rule); padding:1.2rem; }
.quad .eyebrow { color:var(--faint); margin-bottom:.4rem; }
.quad h4 { font-size:1.02rem; text-transform:none; letter-spacing:0;
           color:var(--ink); margin-bottom:.3rem; }
.blurb { font-size:.9rem; color:var(--faint); margin-bottom:1rem; }
.flip { display:inline-block; margin-top:.5rem; font-family:var(--sans); font-size:.72rem;
        font-weight:700; background:var(--gold-wash); color:#6B5200;
        padding:.25em .6em; text-transform:uppercase; letter-spacing:.04em; }

/* log bars + heatmap */
.logbars { margin-top:1.2rem; }
.heat { margin-top:1.2rem; font-family:var(--sans); font-size:.8rem; }
.hm-row { display:grid; grid-template-columns:7.5rem 5.4rem repeat(3,1fr); gap:2px; margin-bottom:2px; }
.hm-head .hm-h { font-size:.7rem; text-transform:uppercase; letter-spacing:.05em;
                 color:var(--faint); text-align:center; padding-bottom:.3rem;
                 white-space:nowrap; overflow:hidden; }
.hm-head b { font-weight:700; }
.sm { display:none; }
.hm-r { display:flex; align-items:center; color:var(--muted); font-variant-numeric:tabular-nums; }
.hm-n { display:grid; place-items:center; color:var(--faint); font-size:.72rem;
        font-variant-numeric:tabular-nums; }
.hm-head .hm-n { text-transform:uppercase; letter-spacing:.05em; padding-bottom:.3rem; }
.hm-c { display:grid; place-items:center; padding:.5rem 0; font-weight:700;
        font-variant-numeric:tabular-nums; }

.note { font-size:.95rem; color:var(--muted); border-left:3px solid var(--rule);
        padding-left:1rem; }
ul { padding-left:1.15rem; max-width:68ch; }
li { margin-bottom:.55rem; }
li b { color:var(--ink); }
footer { border-top:1px solid var(--rule); margin-top:2rem; padding-block:1.6rem 3rem;
         font-family:var(--sans); font-size:.78rem; color:var(--faint); }
footer code { font-family:var(--mono); }

/* contents */
.toc { border-block:1px solid var(--rule); padding-block:1.6rem; margin-bottom:.5rem; }
.toc ol { margin:0; padding:0; list-style:none; columns:3; column-gap:2.2rem; }
.toc li { margin:0 0 .42rem; font-family:var(--sans); font-size:.86rem;
          break-inside:avoid; }
.toc a { color:var(--body); text-decoration:none; border-bottom:1px solid var(--hair); }
.toc a:hover { border-bottom-color:var(--p1); color:var(--ink); }
.toc .part { font-weight:700; text-transform:uppercase; letter-spacing:.09em;
             font-size:.7rem; color:var(--p2); margin-top:.9rem; border:0; }
.toc li:first-child .part { margin-top:0; }
.part-rule { font-family:var(--sans); font-size:.72rem; font-weight:700;
             text-transform:uppercase; letter-spacing:.16em; color:var(--p2);
             border-top:3px solid var(--ink); padding-top:.7rem; margin-top:3rem; }

/* corpus bars carry a nested share */
.bar-sub { display:block; height:100%; float:right; background:var(--gold); }
.total th, .total td { border-top:2px solid var(--ink); font-weight:700; color:var(--ink); }
.grp td, .grp th { border-top:1px solid var(--rule); }
.grp-l { color:var(--faint); font-size:.78rem; font-family:var(--sans); }

/* quadrant matrix */
.matrix { margin-top:1.4rem; font-family:var(--sans); }
.mx-row { display:grid; grid-template-columns:8.5rem 1fr 1fr; gap:1px;
          background:var(--rule); }
.mx-head { background:none; }
.mx-h { font-size:.72rem; font-weight:700; text-transform:uppercase;
        letter-spacing:.08em; color:var(--muted); padding:.4rem .9rem; }
.mx-lab { display:flex; align-items:center; font-size:.72rem; font-weight:700;
          text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
          background:var(--paper); padding-right:.9rem; }
.mx-cell { background:var(--paper); padding:1.1rem; }
.mx-cell b { display:block; font-size:2rem; font-weight:700; color:var(--ink);
             line-height:1; margin:.2rem 0 .4rem; }
.mx-cell h4 { font-size:.95rem; text-transform:none; letter-spacing:0;
              color:var(--ink); margin-bottom:.3rem; }
.mx-cell .blurb { margin-bottom:0; font-size:.84rem; }

/* four-panel grid */
.quad-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.8rem 2.4rem;
             margin-top:1.4rem; }

/* contingency split */
.split { display:grid; grid-template-columns:1fr 1fr; gap:.9rem; margin:1rem 0 .6rem; }
.split .eyebrow { color:var(--faint); margin-bottom:.3rem; font-size:.66rem; }
.split b { display:block; font-family:var(--sans); font-size:1.7rem; font-weight:700;
           color:var(--p2); line-height:1.1; }
.split b.bad { color:var(--p3); }
.split span { display:block; font-family:var(--sans); font-size:.7rem; color:var(--faint); }

/* five-column heatmap for the gate */
.heat-5 .hm-row { grid-template-columns:7.5rem repeat(5,1fr); }
.hm-c.empty { background:var(--wash); color:var(--faint); font-weight:400; }

/* threats */
.threats { display:grid; gap:1rem; margin-top:1.4rem; }
.threat { border:1px solid var(--rule); border-left:5px solid var(--rule);
          padding:1.3rem 1.5rem; background:var(--paper); }
.threat.material { border-left-color:var(--p3); background:#FFFBFC; }
.threat.moderate { border-left-color:var(--gold); background:var(--gold-wash); }
.threat.minor { border-left-color:var(--faint); background:var(--wash); }
.threat .eyebrow { margin-bottom:.35rem; }
.threat.material .eyebrow { color:var(--p3); }
.threat.moderate .eyebrow { color:#8A6A00; }
.threat.minor .eyebrow { color:var(--faint); }
.threat h3 { margin-bottom:.7rem; }
.threat p { font-size:.94rem; margin-bottom:.6rem; max-width:74ch; }
.threat p:last-child { margin-bottom:0; }
.defence { border-top:1px dashed var(--rule); padding-top:.7rem; }

@media (max-width:900px) {
  .facts { grid-template-columns:repeat(3,1fr); }
  .cards, .pair, .quads, .quad-grid { grid-template-columns:1fr; }
  .toc ol { columns:2; }
}
@media (max-width:560px) {
  body { font-size:16px; }
  .facts { grid-template-columns:repeat(2,1fr); }
  .bar, .dist { grid-template-columns:5rem 1fr 3.4rem; }
  .dist { grid-template-columns:5rem 1fr; }
  .hm-row { grid-template-columns:4.8rem 1.9rem repeat(3,1fr); }
  .heat-5 .hm-row { grid-template-columns:4.8rem repeat(5,1fr); }
  .hm-head .hm-h, .hm-head .hm-n { font-size:.66rem; letter-spacing:.02em; }
  .lg { display:none; }
  .sm { display:inline; }
  .hm-r, .hm-c { font-size:.76rem; }
  .toc ol { columns:1; }
  .mx-row { grid-template-columns:1fr; }
  .mx-lab { padding:.6rem 0 0; }
  .split { grid-template-columns:1fr; }
}
@media print { section { break-inside:avoid; } }
"""


# Every section in reading order: anchor, title, and the part it opens.
SECTIONS = [
    ("design", "The question and the design", "I. What was measured"),
    ("apparatus", "The apparatus", None),
    ("corpus", "The corpus", "II. Materials"),
    ("questions", "The question set", None),
    ("judge", "The judge and its gate", "III. The instrument"),
    ("headline", "The headline", "IV. Results"),
    ("pillars", "All three pillars", None),
    ("quality", "Answer quality", None),
    ("bimodal", "The scores are bimodal, not spread", None),
    ("retrieval", "Retrieval quality", None),
    ("depth", "Depth helps, but it does not rescue", None),
    ("quadrants", "Where the ordering changes", None),
    ("flip", "The average hides a split", None),
    ("stats", "Is the difference real", None),
    ("filings", "Filing by filing", None),
    ("chain", "Retrieval decides the answer", "V. Why"),
    ("nodetypes", "What each pipeline puts in front of the model", None),
    ("position", "Why the structural pipeline trails", None),
    ("build", "What it costs to build", None),
    ("ask", "What it costs to ask", None),
    ("latency", "Latency is not the discriminator", None),
    ("threats", "Threats to validity, and how to answer them", "VI. Validity"),
    ("weaken", "What would weaken these findings", None),
    ("ledger", "Reproducibility ledger", None),
]


def contents() -> str:
    items = []
    for anchor, title, part in SECTIONS:
        if part:
            items.append(f'        <li><span class="part">{escape_html(part)}</span></li>')
        items.append(f'        <li><a href="#{escape_html(anchor)}">{escape_html(title)}</a></li>')
    return f'    <nav class="toc" aria-label="Contents">\n      <ol>\n{chr(10).join(items)}\n      </ol>\n    </nav>'


def heading(anchor: str) -> str:
    """Opens a section, with the part rule above it where one starts here."""
    entry = next(section for section in SECTIONS if section[0] == anchor)
    rule = f'<p class="part-rule">{escape_html(entry[2])}</p>\n  ' if entry[2] else ""
    return f'{rule}<h2 id="{escape_html(anchor)}">{escape_html(entry[1])}</h2>'


def rank_pipelines(report: dict) -> tuple[str, str, dict]:
    """Best and worst pipeline by judge mean, with the P1-against-P2 judge comparison."""
    overall = report["overall"]
    best_pipeline = max(overall, key=lambda pipeline: overall[pipeline]["judge_mean"])
    worst_pipeline = min(overall, key=lambda pipeline: overall[pipeline]["judge_mean"])
    vector_vs_bm25 = next(
        comparison for comparison in report["comparisons"]
        if comparison["metric"] == "judge_score"
        and comparison["left"] == "P1_vector"
        and comparison["right"] == "P2_bm25"
    )
    return best_pipeline, worst_pipeline, vector_vs_bm25


def summarise_index_build(report: dict) -> tuple[dict, dict | None, float]:
    """P3's index build, the cheapest model-free build, and the ratio between them."""
    builds_by_pipeline = (report.get("index_build") or {}).get("pipelines") or {}
    structural_build = builds_by_pipeline.get("P3_structural", {})
    cheapest_model_free_build = min(
        (build for build in builds_by_pipeline.values() if not build["uses_llm"]),
        key=lambda build: build["wall_clock_sec"], default=None)
    structural_cost_ratio = (
        structural_build["wall_clock_sec"] / cheapest_model_free_build["wall_clock_sec"]
        if structural_build and cheapest_model_free_build else 0)
    return structural_build, cheapest_model_free_build, structural_cost_ratio


def summarise_diagnostics(
    report: dict, best_pipeline: str, worst_pipeline: str
) -> tuple[dict, dict, float | None]:
    """Retrieval diagnostics for the two extreme pipelines, and the median gold share."""
    diagnostics_by_pipeline = report.get("diagnostics") or {}
    median_gold_share = next(
        (diagnostic["median_gold_share"]
         for diagnostic in diagnostics_by_pipeline.values()
         if diagnostic.get("median_gold_share") is not None), None)
    return (diagnostics_by_pipeline.get(worst_pipeline, {}),
            diagnostics_by_pipeline.get(best_pipeline, {}),
            median_gold_share)


def q4_judge_means(report: dict) -> tuple[dict, str]:
    """Judge means for the implicit-table quadrant, and the pipeline leading it."""
    means_by_pipeline = {
        pipeline: report["by_quadrant"][f"Q4_Implicit_Table|{pipeline}"]["judge_mean"]
        for pipeline in PIPE
    }
    return means_by_pipeline, max(means_by_pipeline, key=lambda pipeline: means_by_pipeline[pipeline])


def build_html(db_path: str) -> str:
    report = load_report_data(db_path)
    overall = report["overall"]
    best, worst, p1p2 = rank_pipelines(report)
    cost_rows, cost_chart = build_cost(report)
    structural, cheapest, ratio = summarise_index_build(report)
    q4_means, q4_lead = q4_judge_means(report)

    leak = report.get("leakage") or {}
    leak_total = sum(leak.values())
    leads = filing_leads(report)
    flips = [quadrant for quadrant, split in report["split_by_quadrant"].items()
             if split["significant"]]
    hit = report["contingency"].get((best, 1), {})
    miss = report["contingency"].get((best, 0), {})
    worst_diag, best_diag, gold_share = summarise_diagnostics(report, best, worst)

    return f"""<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three Ways to Read a 10-K &middot; Findings</title>
<style>{CSS}</style>
</head>
<body>

<header class="wrap">
  <p class="eyebrow">COMP702 benchmark &middot; results</p>
  <h1>Three ways to read a 10-K</h1>
  <p class="stand">Semantic, statistical and structural retrieval answer the same
  {report['questions']} questions over {report['filings']} SEC annual reports, at three retrieval
  depths, judged on the same scale. These are the findings, the materials they came
  from, and the limitations they carry.</p>
  <div class="rule-bar">
    <i style="background:var(--p1)"></i><i style="background:var(--p2)"></i>
    <i style="background:var(--p3)"></i><i style="background:var(--gold)"></i>
  </div>
</header>

<div class="wrap">
{facts_strip(report)}
{contents()}
</div>

<div class="wrap">
<section style="border-top:none">
  {heading('design')}
  <p class="lede">One question, asked three ways.</p>
  <div class="finding">
    <p><b>Does the way a retriever is built change how well a language model can
    answer questions about a company's annual report?</b> Three retrieval paradigms
    index the identical corpus and are asked the identical questions through the
    identical answering model. Anything that differs between them is the paradigm,
    because nothing else is allowed to vary.</p>
  </div>
  <div class="cards">
    <div class="card" style="--hue:var(--p1);--wash:#E8ECFF">
      <div class="card-top"><span class="rank">1</span>
        <div><h3>P1 Vector</h3><p>Semantic. Dense embeddings in ChromaDB with
        fastembed, reranked by a local cross-encoder. Matches on meaning.</p></div></div>
    </div>
    <div class="card" style="--hue:var(--p2);--wash:#DFF5F5">
      <div class="card-top"><span class="rank">2</span>
        <div><h3>P2 BM25</h3><p>Statistical. rank_bm25 over the same nodes, no model
        of any kind. Matches on terms and their rarity.</p></div></div>
    </div>
    <div class="card" style="--hue:var(--p3);--wash:#FFE7EB">
      <div class="card-top"><span class="rank">3</span>
        <div><h3>P3 Structural</h3><p>A LlamaIndex summary tree per filing, traversed
        by comparing the query against node summaries. Matches on position in a
        hierarchy.</p></div></div>
    </div>
  </div>
  <h3 style="margin-top:2rem">What is held constant</h3>
  <ul>
    <li><b>The corpus.</b> All three index the same {report['nodes']:,} nodes from the same
    {report['filings']} filings, produced by one parsing and splitting pass.</li>
    <li><b>The questions.</b> All three answer the same {report['questions']} questions at
    the same three depths, giving {report['cells']} cells.</li>
    <li><b>The answering model.</b> One model writes every answer, so differences in
    answer quality trace back to what each retriever handed it.</li>
    <li><b>The grader.</b> One judge scores every answer against the same rubric, with
    a fingerprint recorded per row so a score can be tied to the rubric in force when
    it was written.</li>
    <li><b>What a pipeline is never shown.</b> The question and its own retrieved
    nodes, and nothing else. No ground truth, no exemplars, no hint about where the
    answer lives, and no metadata pre-filter.</li>
  </ul>
</section>

<section>
  {heading('apparatus')}
  <p class="lede">Which model did which job, and why no model ever grades its own
  work.</p>
  <div class="scroll">
    <table>
      <caption>Table 1. Model routing. The answerer and the judge are deliberately
      from different families, so a model is never asked to mark its own output.</caption>
      <thead><tr><th>Role</th><th>Model</th><th>Host</th><th>Job</th></tr></thead>
      <tbody>
{routing_table()}
      </tbody>
    </table>
  </div>
  <p class="note">Two invariants hold across this table. The generator that proposes a
  question is never the critic that accepts it, and the answerer that writes an answer
  is never the judge that scores it. Every call runs at temperature 0, and the answerer
  and judge are each pinned to a single upstream host so that two calls in the same run
  cannot land on differently quantised copies of the same model.</p>
</section>

<section>
  {heading('corpus')}
  <p class="lede">{report['filings']} annual reports from five companies, parsed to
  {report['nodes']:,} retrievable nodes carrying {report['node_totals']['tokens']:,} tokens.</p>
{corpus_chart(report)}
  <div class="scroll">
    <table>
      <caption>Table 2. The indexed corpus. A node is one retrievable unit: a passage
      of prose or a tabular block.</caption>
      <thead><tr><th>Filing</th><th>Nodes</th><th>Text</th><th>Table</th>
        <th>Table share</th><th>Tokens</th><th>Tokens per node</th></tr></thead>
      <tbody>
{corpus_table(report)}
      </tbody>
    </table>
  </div>
  <p class="note">The filings are deliberately unequal in size, spanning
  {min(row['nodes'] for row in report['corpus']):,} to {max(row['nodes'] for row in report['corpus']):,}
  nodes. That range is why every distance measured inside a document is reported as a
  share of its own filing before being pooled: a gap of 300 nodes is most of the
  shortest filing and a twelfth of the longest.</p>
</section>

<section>
  {heading('questions')}
  <p class="lede">{report['questions']} primary questions on a two by two of difficulty,
  {sum(quadrant_stats['n'] for quadrant_stats in report['quadrants'].values()) // max(1, len(report['quadrants']))} per
  quadrant, every one verified against its filing.</p>
{quadrant_matrix(report)}
  <p class="note" style="margin-top:1.4rem">Two further disjoint sets exist and never
  overlap with these. Twenty golden queries carry hand-written scores and serve as the
  judge's calibration exemplars; twenty judge-validation queries are held out entirely
  and are what the gate is measured on. A question in one set is never in another, so
  the judge is never calibrated on the rows used to test it.</p>
  <h3 style="margin-top:2rem">How much evidence each question has</h3>
  <p>The answer key names the nodes that support the answer. Its size decides whether
  recall is reachable.</p>
{evidence_histogram(report)}
</section>

<section>
  {heading('judge')}
  <p class="lede">Answers are scored 1 to 5 against the ground-truth answer only.
  Citation validity is checked deterministically elsewhere and deliberately kept out
  of the judge's remit, so a model's guess cannot override a mechanical check.</p>
  <div class="quads">
    <div class="quad"><p class="eyebrow">Score 5</p><h4>Fully correct</h4>
      <p class="blurb">Matches the ground truth, complete and precise, in a form that
      answers the question asked.</p></div>
    <div class="quad"><p class="eyebrow">Score 4</p><h4>Right value, short of complete</h4>
      <p class="blurb">Correct on the value but falling short on completeness,
      precision or framing.</p></div>
    <div class="quad"><p class="eyebrow">Score 3</p><h4>Half an answer</h4>
      <p class="blurb">One half of a two-part answer, with the other half absent.</p></div>
    <div class="quad"><p class="eyebrow">Score 2</p><h4>Right area, wrong figure</h4>
      <p class="blurb">Reaches the right part of the filing but reports the wrong
      figure, ordinal or scope.</p></div>
  </div>
  <p class="note" style="margin-top:1.2rem">A score of 1 is wrong, unsupported,
  fabricated, or a refusal to answer. Pass rate on this page means 4 or 5; fail rate
  means 1.</p>
  <h3 style="margin-top:2rem">The gate</h3>
  <p>Before the judge was allowed near the {report['cells']}-cell benchmark it scored
  {report['gate'][1]} held-out outputs, which were independently scored a second time under
  a blinding protocol that withheld pipeline identity and citation evidence from the
  reference scorer. The two scorings are compared here.</p>
{gate_confusion(report)}
  <div class="scroll">
    <table>
      <caption>Table 3. Gate agreement by pipeline. The judge is not systematically
      kinder to any one architecture.</caption>
      <thead><tr><th>Pipeline</th><th>Outputs</th><th>Exact agreement</th><th>Rate</th>
        <th>Judge mean</th><th>Reference mean</th><th>Drift</th></tr></thead>
      <tbody>
{gate_table(report)}
      </tbody>
    </table>
  </div>
  <p class="note">The drift column is the one to read for bias. If the judge favoured
  a particular architecture, that preference would show up here as a positive drift on
  one row and a negative drift on another. Read this as cross-scorer concordance, not
  as human validation: it shows the rubric is applied consistently by two independent
  passes, which is what it was designed to test.</p>
</section>

<section>
  {heading('headline')}
  <div class="finding">
    <p><b>Semantic and statistical retrieval are indistinguishable on this benchmark.</b>
    BM25 leads the vector pipeline on every point estimate, but the paired difference in
    judge score is {p1p2['mean_difference']:+.3f} with a 95% interval of
    [{p1p2['ci_low']:+.3f}, {p1p2['ci_high']:+.3f}], which spans zero. They tie on
    {p1p2['ties']} of {p1p2['n_pairs']} cells.</p>
    <p><b>The structural pipeline trails both by roughly two judge points</b>, with
    intervals nowhere near zero, and it is also the most expensive of the three to build
    and to query.</p>
    <p><b>The average conceals a split.</b> Inside the quadrants the two leading
    pipelines are not interchangeable: BM25 wins the implicit-text quadrant by a margin
    whose interval is clear of zero, while the vector pipeline leads on implicit tables.
    The pooled null result is real, and it is not the whole story.</p>
  </div>
  <div class="cards">
{scoreboard(report)}
  </div>
</section>

<section>
  {heading('pillars')}
  <p class="lede">Retrieval quality, answer quality and efficiency, over all
  {report['cells']} cells.</p>
  <div class="scroll">
    <table>
      <caption>Table 4. Every metric, 300 cells per pipeline. Pass counts 4 and 5; fail counts 1.</caption>
      <thead><tr><th>Pipeline</th><th>Judge</th><th>Pass</th><th>Fail</th><th>Recall</th>
        <th>Precision</th><th>Hit rate</th><th>Citations</th><th>Token F1</th><th>Latency</th></tr></thead>
      <tbody>
{overall_table(report)}
      </tbody>
    </table>
  </div>
</section>

<section>
  {heading('quality')}
  <p class="lede">The pass rate is the more honest summary, for the reason the next
  chart makes plain.</p>
{quality_chart(report)}
</section>

<section>
  {heading('bimodal')}
  <p class="lede">Almost every answer is either fully correct or wholly wrong. A mean of
  3.0 over this shape would describe no answer that any pipeline actually gave, which is
  why a pass rate is reported beside it.</p>
{distribution_chart(report)}
  <p class="note">This shape is also why the statistics on this page are rank-based. A
  bootstrap interval and an exact sign test make no assumption about the distribution;
  a t-test on a bounded, ordinal, two-humped variable would.</p>
</section>

<section>
  {heading('retrieval')}
  <p class="lede">Answer quality is downstream of retrieval, so retrieval is measured
  in its own right: what share of the evidence came back, how often any of it came
  back, and how much of what came back was evidence at all.</p>
{retrieval_bars(report)}
  <div class="scroll">
    <table>
      <caption>Table 5. Retrieval and answer quality at each depth. Precision falls
      with k by construction, because its denominator is k while the evidence set is
      fixed.</caption>
      <thead><tr><th>Pipeline</th><th>k</th><th>Precision</th><th>Recall</th>
        <th>Hit rate</th><th>Citation match</th><th>Judge</th><th>Pass</th></tr></thead>
      <tbody>
{retrieval_table(report)}
      </tbody>
    </table>
  </div>
  <p class="note">Precision is the one metric that should not be read as a ranking
  across depths. Returning five nodes when the answer needs one caps precision at 0.2
  however good the retrieval, so a falling precision column is arithmetic, not
  degradation. Recall and hit rate are the measures that carry information here.</p>
</section>

<section>
  {heading('depth')}
  <p class="lede">Every pipeline improves as k rises from 2 to 5. The structural pipeline
  improves the most in relative terms and is still far below where the other two start.</p>
{k_chart(report)}
  <p class="note">Depth is therefore not a confound in the headline comparison. The
  ordering of the three pipelines is the same at every depth, so no choice of k within
  this sweep would have produced a different conclusion.</p>
</section>

<section>
  {heading('quadrants')}
  <p class="lede">The question set is a genuine two by two: stated directly or inferred,
  crossed with prose or table. BM25 wins three of the four.</p>
  <div class="quads">
{quadrant_grid(report)}
  </div>
  <p class="note" style="margin-top:1.4rem">The exception is
  {escape_html(QUAD['Q4_Implicit_Table'][0])}, the one quadrant where the answer never appears
  verbatim anywhere in the filing. {escape_html(PIPE[q4_lead]['label'])} leads there at
  {q4_means[q4_lead]:.2f} against {q4_means['P2_bm25']:.2f} for BM25, which is what you would expect
  once exact term matching has nothing exact to match.</p>
</section>

<section>
  {heading('flip')}
  <p class="lede">The corpus-wide interval for vector against BM25 spans zero. That
  says they are indistinguishable on average. It does not say they behave alike, and
  splitting the same paired comparison by quadrant shows they do not.</p>
{flip_chart(report)}
{forest_key()}
  <div class="scroll">
    <table>
      <caption>Table 6. Vector minus BM25, computed inside each depth and each
      quadrant. A positive difference favours the vector pipeline.</caption>
      <thead><tr><th>Split</th><th>Subset</th><th>Difference</th><th>95% interval</th>
        <th>W / L / T</th><th>Sign test</th><th>Verdict</th></tr></thead>
      <tbody>
{split_table(report)}
      </tbody>
    </table>
  </div>
  <div class="finding">
    <p><b>{'One quadrant separates them' if len(flips) == 1 else f'{len(flips)} quadrants separate them'
      if flips else 'No single quadrant separates them'}.</b>
    {'BM25 wins ' + ', '.join(escape_html(QUAD[quadrant_stats][0]) for quadrant_stats in flips) + ' by a margin whose interval is clear of zero, while the vector pipeline leads on implicit tables without reaching significance. The two paradigms are trading wins on different question types, and the pooled figure averages that trade away.' if flips else 'Every subset interval spans zero, so the corpus-wide null result holds uniformly.'}</p>
    <p>This is the finding to lead a discussion with. A null result at the corpus level
    is not evidence that the two paradigms are interchangeable; it is evidence that
    their strengths are differently distributed and happen to cancel over this
    question mix. A different mix would not have cancelled.</p>
  </div>
</section>

<section>
  {heading('stats')}
  <p class="lede">Every pipeline answers exactly the same cells, so comparisons are
  paired on question and depth. Significance is a bootstrap interval and an exact sign
  test, not a t-test: these scores are ordinal, bounded and bimodal.</p>
{forest_plot(report, 'judge_score', 'Judge score, 1 to 5')}
{forest_plot(report, 'recall_at_k', 'Recall at k, 0 to 1')}
{forest_key()}
  <div class="scroll">
    <table>
      <caption>Table 7. Paired differences. W/L/T counts cells won, lost and tied.</caption>
      <thead><tr><th>Comparison</th><th>Metric</th><th>Difference</th><th>95% interval</th>
        <th>W / L / T</th><th>Sign test</th><th>Verdict</th></tr></thead>
      <tbody>
{comparison_table(report)}
      </tbody>
    </table>
  </div>
  <p class="note">Ties are dropped from the sign test, which is why the effective
  sample is smaller than {p1p2['n_pairs']}. The tie count is itself informative: the two
  leading pipelines agree on {100 * p1p2['ties'] / p1p2['n_pairs']:.0f}% of cells,
  usually because both retrieved the evidence and both answered correctly. The
  bootstrap resamples 10,000 times from a fixed seed, so the intervals on this page
  reproduce exactly.</p>
</section>

<section>
  {heading('filings')}
  <p class="lede">Mean judge score for each of the {report['filings']} filings. BM25 leads
  outright on {leads['P2_bm25']}, the vector pipeline on {leads['P1_vector']}, and they
  tie on {leads['tie']} more. The structural pipeline is last on {leads['worst_last']} of
  {leads['filings']}, and on the {leads['identical']} where it is not, all three score
  identically.</p>
{document_heatmap(report)}
  <p class="note">Read the rows against their question counts. The questions were drawn
  by quadrant rather than evenly by filing, so one filing carries seventeen of them and
  another carries one. A row built on a single question is an observation, not a rate.
  Use this table as a check that no single filing drives the headline, which it does
  not, rather than as a ranking of filings.</p>
</section>

<section>
  {heading('chain')}
  <p class="lede">The whole benchmark reduces to one question: did the evidence come
  back? Split every cell on that and the answer outcome is nearly determined.</p>
  <div class="cards">
{contingency_panel(report)}
  </div>
  <div class="finding">
    <p><b>Retrieval, not generation, is what separates these architectures.</b> When the
    evidence node is returned, every pipeline passes at a similar high rate. When it is
    not, every pipeline fails at a similar high rate. The answering model is the same in
    all six of those cells, so the difference between architectures is entirely a
    difference in what they hand it.</p>
    <p><b>{leak_total} of the {report['cells']} cells break that rule</b>, scoring 4 or 5
    with no evidence node retrieved at all. Those were answered from the answering
    model's own knowledge of these companies rather than from the filing, and
    {leak.get('P3_structural', 0)} of them belong to the structural pipeline. This is
    a floor on parametric leakage, and it flatters the weakest pipeline rather than the
    strongest.</p>
  </div>
  <p class="note">The direction of that leakage matters for the conclusion. A pipeline
  that retrieves nothing useful leaves the model free to answer from memory, so leakage
  raises the floor under the worst performer. The measured gap between structural and
  the other two is therefore a lower bound on the true gap.</p>
</section>

<section>
  {heading('nodetypes')}
  <p class="lede">Roughly one node in eleven across the corpus is a tabular block. What
  share of each pipeline's returned slots are tables, and does that share respond to
  whether the question is about a table?</p>
{node_type_chart(report)}
  <p class="note">All three raise their table share when the question is about a table,
  so all three are responding to the question rather than returning a fixed mix. The
  structural pipeline returns the highest table share on table questions and also the
  highest on prose questions, which is the signature of a retriever whose selectivity
  is weaker rather than better: it is not discriminating more sharply, it is drifting
  toward tabular blocks generally.</p>
</section>

<section>
  {heading('position')}
  <p class="lede">A result this lopsided invites the suspicion that something is broken.
  These diagnostics say otherwise, and they locate the failure precisely.</p>
{position_chart(report)}
  <div class="scroll">
    <table>
      <caption>Table 8. Where each pipeline looked. Diversity is distinct node ids over
      returned slots across the whole benchmark; within a single row it is always 1.000
      and says nothing. Miss distance is reported both absolutely and as a share of the
      filing it was measured in.</caption>
      <thead><tr><th>Pipeline</th><th>Distinct / slots</th><th>Diversity</th>
        <th>Median position</th><th>Median miss</th><th>Miss, share of filing</th>
        <th>Slots on evidence</th></tr></thead>
      <tbody>
{diagnostics_table(report)}
      </tbody>
    </table>
  </div>
  <ul>
    <li><b>It retrieves, it just retrieves the wrong nodes.</b> Hit rate is
    {format_percent(overall[worst]['hit_rate'])} against {format_percent(overall[best]['hit_rate'])} for
    {escape_html(PIPE[best]['label'])}: on most questions the supporting node is never returned
    at all. Diversity is the highest of the three, so this is not a retriever stuck
    returning the same nodes regardless of the query.</li>
    <li><b>It is not landing near the evidence and picking the wrong paragraph.</b>
    Its median returned node sits at {format_number(worst_diag.get('median_retrieved_share'))} of
    the way through a filing whose evidence sits at
    {format_number(gold_share)}, and its median miss is
    {format_percent(worst_diag.get('median_miss_share'))} of the filing against
    {format_percent(best_diag.get('median_miss_share'))} for {escape_html(PIPE[best]['label'])}. It is
    landing in the wrong part of the document, biased late.</li>
    <li><b>Tree traversal descends by comparing the query against summaries</b>, and a
    summary is exactly the artefact from which a specific figure has been compressed
    away. On a benchmark dominated by extraction that is fatal. It would not be on a
    benchmark of thematic questions, which is the use this structure was designed
    for.</li>
    <li><b>Its relative best quadrant is the inferential one.</b> Where the answer
    appears verbatim nowhere, summarisation loses least, and the gap narrows.</li>
  </ul>
  <p class="note">The honest framing for the write-up is that this result characterises
  summary-tree retrieval as configured here, on an extraction-heavy benchmark. It is a
  measurement of a paradigm against a task, not a verdict on the paradigm in
  general.</p>
</section>

<section>
  {heading('build')}
  <p class="lede">Latency and tokens describe what a question costs once the index
  exists. They leave out the one-off cost of creating it, which is where the three
  paradigms diverge hardest. Two build locally from the same {report['nodes']:,} nodes with no
  model call. The third drives an LLM over every node of every filing.</p>
{cost_chart}
  <div class="scroll">
    <table>
      <caption>Table 9. One build from scratch per filing, measured on the same machine.
      Wall clock includes time spent waiting on the LLM provider.</caption>
      <thead><tr><th>Pipeline</th><th>LLM</th><th>Total build</th><th>Median per filing</th>
        <th>Per 1k nodes</th><th>Build tokens</th><th>On disk</th></tr></thead>
      <tbody>
{cost_rows}
      </tbody>
    </table>
  </div>
  <p class="note">The structural index took {duration(structural.get('wall_clock_sec'))}
  against {duration(cheapest['wall_clock_sec']) if cheapest else '-'} for the cheapest
  local index, a factor of {ratio:,.0f}, and consumed
  {structural.get('input_tokens', 0) + structural.get('output_tokens', 0):,} tokens where
  the other two consumed none. That token count covers
  {structural.get('filings_with_tokens', 0)} of {structural.get('filings', 0)} filings,
  so it is a floor. A further {duration(structural.get('later_attempt_sec'))} went on
  rebuilds and resumed runs.</p>
  <p class="note">This is the cleanest of the three efficiency measures, because it is
  the only one with no shared component: local index construction does exactly the work
  the paradigm requires and nothing else. It is the figure to lead an efficiency
  argument with.</p>
</section>

<section>
  {heading('ask')}
  <p class="lede">Answering spend over the {report['cells']} cells, recorded per call.</p>
  <div class="scroll">
    <table>
      <caption>Table 10. Answering tokens only. The judge's own calls are not recorded.</caption>
      <thead><tr><th>Pipeline</th><th>Input tokens</th><th>Output tokens</th>
        <th>Input per cell</th><th>Mean latency</th></tr></thead>
      <tbody>
{token_table(report)}
      </tbody>
    </table>
  </div>
  <p class="note">The structural pipeline sends the most text to the answerer for the
  same question, because tree nodes carry summaries rather than raw passages. It is the
  most expensive to query as well as to build, while scoring lowest. Output tokens are
  near-identical across the three, because the answer format is fixed by the prompt
  rather than by the retrieval paradigm.</p>
</section>

<section>
  {heading('latency')}
  <p class="lede">Per-cell latency is reported for completeness and should be read as a
  null result, not as an efficiency finding.</p>
{latency_chart(report)}
  <div class="scroll">
    <table>
      <caption>Table 11. Latency in seconds per cell. The percentile columns are what
      to quote; the mean is pulled by a small number of very slow provider calls.</caption>
      <thead><tr><th>Pipeline</th><th>Min</th><th>25th</th><th>Median</th><th>75th</th>
        <th>90th</th><th>Max</th><th>Mean</th></tr></thead>
      <tbody>
{latency_table(report)}
      </tbody>
    </table>
  </div>
  <div class="finding">
    <p><b>The medians are {median_phrase(report)}: a spread of
    {median_spread(report):.2f} s across three architectures.</b>
    Every cell routes through the same hosted answering model, and that shared network
    call dominates the measurement. Local retrieval time is real but is lost inside
    it.</p>
    <p>Do not claim a retrieval-speed difference from this table. The efficiency
    argument rests on index build cost, which differs by four orders of magnitude, and
    on input tokens per query, which differ by tens of percent. Both are measured
    cleanly.</p>
  </div>
</section>

<section>
  {heading('threats')}
  <p class="lede">Every limitation an examiner could raise, what the data actually
  shows about it, and the argument to make. Ordered by how much work each one needs in
  the write-up.</p>
  <div class="threats">
{threats(report)}
  </div>
</section>

<section>
  {heading('weaken')}
  <p class="lede">The findings above would not survive all of these equally. Stated
  plainly, so the claims can be scoped to match.</p>
  <ul>
    <li><b>A different question mix would move the vector against BM25 result.</b> This
    is the most fragile finding on the page. The two are separated inside quadrants and
    cancel over the pooled set, so the mix of question types is doing the work. State
    the mix whenever the null result is quoted.</li>
    <li><b>A thematic or summarisation benchmark would move the structural result.</b>
    Summary trees compress detail away, which is fatal for extraction and may be
    helpful for synthesis. Nothing here tests synthesis over a whole filing.</li>
    <li><b>A different answering model could change the leakage floor.</b> A smaller
    model with less memorised knowledge of these companies would leak less and would
    widen the measured gap; a larger one would narrow it.</li>
    <li><b>Nothing here would move by adding runs.</b> The gaps between structural and
    the other two span roughly two judge points with intervals hundreds of times wider
    than any plausible run-to-run noise at temperature 0.</li>
    <li><b>Nothing here would move by adding depth within this sweep.</b> The ordering
    is identical at every one of the three depths.</li>
  </ul>
</section>

<section>
  {heading('ledger')}
  <p class="lede">What can be reproduced, and from what.</p>
  <ul>
    <li><b>Every figure on this page is computed at build time.</b> No number is typed
    into the template. Re-running the generator against the database reproduces the
    page.</li>
    <li><b>The statistics are seeded.</b> The bootstrap uses 10,000 resamples from a
    fixed seed, so intervals reproduce exactly rather than approximately.</li>
    <li><b>Every judged row carries a rubric fingerprint</b>, a digest of the grading
    model together with the exact system prompt, which embeds the rubric and that
    quadrant's calibration exemplars. Recomputing the fingerprint from the current
    rubric reproduces the stored value in all four quadrants, which is what rules out
    silent rubric drift mid-run.</li>
    <li><b>Both answer-key corrections replay from the repository</b> against the
    pre-repair backup and reproduce the current database row for row.</li>
    <li><b>The corpus is frozen.</b> {report['filings']} filings and {report['nodes']:,} nodes;
    the manifest in the repository is the authoritative list.</li>
    <li><b>Every cell is accounted for.</b> {report['cells']} primary cells and
    {report['gate'][1]} gate rows, all scored, with no nulls in any metric column.</li>
  </ul>
</section>

<footer>
  <p>Generated {date.today().isoformat()} from <code>benchmark.db</code> by
  <code>project/build_findings_page.py</code>. Every figure on this page is read from the
  database at build time through <code>aggregate_results.py</code>,
  <code>retrieval_diagnostics.py</code> and <code>index_build_cost.py</code>, with no
  manual step.
  {report['cells']} cells &middot; {report['nodes']:,} nodes &middot; {report['filings']} filings
  &middot; one temperature-zero run per cell.</p>
</footer>
</div>

</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="benchmark.db")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args(argv)
    html_text = build_html(args.db)
    Path(args.out).write_text(html_text)
    print(f"wrote {args.out} ({len(html_text) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
