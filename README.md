# RAGBench-10K

A controlled comparison of three retrieval paradigms for retrieval-augmented generation, evaluated on SEC 10-K annual reports.

The question is narrow on purpose: **when the corpus is dense financial prose and tables, does the retrieval strategy actually change answer quality, and where does each strategy fail?** Generation is held constant across all three pipelines so that any measured difference is attributable to retrieval alone.

## The three pipelines

| ID | Paradigm | Retrieval |
|---|---|---|
| `P1_vector` | Semantic | `bge-small-en-v1.5` embeddings in ChromaDB, with a local cross-encoder reranker |
| `P2_bm25` | Statistical | Okapi BM25 (`rank_bm25`) with a custom tokenizer that preserves numbers, decimals, percentages and currency |
| `P3_structural` | Structural | Traversal of a cached per-filing LlamaIndex `TreeIndex` of parent summaries over child nodes |

P2 is kept purely statistical, with no LLM anywhere in the retrieval path, so the semantic-versus-statistical contrast stays clean. P1 gets no metadata pre-filter and no section head start.

## Corpus and benchmark

- **13 SEC 10-K filings** across 5 companies at uneven fiscal-year depth: AAPL, MSFT and TSLA for FY2023 to FY2025; JPM and JNJ for FY2023 to FY2024.
- Parsed into **18,297 nodes**, with tables preserved as intact, indivisible nodes.
- **140 generated and verified queries**, split into three disjoint sets: 100 primary (PQ), 20 golden (GQ, hand-labelled, used only as judge exemplars), 20 judge-validation (JEQ, used only for the gate).
- Queries span four quadrants crossing factual against analytical and text-sourced against table-sourced.
- The full matrix is **900 cells**: 100 PQ x 3 pipelines x K in {2, 3, 5}, each run once at `temperature = 0`.

Queries are produced by a generator model and then independently verified by a critic model from a different family, which searches the whole filing blind to the proposed answer and citations. Acceptance is a deterministic code check on node IDs and values, never a model's judgement.

## Evaluation

Scoring is tri-pillar:

1. **Retrieval**: Precision@K, Recall@K, evidence hit rate.
2. **Answer quality**: an LLM judge on a 1 to 5 scale, backed by deterministic citation auditing, token-F1 and numeric-normalised exact match.
3. **Efficiency**: latency, token spend, and one-time index build cost.

The judge is gated before it is trusted. It scores the 60 JEQ outputs (20 queries x 3 pipelines at K = 5) against independently produced reference scores, and must clear 80% concordance before the full run is permitted. It cleared at **86.7% (52 of 60)**. The judge sees only the five quadrant-matched exemplars for the item in front of it, never all 20, and has no search tool.

## Headline result

- **P1_vector and P2_bm25 are statistically indistinguishable.** Paired difference -0.160 judge points, 95% CI [-0.340, +0.013], which straddles zero. On this corpus, a well-tokenised BM25 index matches a dense retriever with a reranker.
- **P3_structural trails both by roughly two judge points**, with confidence intervals far from zero. Summarisation up the tree drops the cell values and footnotes that table-sourced questions depend on.

## Layout

```
core/                SQLite state layer (nodes, queries, golden_queries,
                     judge_validation, results) in WAL mode, the shared
                     throttle helpers, and the durable embedding-model cache
llm_client/          Per-provider async clients behind one LLMFactory, with
                     backoff, jitter and a concurrency semaphore
ingest/              EDGAR fetch, parsing, node construction, parse audit
dataset_generation/  Generator, critic with search tool, cross-check, and the
                     golden-query labelling tools
pipelines/           Base retriever, shared answerer, and vector/ bm25/
                     structural/ implementations plus their index builders
judge/               Judge prompt assembly, deterministic metrics, numeric
                     normalisation
benchmark/           Runs the 900-cell matrix, committing per row, held behind
                     the judge gate
gate/                Scores the 60 gate outputs and reports concordance
analysis/            Tri-pillar aggregation, per-quadrant breakdown, retrieval
                     diagnostics, evidence localisation, index build cost
reporting/           Figures, the self-contained HTML results report, and the
                     standalone findings page
scripts/             Backup, reload, and answer-key repair
demo/                Small side-by-side comparison harness
tests/               560 tests
```

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in your API keys
```

Run every command from the repository root. Paths such as `benchmark.db`,
`storage/` and `artifacts/` are resolved relative to the working directory.

Retrieval and indexing run locally on CPU. Only dataset generation, answering and judging call out to a model, all on free tiers.

```bash
# 1. Fetch and parse the filings into the node store. This creates
#    benchmark.db and its five tables on first run.
uv run python -m ingest.run_ingestion

# 2. Build the three indexes. The first two are local and free; the third
#    drives an LLM over every node and is the one that costs real quota.
uv run python -m pipelines.vector.build_vector_index
uv run python -m pipelines.bm25.build_bm25_index
uv run python -m pipelines.structural.build_summary_index

# 3. Generate and verify the 140 queries, then apply the judge's
#    calibration labels to the 20 golden ones.
uv run python -m dataset_generation.run_dataset_generation
uv run python -m dataset_generation.write_gq_labels

# 4. The judge gate. Concordance must exceed 80% before the full run is
#    worth starting; below that, retune the rubric or swap the judge model.
uv run python -m benchmark.validation_gate     # 60 gate rows, judge-scored
uv run python -m gate.gate_reference_scores    # independent reference scores
uv run python -m gate.score_gate_outputs       # the concordance check

# 5. The 900-cell matrix. Paced across days against free-tier ceilings,
#    and resumable after an interruption.
uv run python -m benchmark.loop_executor

# 6. Aggregate, then render.
uv run python -m analysis.aggregate_results
uv run python -m reporting.build_figures
uv run python -m reporting.build_report
```

Two ordering constraints are easy to miss. The judge gate at step 4 exists to be passed before step 5 is started, since the full run is what spends the quota. And `build_figures` must run before `build_report`, which embeds whatever `figure-5-*.png` it finds in `artifacts/figures` rather than rendering them itself.

Set `LOCAL_TEST_THROTTLE=true` in `.env` to cap every loop at three items. Run every stage that way first. The `results` table has a uniqueness constraint on `(source_set, query_id, pipeline, k_value)`, so an interrupted run resumes from the last committed row rather than re-spending quota.

### Optional steps

To hand-write the golden-query labels instead of using the committed calibration set, export a template, fill it in, and read it back:

```bash
uv run python -m dataset_generation.gq_label_export   # writes golden_queries_to_label.md
uv run python -m dataset_generation.gq_label_import   # reads it back
```

To regenerate the index build cost snapshot rather than using the committed `data/index_build_costs.json`:

```bash
uv run python -m analysis.index_build_cost
```

That one re-measures P1 and P2 by rebuilding into a scratch directory, and reads P3's cost back from the append-only log its build wrote, since provider queueing cannot be replayed.

### Tests

```bash
uv run pytest -q
```

A clean checkout runs green: 552 pass and 8 skip. The skips are the six
calibration guards, which assert properties of a populated `golden_queries`
table and have nothing to check until dataset generation has run, plus two
tests marked `live`, which make real network calls and stay skipped unless
`RUN_LIVE_GROQ_TESTS=1`.

## Model routing

Roles are pinned so that no model ever grades its own family's output.

| Role | Model | Provider |
|---|---|---|
| Query generator | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter |
| Query critic | `openai/gpt-oss-20b:free` | OpenRouter |
| P3 index build | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA NIM |
| Pipeline answerer | `llama-3.3-70b-versatile` | Groq |
| Judge | `qwen/qwen3.6-27b` | Groq |

Generator and critic are different families, and the answerer and judge are different families. That separation is the anti-self-grading invariant the evaluation rests on.

## Not in this repository

Generated artefacts are excluded to keep the repository small: the raw and parsed filings under `data/`, the built indexes under `storage/`, run logs, and `benchmark.db` itself. All of them are reproducible from the commands above. `data/filings_manifest.json` records exactly which filings the corpus is pinned to.
