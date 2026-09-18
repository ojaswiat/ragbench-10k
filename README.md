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

Retrieval and indexing run locally on CPU. Only generation, dataset creation and judging call out to a model, all on free tiers.

```bash
# Ingest filings and build the node store
uv run python -m ingest.run_ingestion

# Build the three indexes
uv run python -m pipelines.vector.build_vector_index
uv run python -m pipelines.bm25.build_bm25_index
uv run python -m pipelines.structural.build_summary_index

# Generate and verify the query set
uv run python -m dataset_generation.run_dataset_generation

# Run the benchmark, then aggregate
uv run python -m benchmark.loop_executor
uv run python -m analysis.aggregate_results
uv run python -m reporting.build_report
```

Set `LOCAL_TEST_THROTTLE=true` in `.env` to cap every loop at three items. Run every stage that way first. The `results` table has a uniqueness constraint on `(source_set, query_id, pipeline, k_value)`, so an interrupted run resumes from the last committed row rather than re-spending quota.

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
