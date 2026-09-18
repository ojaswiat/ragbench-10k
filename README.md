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
demo/                Zero-key side-by-side harness, plus the node store and
                     summary tree it ships for its one filing
tests/               560 tests
```

## Running it

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Run every command from the repository root: `benchmark.db`, `storage/` and `artifacts/` are all resolved relative to the working directory.

```bash
uv sync
```

### See it working, in about two minutes, with no API key

```bash
uv run python -m demo.main
```

This is the fastest way to see what the project actually does. It takes one filing, Microsoft's FY2023 10-K, picks one query from each of the four difficulty quadrants, and sends each query through all three retrieval pipelines at K = 3. For every query it prints the ranked evidence each pipeline returned, marks which nodes are in the ground truth, and scores precision@3 and recall@3 against the stored citations.

**It makes no API calls and reads no key.** All three retrievers run locally: P1 embeds on CPU, P2 is pure BM25 arithmetic, and P3 walks its tree with a MockLLM bound in place of a real one. Precision and recall here are set arithmetic against citations already stored with the query, not a model's judgement.

Two of the demo's inputs cannot be regenerated offline, so they ship with the repository in `demo/assets/` (2.3 MB total):

- The node store for that one filing, which is the output of a paid parse of the source document.
- Its P3 summary tree, which was built by driving an LLM over every node.

The demo copies both into place on first run and leaves an existing `benchmark.db` untouched. The P1 and P2 indexes are not shipped, because they build locally in seconds from the nodes.

Expect roughly two minutes the first time, most of it downloading the `bge-small` embedder and the reranker and embedding 1,103 nodes. Subsequent runs take about 45 seconds, since both indexes are then cached on disk.

The demo ends by saying plainly that four queries on one filing is an illustration of the mechanism and not a finding. The finding is the 900-run benchmark below.

### Full reproduction, with your own API keys

```bash
cp .env.example .env   # then fill in your keys
```

| Variable | Used for | Where to get one |
|---|---|---|
| `GROQ_API_KEY` | Pipeline answerer and the judge | [console.groq.com](https://console.groq.com/keys) |
| `OPENROUTER_API_KEY` | Query generator and critic | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `NIM_API_KEY` | The P3 summary tree build | [build.nvidia.com](https://build.nvidia.com) |
| `LLAMA_CLOUD_API_KEY` | Parsing the filings | [cloud.llamaindex.ai](https://cloud.llamaindex.ai/api-key) |
| `SEC_EDGAR_USER_AGENT` | Not a key: the contact string EDGAR requires | Your own `"name email"` |

**`LOCAL_TEST_THROTTLE` defaults to `true`.** While it is on, every stage caps at three items: three filings ingested and indexed, three sections generated, three benchmark cells, three judge rows. That is the right way to check the plumbing end to end for a few pennies of quota, and it is what you want on a first pass. It is not a run of the benchmark. Set it to `false` in `.env` for the real thing.

```bash
# 1. Fetch and parse the filings into the node store. Creates benchmark.db
#    and its five tables on first run.
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

Two ordering constraints are easy to miss. The judge gate at step 4 exists to be cleared before step 5 starts, because step 5 is what spends the quota. And `build_figures` must run before `build_report`, which embeds whatever `figure-5-*.png` it finds in `artifacts/figures` rather than rendering them itself.

Step 5 is resumable. The `results` table is unique on `(source_set, query_id, pipeline, k_value)`, so an interrupted run reads back what it already committed and skips those cells rather than paying for them twice.

Be aware that a full reproduction will not reproduce the exact numbers in this README. The query set is model-generated, so a fresh run produces a different 140 queries, and the scores move with them. What should reproduce is the shape of the result: P1 and P2 close enough to be indistinguishable, P3 clearly behind on the table-sourced quadrants.

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

Generated artefacts are excluded to keep the repository small: the raw and parsed filings under `data/`, the built indexes under `storage/`, run logs, and `benchmark.db` itself. All of them are reproducible from the commands above, and `data/filings_manifest.json` records exactly which filings the corpus is pinned to.

The one deliberate exception is `demo/assets/`, which carries the node store and P3 summary tree for a single filing, 2.3 MB in total. Those two cannot be regenerated without a paid parse and a run of LLM calls, and without them the demo cannot run at all, which would leave nothing in the repository that a reader can execute.
