# RAGBench-10K

A controlled comparison of three ways of retrieving evidence for a question, tested on real SEC 10-K annual reports.

The question is deliberately narrow. **When the source material is dense financial prose and tables, does the choice of retrieval strategy change the quality of the answer, and where does each strategy break down?**

Generation is held fixed across all three pipelines. The same model answers every question with the same settings, so any difference in score comes from retrieval and nothing else.

---

## The three pipelines

| ID | Paradigm | How it retrieves |
|---|---|---|
| `P1_vector` | Semantic | `bge-small-en-v1.5` embeddings in ChromaDB, then a local cross-encoder reranker |
| `P2_bm25` | Statistical | Okapi BM25 with a custom tokeniser that keeps numbers, decimals, percentages and currency |
| `P3_structural` | Structural | Walks a per-filing summary tree from root to leaf |

- P2 has no model anywhere in its retrieval path, which keeps the semantic versus statistical contrast clean.
- P1 gets no metadata filter and no section head start, so it cannot narrow the search before it begins.
- P3 reuses a summary tree built once per filing, never rebuilt at query time.

## The corpus and the question set

- **13 SEC 10-K filings**, five companies at uneven year depth: AAPL, MSFT and TSLA for FY2023 to FY2025, JPM and JNJ for FY2023 to FY2024.
- Parsed into **18,297 nodes**, with tables kept whole rather than split across chunks.
- **140 questions**, generated from the filings and then verified, split into three sets that never overlap:
  - 100 primary questions, the ones actually scored.
  - 20 golden questions, hand-labelled, used only to teach the judge.
  - 20 judge-validation questions, used only to test the judge.
- Questions sit in four quadrants, crossing factual against analytical and text-sourced against table-sourced.
- The full run is **900 cells**: 100 questions x 3 pipelines x 3 values of K, each run once at temperature 0.

## How it is scored

Three pillars:

1. **Retrieval**: precision@K, recall@K, evidence hit rate.
2. **Answer quality**: a judge model scoring 1 to 5, backed by citation auditing, token-F1 and numeric exact match done in code.
3. **Efficiency**: latency, tokens spent, and what it cost to build each index in the first place.

The judge has to earn its place before it grades anything. It scores 60 validation outputs against independently produced reference scores and must agree on more than 80% of them. It cleared at **86.7%**, 52 of 60. It sees only the five exemplars matching the question's quadrant, never all 20, and it has no search tool.

## Headline result

- **P1 and P2 cannot be told apart.** The paired difference is -0.160 judge points with a 95% interval of [-0.340, +0.013], which crosses zero. A well-tokenised BM25 index matches a dense retriever with a reranker on this material.
- **P3 trails both by roughly two judge points**, with intervals nowhere near zero. Summarising up the tree loses the cell values and footnotes that table questions depend on.

---

# Part 1: Run the demo

**Start here.** It shows you what the project does in about two minutes, and it needs no API key at all.

### What you need

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- Nothing else. No account, no key, no `.env` file.

### Steps

```bash
git clone git@github.com:ojaswiat/ragbench-10k.git ragbench-10k
cd ragbench-10k
uv sync
uv run python -m demo.main
```

### What it does

- Takes one filing, Microsoft's FY2023 10-K, which is 1,103 nodes.
- Picks one question from each of the four quadrants, so four questions in total.
- Sends each question through all three pipelines at K = 3.
- Prints the ranked evidence each pipeline returned, and marks which nodes were correct.
- Scores precision@3 and recall@3 against the citations already stored with the question.
- Finishes with a summary table and a plain explanation of how each pipeline succeeded or failed.

### What it costs

- **No API calls, and no key is read.** P1 embeds on your CPU, P2 is pure arithmetic, and P3 walks its tree with a mock model standing in for a real one.
- The scores are set arithmetic against stored citations, not a model's opinion.

### How long it takes

- **First run: about two minutes.** Most of that is downloading the embedding model and the reranker, then embedding 1,103 nodes.
- **Later runs: about 45 seconds.** Both indexes are cached on disk by then.

### What you should see

For each of the four questions the run prints the ground-truth node, then the three nodes each pipeline returned, with a `*` marking the ones that were correct. It ends with this summary:

```
==============================================================================
  SUMMARY   ·   mean over 4 queries at K=3
==============================================================================
                      Precision@3    Recall@3    Mean latency
                   ------------------------------------------
  P1  semantic               0.42        0.69           4.94s
  P2  statistical            0.50        0.75           0.02s
  P3  structural             0.00        0.00           3.88s
```

Reading the columns:

- **Precision@3**: of the three nodes a pipeline returned, the fraction that were in the ground truth. Higher is better.
- **Recall@3**: of the nodes the ground truth lists, the fraction the pipeline found. Higher is better.
- **Mean latency**: average time to retrieve the evidence for one question.

What it tells you:

- P1 and P2 land close together, which is the headline result in miniature.
- P3 misses on all four questions, so it scores zero on both measures.
- P2 is two orders of magnitude faster than P1, because it runs no model and does no embedding.
- Your latency figures will differ, since they depend on your machine. The pattern should not.

### Why two files ship with the repository

The demo needs a node store and a P3 summary tree, and neither can be made offline:

- The node store comes out of a paid parse of the source filing.
- The summary tree is built by running a model over every node.

Both are included in `demo/assets/` for that one filing, 2.3 MB in total, and copied into place on the first run. If you already have a real `benchmark.db`, the demo leaves it alone. The P1 and P2 indexes are not shipped, because they build locally in seconds.

### One honest caveat

Four questions on one filing shows you the mechanism. It is not a finding. The finding is the 900-run benchmark in Part 2.

---

# Part 2: Run the full project

This rebuilds the whole thing from nothing. It needs your own API keys, and it takes days rather than minutes.

### What you need

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

| Variable | What it is for | Where to get one |
|---|---|---|
| `LLAMA_CLOUD_API_KEY` | Parsing the filings | [cloud.llamaindex.ai](https://cloud.llamaindex.ai/api-key) |
| `NIM_API_KEY` | Building the P3 summary trees | [build.nvidia.com](https://build.nvidia.com) |
| `OPENROUTER_API_KEY` | Generating and checking the questions | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `GROQ_API_KEY` | Answering questions and judging answers | [console.groq.com](https://console.groq.com/keys) |
| `SEC_EDGAR_USER_AGENT` | Not a key. The contact string EDGAR asks for | Your own `"name email"` |

All four services have a free tier. The project is built to stay inside them.

### Read this before you start

- **`LOCAL_TEST_THROTTLE` defaults to `true`.** While it is on, every stage stops after three items: three filings, three sections, three benchmark cells, three judge rows.
- Leave it on for your first pass. It proves the whole chain works for a few pennies of quota.
- Set it to `false` in `.env` when you want the real run.
- Run every command from the repository root. Paths are worked out from your working directory.
- Steps 2 and 6 are the expensive ones. Everything else is cheap or free.

**If you ran the demo first, clear its state before you start.** The demo seeds `benchmark.db` with the nodes and 16 questions for its one filing, and ingestion skips any filing that already has nodes. Left in place, those 16 would be counted towards the question set and you would end up with a mix of bundled and freshly generated questions.

```bash
rm -rf benchmark.db benchmark.db-wal benchmark.db-shm storage artifacts
```

Delete both or neither. Keeping `storage/` while deleting the database leaves a summary tree pointing at node IDs that no longer exist. Nothing here is precious: the demo rebuilds its own state the next time you run it.

---

### Step 1. Download and parse the filings

```bash
uv run python -m ingest.run_ingestion
```

- Downloads 13 10-K filings from SEC EDGAR against a pinned manifest, so you get the same documents every time.
- Sends each one to LlamaParse in atomic-table mode, so a table is never cut in half.
- Builds nodes and writes them to SQLite, creating `benchmark.db` and its five tables on the first run.
- Gives every node a stable ID, its parent section heading, a text or table type, and a page number for display only.
- **Produces**: 18,297 nodes, about 90 MB of raw filings and 10 MB of parsed markdown.
- **Time**: varies with the LlamaParse queue. This is the slowest stage that does not involve an LLM.
- Parsed output is cached, so running it again does not pay to parse twice.

### Step 2. Build the three indexes

```bash
uv run python -m pipelines.vector.build_vector_index
uv run python -m pipelines.bm25.build_bm25_index
uv run python -m pipelines.structural.build_summary_index
```

These are very different jobs, and the difference is itself one of the results. Measured on the full 13-filing corpus:

| Index | Time | Tokens | On disk | Uses a model |
|---|---|---|---|---|
| P1 vector | 22 minutes | 0 | 129 MB | No, local CPU |
| P2 BM25 | under 1 second | 0 | 7 MB | No |
| P3 summary tree | **10.6 hours** | 1.6M in, 892K out | 27 MB | Yes |

- P3 is the only index allowed to call a model, and it runs once and is cached from then on.
- It runs on NVIDIA NIM's free tier, which caps requests rather than tokens, so this bulk job does not eat the tighter daily token budget.
- Expect to leave it running overnight. It is the single longest stage in the project.

### Step 3. Generate and check the question set

```bash
uv run python -m dataset_generation.run_dataset_generation
```

- A generator model reads one section at a time and writes questions, answers, and the node IDs the answer comes from.
- A critic model from a **different family** then checks each one, blind. The proposed answer and citations are hidden from it, and it searches the whole filing to find the evidence itself.
- The two are compared **in code**, on node IDs and values. Anything that does not match is thrown away and regenerated.
- The survivors are split into the three sets that never overlap: 100 primary, 20 golden, 20 judge-validation.
- **Produces**: 140 verified questions in `benchmark.db`.

### Step 4. Teach the judge

```bash
uv run python -m dataset_generation.write_gq_labels
```

- Applies the calibration labels to the 20 golden questions.
- These are the only thing telling the judge what each score means, so the set deliberately spans the range and includes a poor answer for every quadrant.
- **Produces**: a labelled golden set. Without this the judge has no exemplars and cannot run.

### Step 5. The judge gate

```bash
uv run python -m benchmark.validation_gate     # 60 outputs, judge-scored
uv run python -m gate.gate_reference_scores    # independent reference scores
uv run python -m gate.score_gate_outputs       # the agreement check
```

- Runs the 20 validation questions through all three pipelines at K = 5, giving 60 outputs.
- Scores them twice, once by the judge and once independently, with neither seeing the other.
- Reports the agreement rate. **It must be above 80%.**
- If it fails, change the rubric or swap the judge model, then run it again.
- **This gate exists to be cleared before step 6**, because step 6 is what spends the quota. Do not skip it.

### Step 6. The benchmark itself

```bash
uv run python -m benchmark.loop_executor
```

- Runs all 900 cells: 100 questions x 3 pipelines x K in {2, 3, 5}.
- Each cell runs once at temperature 0. No repeated sampling.
- Every answer is scored by the judge and by the code metrics.
- **Time**: two to three weeks of intermittent running on strict free tiers, because it is limited by daily token caps rather than by speed.
- **It is safe to stop and restart.** The results table is unique on `(source_set, query_id, pipeline, k_value)`, so it reads back what it already saved and skips those cells. An interruption never costs you quota twice.

### Step 7. Aggregate and report

```bash
uv run python -m analysis.aggregate_results
uv run python -m reporting.build_figures
uv run python -m reporting.build_report
```

- `aggregate_results` prints the comparison to your terminal: overall scores, a breakdown by K, a breakdown by quadrant, and paired comparisons between pipelines with confidence intervals.
- `build_figures` writes six charts to `artifacts/figures/`: answer quality, retrieval quality, the effect of K, the quadrant breakdown, the score distribution, and the paired comparisons.
- `build_report` writes `artifacts/results-report.html`, one self-contained file with the figures embedded in it.
- **Order matters here.** `build_figures` must run before `build_report`, because the report embeds the images it finds on disk rather than drawing them itself.

### What you will and will not reproduce

- You **will not** get the exact numbers in this README. The question set is model-generated, so a fresh run writes a different 140 questions and the scores move with them.
- You **should** get the same shape: P1 and P2 close enough to be indistinguishable, and P3 clearly behind on the table-sourced quadrants.

### Optional extras

Hand-write the golden labels instead of using the committed set:

```bash
uv run python -m dataset_generation.gq_label_export   # writes a template to fill in
uv run python -m dataset_generation.gq_label_import   # reads it back
```

Re-measure the index build costs rather than using the committed snapshot:

```bash
uv run python -m analysis.index_build_cost
```

---

## Tests

```bash
uv run pytest -q
```

- A clean checkout gives **552 passed, 8 skipped**.
- Six of the skips are calibration guards on the golden set. They have nothing to check until step 4 has run.
- The other two make real network calls and stay skipped unless you set `RUN_LIVE_GROQ_TESTS=1`.

## Repository layout

```
core/                SQLite state layer in WAL mode, shared throttle helpers,
                     and the durable embedding-model cache
llm_client/          One async client per provider behind a single factory,
                     with backoff, jitter and a concurrency cap
ingest/              EDGAR download, parsing, node building, parse audit
dataset_generation/  Generator, critic with its search tool, the code
                     cross-check, and the golden-label tools
pipelines/           Base retriever, the shared answerer, and the vector,
                     bm25 and structural implementations with their builders
judge/               Judge prompt assembly, deterministic metrics, numeric
                     normalisation
benchmark/           Runs the 900-cell matrix, held behind the judge gate
gate/                Scores the 60 gate outputs and reports agreement
analysis/            Aggregation, quadrant breakdown, retrieval diagnostics,
                     evidence localisation, index build cost
reporting/           The six figures, the HTML report, the findings page
demo/                The zero-key demo, plus the node store and summary tree
                     it ships for its one filing
scripts/             Backup, reload, answer-key repair
tests/               560 tests
```

## Model routing

Roles are pinned so that no model ever marks its own family's work.

| Role | Model | Provider |
|---|---|---|
| Question generator | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter |
| Question critic | `openai/gpt-oss-20b:free` | OpenRouter |
| P3 index build | `nvidia/nemotron-3-super-120b-a12b` | NVIDIA NIM |
| Pipeline answerer | `llama-3.3-70b-versatile` | Groq |
| Judge | `qwen/qwen3.6-27b` | Groq |

The generator and the critic are different families. The answerer and the judge are different families. That separation is what stops the evaluation marking its own homework.

## What is not in this repository

Generated files are left out to keep the repository small:

- The raw and parsed filings under `data/`
- The built indexes under `storage/`
- Run logs
- `benchmark.db` itself

All of them are rebuilt by the steps in Part 2, and `data/filings_manifest.json` pins exactly which filings the corpus uses.

The one deliberate exception is `demo/assets/`, 2.3 MB holding the node store and summary tree for a single filing. Neither can be regenerated without a paid parse and a long run of model calls, and without them there would be nothing in this repository that a reader could actually run.
