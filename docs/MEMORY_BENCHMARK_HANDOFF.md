# Memory benchmark handoff

Last updated: 2026-07-14

This document is the starting point for the next development session. It records the current benchmark infrastructure, the two stored reference runs, the conclusions that are already supported by data, and the next experiments that must be kept isolated from one another.

## Start here

Repository: `Lotargo/chat-openwebui`

Implementation baseline commit summarized by this handoff, before the documentation-only merge:

```text
c419434f8f7f07a5da26b8095e458f997f21d5b6
```

Relevant workflow:

```text
.github/workflows/live-memory-benchmarks.yml
```

Benchmark implementation:

```text
src/py-engine/agentic_circuit/benchmarks/
```

Neon analytics documentation:

```text
docs/BENCHMARK_ANALYTICS.md
```

Database schema migration:

```text
db/migrations/001_ml_eval_registry.sql
```

## Infrastructure status

The live benchmark workflow is operational.

It currently:

- runs LongMemEval-S adapted cases, LoCoMo-10 adapted cases and the internal memory lifecycle suite;
- starts local Qdrant, multilingual E5 and the GTE multilingual reranker in GitHub Actions;
- enforces four deterministic safety cases;
- stores every completed run in Neon under the `ml_eval` schema;
- stores suite aggregates, every individual case and provider telemetry;
- appends deltas against the previous compatible run to the Markdown artifact;
- uploads JSON and Markdown artifacts with 90-day retention;
- serializes expensive runs without cancelling an already running benchmark;
- uses Node.js 24-compatible GitHub Actions versions.

Required repository secrets:

```text
OPENCODE_ZEN_API_KEY
DATABASE_DIRECT_URL
DATABASE_URL
```

`DATABASE_DIRECT_URL` is preferred. `DATABASE_URL` is the fallback.

These keys are not required by the current benchmark path:

```text
LANGSEARCH_API_KEY
QDRANT_API_KEY
```

The benchmark uses local Qdrant and does not call LangSearch.

## Neon registry

Project database: `neondb`

Schema:

```text
ml_eval
```

Objects:

- `benchmark_runs`
- `benchmark_suite_metrics`
- `benchmark_case_results`
- `benchmark_provider_usage`
- `benchmark_suite_deltas`

The registry is already populated with two comparable runs in category:

```text
live-memory-adapted-v1
```

## Stored reference runs

### Run 1: initial baseline

```text
GitHub run: 29304658723
Git commit: 0a1dbffd10b9760aa9268e205dd23ed28989c071
Seed: 20260714
Cases: 6 LongMemEval + 8 LoCoMo + 6 internal
```

### Run 2: second reference run

```text
GitHub run: 29309563088
Git commit: c419434f8f7f07a5da26b8095e458f997f21d5b6
Seed: 20260714
Cases: 6 LongMemEval + 8 LoCoMo + 6 internal
```

### Aggregate comparison

| Suite | Metric | Run 1 | Run 2 |
|---|---:|---:|---:|
| LongMemEval-S | Recall@10 | 1.0000 | 1.0000 |
| LongMemEval-S | MRR@10 | 0.7083 | 0.7083 |
| LongMemEval-S | Token F1 | 0.1468 | 0.0306 |
| LongMemEval-S | Mean latency | 62.95 s | 68.09 s |
| LoCoMo-10 | Recall@10 | 0.7188 | 0.7188 |
| LoCoMo-10 | MRR@10 | 0.3802 | 0.3802 |
| LoCoMo-10 | Token F1 | 0.0506 | 0.0348 |
| LoCoMo-10 | Mean latency | 105.45 s | 94.01 s |
| Internal lifecycle | Passed | 6/6 | 4/6 |
| Internal lifecycle | Mean latency | 14.34 s | 6.14 s |

The four blocking safety checks passed in both runs:

- project isolation;
- conversation isolation;
- supersession;
- unknown abstention.

The two non-blocking LLM-dependent cases failed in Run 2:

- `live gate preference extraction` returned `[]`;
- `live gate knowledge update` returned `supabase neon` instead of a clean superseding memory statement.

## Provider telemetry

Current benchmark model chain:

```text
Primary: big-pickle
Fallback: mimo-v2.5-free
Judge: mimo-v2.5-free
```

Run 1:

| Model | Attempts | Successes | Failures |
|---|---:|---:|---:|
| big-pickle | 40 | 24 | 16 |
| mimo-v2.5-free | 25 | 3 | 22 |

Run 2:

| Model | Attempts | Successes | Failures |
|---|---:|---:|---:|
| big-pickle | 39 | 27 | 12 |
| mimo-v2.5-free | 22 | 2 | 20 |

MiMo did not provide a usable external judge score in either reference run. External `judge_accuracy` remained null.

## Conclusions already supported by data

### Stable layer

The retrieval layer is reproducible across the two runs:

- LongMemEval Recall@10 and MRR@10 were identical;
- LoCoMo Recall@10 and MRR@10 were identical;
- scope isolation and deterministic lifecycle behavior remained intact.

Do not start by replacing embeddings, Qdrant or the reranker. The current evidence does not identify them as the primary source of variance.

### Unstable layer

Most variance appears after raw retrieval:

- LLM memory extraction;
- `MEMORY_SELECT` selection count;
- free-form reader output;
- remote judge behavior.

The same retrieved candidates can produce selection counts ranging from zero to six between runs. This can erase relevant evidence even when raw Recall@10 is correct.

### Metric limitation

Token F1 is strongly affected by translation, inflection and verbose answers. It must not be treated as the sole answer-quality metric.

A neutral factual reader and a reliable semantic judge are required before interpreting answer quality as a single score.

## Decision for the next session: replace MiMo

Replace every benchmark role currently using:

```text
mimo-v2.5-free
```

with:

```text
deepseek-v4-flash-free
```

Unless a later experiment explicitly separates the roles, this replacement applies to both:

- fallback generation;
- external benchmark judge.

Keep `big-pickle` as the primary model for the first comparison after the change.

The first post-change run must use the same inputs:

```text
LongMemEval cases: 6
LoCoMo cases: 8
Seed: 20260714
Category: live-memory-adapted-v1
```

Do not combine this model replacement with fusion, selector, reader or chunking changes. The purpose of the first run is to isolate the provider effect.

Before running, add or verify telemetry for:

- actual model used for every role;
- judge request errors;
- judge response parse errors;
- empty judge responses;
- raw judge response retained only when parsing fails and without secrets;
- fallback reason;
- parameter-compatibility retry reason.

## Fusion hypothesis: RRF versus RSF

The current hybrid retrieval uses weighted Reciprocal Rank Fusion with:

```text
RAG_RRF_K=60
RAG_DENSE_WEIGHT=0.6
RAG_LEXICAL_WEIGHT=0.4
```

Working hypothesis: rank-only RRF may discard useful score magnitude and can behave poorly without tuning for this corpus.

Investigate an RSF-style normalized score fusion as a separate experiment. In this document, RSF means relative or normalized score fusion: dense and lexical scores are converted to comparable scales before applying weights.

Do not replace RRF blindly. Add a configurable fusion strategy and compare both methods on the same fixed subset.

Suggested configuration boundary:

```text
RAG_FUSION_METHOD=rrf|rsf
```

For RSF, test normalization explicitly rather than hiding it inside the implementation. Candidate variants:

1. Per-query min-max normalization with an epsilon for flat lists.
2. Robust percentile or quantile scaling to reduce outlier sensitivity.
3. Z-score normalization with clipping, only when score distributions are sufficiently stable.
4. Rank normalization as a fallback when one source returns unusable score distributions.

Record enough diagnostics to understand the result:

- raw dense score;
- raw lexical score;
- dense rank;
- lexical rank;
- normalized dense score;
- normalized lexical score;
- final fused score;
- normalization method;
- source list size;
- flat-distribution and missing-source flags.

Keep the initial weights at `0.6 / 0.4` for the first RRF-versus-RSF comparison. Weight tuning must be a later experiment.

Important: the two existing runs show stable RRF retrieval metrics. Therefore RSF is an optimization hypothesis, not yet a demonstrated fix for the larger end-to-end instability. Selector and reader failures can dominate even when fusion is unchanged.

## Ordered next experiments

Perform one change family at a time.

### Experiment A: provider replacement

- replace MiMo with `deepseek-v4-flash-free`;
- keep primary model, retrieval, selector, reader and fusion unchanged;
- run the fixed 6/8/20260714 subset;
- compare provider success rate, judge coverage and all existing metrics.

Success criteria:

- external judge returns a parseable result for most cases;
- fallback success rate is materially better than MiMo;
- no regression in the four blocking safety checks.

### Experiment B: observability and metric separation

Add these metrics before changing selector behavior:

- raw retrieval recall;
- selected recall;
- answer correctness;
- selector empty rate on answerable cases;
- judge coverage;
- judge parse-failure rate;
- answer language match;
- concise-answer compliance.

This must expose where evidence is lost:

```text
index -> raw retrieval -> selected context -> reader answer -> judge
```

### Experiment C: neutral benchmark reader

Create a benchmark-only factual reader that:

- answers in the language of the question;
- returns only the requested value or short phrase;
- does not use the Lisa personality;
- does not add recommendations;
- abstains only when the selected context truly lacks the answer.

Keep the production conversational reader separate.

### Experiment D: selector stabilization

Investigate a deterministic fallback for answerable retrieval results.

Possible rule:

- when the LLM selector returns zero items but raw retrieval contains candidates above an evidence threshold, pass a small top-N fallback set to the reader;
- retain zero selection for real abstention cases;
- measure the effect with `selected_recall` and unsupported-context rate.

Do not simply force top-1 for every query. That would damage abstention behavior.

### Experiment E: RRF versus normalized RSF

- add a fusion strategy flag;
- keep datasets, seed, models, selector and reader fixed;
- compare RRF and each normalization candidate;
- evaluate Recall@10, MRR@10, selected recall and downstream answer correctness;
- inspect per-category results, not only global means.

### Experiment F: long-session chunking

Current memory text is truncated by `RAG_MAX_MEMORY_CHARS=6000` before indexing.

The preferred fix is chunking with a stable parent identity, for example:

```text
parent_session_id
chunk_id
chunk_index
```

Do not solve this only by setting a very large character limit. Chunking should preserve session grouping while allowing the relevant part of a long session to be retrieved independently.

## Experiment discipline

For every comparison:

- use the same seed and case counts;
- change one subsystem at a time;
- write a new category only when the benchmark semantics or dataset composition changes;
- retain all failed cases in Neon;
- inspect case-level deltas, not only averages;
- do not call an adapted subset score an official leaderboard result;
- keep deterministic safety checks blocking;
- keep stochastic extraction cases informational until they are made stable enough to gate CI.

## Immediate first task for the next session

1. Read this file and `docs/BENCHMARK_ANALYTICS.md`.
2. Locate every benchmark reference to `mimo-v2.5-free`.
3. Replace benchmark fallback and judge roles with `deepseek-v4-flash-free`.
4. Improve judge error telemetry before launching the run.
5. Run the same `6 / 8 / 20260714` benchmark.
6. Query Neon and compare against runs `29304658723` and `29309563088`.
7. Only after that start the RRF-versus-RSF branch.
