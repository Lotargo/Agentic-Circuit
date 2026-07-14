# Memory benchmark handoff

Last updated: 2026-07-14

This document is the starting point for the next development session. It records the current benchmark infrastructure, three comparable stored runs, the completed provider experiment and the next isolated experiment.

## Start here

Repository:

```text
Lotargo/chat-openwebui
```

Relevant workflow:

```text
.github/workflows/live-memory-benchmarks.yml
```

Benchmark implementation:

```text
src/py-engine/src/agentic_circuit/benchmarks/
```

Analytics documentation:

```text
docs/BENCHMARK_ANALYTICS.md
```

Completed Experiment A report:

```text
docs/MEMORY_BENCHMARK_EXPERIMENT_A.md
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
- enforces four deterministic blocking safety cases;
- stores every completed run in Neon under the `ml_eval` schema;
- stores suite aggregates, every individual case and provider telemetry;
- appends deltas against the previous compatible run to the Markdown artifact;
- uploads JSON and Markdown artifacts with 90-day retention;
- serializes expensive runs without cancelling an already running benchmark;
- uses Node.js 24-compatible GitHub Actions versions;
- accepts independent manual `fallback_model` and `judge_model` inputs;
- keeps `mimo-v2.5-free` as the scheduled default until another provider passes an isolated comparison.

Provider telemetry now records:

- requested attempts, successes and failures by model;
- request role for `memory_select`, `memory_extract`, `benchmark_reader` and `benchmark_judge`;
- provider-reported model identifier on successful responses;
- fallback reason;
- parameter-compatibility retry reason;
- judge request errors;
- empty judge responses;
- judge parse errors;
- bounded raw judge responses only when parsing fails.

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

Project: `liza-ds`

Database:

```text
neondb
```

Schema:

```text
ml_eval
```

Objects:

- `benchmark_runs`;
- `benchmark_suite_metrics`;
- `benchmark_case_results`;
- `benchmark_provider_usage`;
- `benchmark_suite_deltas`.

Comparable category:

```text
live-memory-adapted-v1
```

## Stored comparable runs

### Run 1: initial baseline

```text
GitHub run: 29304658723
Git commit: 0a1dbffd10b9760aa9268e205dd23ed28989c071
Primary: big-pickle
Fallback: mimo-v2.5-free
Judge: mimo-v2.5-free
Seed: 20260714
Cases: 6 LongMemEval + 8 LoCoMo + 6 internal
```

### Run 2: second reference

```text
GitHub run: 29309563088
Git commit: c419434f8f7f07a5da26b8095e458f997f21d5b6
Primary: big-pickle
Fallback: mimo-v2.5-free
Judge: mimo-v2.5-free
Seed: 20260714
Cases: 6 LongMemEval + 8 LoCoMo + 6 internal
```

### Run 3: Experiment A

```text
GitHub run: 29312255221
Reported benchmark commit: caf1fe7e33d4e991bbf288138d949d77dad0c17c
Primary request: big-pickle
Fallback: deepseek-v4-flash-free
Judge: deepseek-v4-flash-free
Seed: 20260714
Cases: 6 LongMemEval + 8 LoCoMo + 6 internal
```

The Experiment A artifact is named:

```text
live-memory-benchmark-6
```

## Aggregate comparison

| Suite | Metric | Run 1 | Run 2 | Experiment A |
|---|---:|---:|---:|---:|
| LongMemEval-S | Recall@10 | 1.0000 | 1.0000 | 1.0000 |
| LongMemEval-S | MRR@10 | 0.7083 | 0.7083 | 0.7083 |
| LongMemEval-S | Token F1 | 0.1468 | 0.0306 | 0.1187 |
| LongMemEval-S | Mean latency | 62.95 s | 68.09 s | 50.37 s |
| LoCoMo-10 | Recall@10 | 0.7188 | 0.7188 | 0.7188 |
| LoCoMo-10 | MRR@10 | 0.3802 | 0.3802 | 0.3802 |
| LoCoMo-10 | Token F1 | 0.0506 | 0.0348 | 0.0534 |
| LoCoMo-10 | Mean latency | 105.45 s | 94.01 s | 90.66 s |
| Internal lifecycle | Passed | 6/6 | 4/6 | 6/6 |
| Internal lifecycle | Mean latency | 14.34 s | 6.14 s | 10.42 s |

The four blocking safety checks passed in all three runs:

- project isolation;
- conversation isolation;
- supersession;
- unknown abstention.

Experiment A also passed both non-blocking LLM-dependent internal cases:

- live gate preference extraction;
- live gate knowledge update.

## Experiment A verdict

Experiment A is complete and did not support replacing MiMo with DeepSeek.

Provider reliability:

| Run | Fallback / judge | Attempts | Successes | Failures | Success rate |
|---|---|---:|---:|---:|---:|
| Run 1 | `mimo-v2.5-free` | 25 | 3 | 22 | 12.0% |
| Run 2 | `mimo-v2.5-free` | 22 | 2 | 20 | 9.1% |
| Experiment A | `deepseek-v4-flash-free` | 22 | 1 | 21 | 4.5% |

Judge coverage in Experiment A:

```text
0 / 14
```

All fourteen DeepSeek judge requests returned an empty completion.

```text
judge_request_errors: 0
judge_empty_responses: 14
judge_parse_errors: 0
```

Success criteria:

| Criterion | Result |
|---|---|
| Parseable external judge for most cases | Failed: 0/14 |
| Fallback materially better than MiMo | Failed: 4.5%, below both references |
| No blocking safety regression | Passed: 4/4, with 6/6 total internal cases passing |

Decision:

- do not promote `deepseek-v4-flash-free` to the scheduled default;
- retain role-aware telemetry;
- retain configurable manual fallback and judge inputs;
- keep scheduled MiMo defaults until a provider passes a controlled run.

## Provider alias finding

Successful requests made with the primary model name `big-pickle` were reported by OpenCode Zen as:

```text
deepseek-v4-flash
```

Successful response metadata:

| Role | Provider-reported model | Count |
|---|---|---:|
| benchmark reader | `deepseek-v4-flash` | 14 |
| memory selector | `deepseek-v4-flash` | 10 |
| memory extraction | `deepseek-v4-flash` | 3 |
| memory selector fallback | `deepseek-v4-flash-free` | 1 |

This does not prove the hidden implementation behind `big-pickle`. It does mean the provider currently reports a different model identifier, so the alias or provider metadata must be investigated before treating `big-pickle` and DeepSeek as independent backends.

## Conclusions supported by three runs

### Retrieval is stable

Raw retrieval is exactly reproducible across all three comparable runs:

- LongMemEval Recall@10: `1.0000`;
- LongMemEval MRR@10: `0.7083`;
- LoCoMo Recall@10: `0.7188`;
- LoCoMo MRR@10: `0.3802`.

Do not start by replacing embeddings, Qdrant, the reranker or RRF. Three runs do not identify retrieval as the primary source of variance.

### Evidence loss remains downstream

Selector output remains unstable despite identical retrieval.

Mean selected records:

| Suite | Run 1 | Run 2 | Experiment A |
|---|---:|---:|---:|
| LongMemEval-S | 4.000 | 0.833 | 1.500 |
| LoCoMo-10 | 4.500 | 2.750 | 2.375 |

Zero-selection cases:

| Suite | Run 1 | Run 2 | Experiment A |
|---|---:|---:|---:|
| LongMemEval-S | 2/6 | 2/6 | 3/6 |
| LoCoMo-10 | 2/8 | 3/8 | 2/8 |

The evidence path that needs measurement is:

```text
index -> raw retrieval -> selected context -> reader answer -> judge
```

### Token F1 remains insufficient

Token F1 changes materially while retrieval is identical. Translation, inflection, verbosity and answer format affect it strongly.

Do not interpret Token F1 as answer correctness without a reliable semantic judge and output-format diagnostics.

## Next experiment: Experiment B

Experiment B is now the immediate next task.

Add explicit observability and metric separation without changing selector, reader, retrieval or fusion behavior.

Required metrics:

- raw retrieval recall;
- selected recall;
- answer correctness;
- selector empty rate on answerable cases;
- judge coverage;
- judge empty-response rate;
- judge parse-failure rate;
- answer language match;
- concise-answer compliance.

Required case-level diagnostics:

- relevant raw document labels;
- selected document labels;
- whether relevant evidence was present in raw retrieval;
- whether relevant evidence survived selection;
- answer language;
- answer length or requested-value compliance;
- judge request outcome category;
- requested model and provider-reported model by role.

Do not add deterministic selector fallback yet. Experiment B must first show exactly where evidence disappears.

## Later isolated experiments

### Experiment C: neutral benchmark reader

Create a benchmark-only factual reader that:

- answers in the language of the question;
- returns only the requested value or short phrase;
- does not use the Lisa personality;
- does not add recommendations;
- abstains only when selected context lacks the answer.

Keep the production conversational reader separate.

### Experiment D: selector stabilization

Investigate a deterministic fallback only after selected recall and answerable zero-selection rate are available.

Candidate rule:

- when the LLM selector returns zero items but raw retrieval contains sufficiently strong evidence, pass a small top-N fallback set;
- retain zero selection for genuine abstention cases.

Do not force top-1 for every query.

### Experiment E: RRF versus normalized RSF

The current hybrid retrieval uses weighted RRF:

```text
RAG_RRF_K=60
RAG_DENSE_WEIGHT=0.6
RAG_LEXICAL_WEIGHT=0.4
```

RSF remains an optimization hypothesis, not a demonstrated fix.

When this experiment starts:

- add `RAG_FUSION_METHOD=rrf|rsf`;
- keep datasets, seed, models, selector and reader fixed;
- retain weights `0.6 / 0.4` for the first comparison;
- record raw dense and lexical scores, ranks, normalized scores, final score and normalization flags;
- compare Recall@10, MRR@10, selected recall and downstream correctness.

Do not start Experiment E as a reaction to Experiment A. Retrieval was identical for the third consecutive run.

### Experiment F: long-session chunking

Current memory text is truncated by `RAG_MAX_MEMORY_CHARS=6000` before indexing.

Preferred design:

```text
parent_session_id
chunk_id
chunk_index
```

Do not solve this only by increasing the character limit. Chunking must preserve stable parent identity and allow relevant portions of long sessions to be retrieved independently.

## Experiment discipline

For every comparison:

- use the same seed and case counts;
- change one subsystem at a time;
- create a new category only when benchmark semantics or dataset composition changes;
- retain failed cases in Neon;
- inspect case-level deltas, not only averages;
- do not call adapted subset scores official leaderboard results;
- keep deterministic safety checks blocking;
- keep stochastic extraction cases informational until stable enough to gate CI.

## Immediate task for the next session

1. Read this file and `docs/MEMORY_BENCHMARK_EXPERIMENT_A.md`.
2. Keep scheduled provider defaults unchanged.
3. Implement Experiment B metrics and case-level diagnostics.
4. Add tests for judge outcome classification and selected-recall calculation.
5. Run the same `6 / 8 / 20260714` subset.
6. Query Neon and compare against runs `29304658723`, `29309563088` and `29312255221`.
7. Only after Experiment B results decide whether to proceed to neutral reader or selector fallback.
8. Do not begin RRF-versus-RSF work yet.
