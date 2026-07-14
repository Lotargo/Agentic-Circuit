# Memory benchmark Experiment A

Date: 2026-07-14

## Scope

Experiment A isolated the provider replacement requested in `docs/MEMORY_BENCHMARK_HANDOFF.md`.

Changed only for the experimental run:

```text
Primary: big-pickle
Fallback: deepseek-v4-flash-free
Judge: deepseek-v4-flash-free
LongMemEval cases: 6
LoCoMo cases: 8
Internal cases: 6
Seed: 20260714
Category: live-memory-adapted-v1
```

Retrieval, embeddings, Qdrant, reranker, weighted RRF, selector prompts, reader prompts, datasets and case sampling were unchanged.

Experimental GitHub Actions run:

```text
29312255221
```

Artifact:

```text
live-memory-benchmark-6
```

The run completed successfully, passed deterministic safety enforcement and was persisted to Neon.

## Telemetry added before the run

Provider telemetry now records:

- requested attempts, successes and failures by model;
- request role and model for `memory_select`, `memory_extract`, `benchmark_reader` and `benchmark_judge`;
- the model identifier returned by the provider on successful requests;
- model fallback reason;
- parameter-compatibility retry reason;
- judge request errors;
- empty judge responses;
- judge parse errors;
- bounded raw judge responses only when parsing fails.

No API keys, connection strings or private user conversations are stored in this telemetry.

## Aggregate results

| Suite | Metric | Run 1: MiMo | Run 2: MiMo | Experiment A: DeepSeek |
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

Raw retrieval remained exactly reproducible across all three runs. Experiment A did not change Recall@10 or MRR@10 for either external suite.

Token F1 moved inside the already observed stochastic range. It is not evidence that the provider replacement improved answer correctness because external judge coverage remained zero.

## Provider reliability

| Run | Fallback / judge model | Attempts | Successes | Failures | Success rate |
|---|---|---:|---:|---:|---:|
| Run 1 | `mimo-v2.5-free` | 25 | 3 | 22 | 12.0% |
| Run 2 | `mimo-v2.5-free` | 22 | 2 | 20 | 9.1% |
| Experiment A | `deepseek-v4-flash-free` | 22 | 1 | 21 | 4.5% |

DeepSeek was not materially better than MiMo. Its measured success rate was lower than both reference runs.

The only successful DeepSeek fallback was a `memory_select` request. Seven other DeepSeek `memory_select` attempts failed.

## Judge result

External judge coverage:

```text
0 / 14
```

All fourteen `benchmark_judge` requests to `deepseek-v4-flash-free` failed because the provider returned an empty completion.

```text
judge_request_errors: 0
judge_empty_responses: 14
judge_parse_errors: 0
```

This is a request-output failure, not a JSON parsing problem. There was no raw response to retain because every judge completion was empty.

## Actual model identifier returned by the provider

The benchmark requested `big-pickle` for primary reader, selector and extraction calls. On successful requests, the OpenCode Zen response reported:

```text
deepseek-v4-flash
```

Role-level successful response metadata:

| Role | Provider-reported model | Successful responses |
|---|---|---:|
| `benchmark_reader` | `deepseek-v4-flash` | 14 |
| `memory_select` | `deepseek-v4-flash` | 10 |
| `memory_select` fallback | `deepseek-v4-flash-free` | 1 |
| `memory_extract` | `deepseek-v4-flash` | 3 |

This does not prove the exact hidden implementation behind `big-pickle`. It does show that the provider currently reports `deepseek-v4-flash` rather than `big-pickle` for successful primary calls. The alias or provider metadata should be investigated before treating the two names as independent model backends.

## Selector behavior

Provider replacement did not stabilize selection.

| Suite | Run 1 mean selected | Run 2 mean selected | Experiment A mean selected |
|---|---:|---:|---:|
| LongMemEval-S | 4.000 | 0.833 | 1.500 |
| LoCoMo-10 | 4.500 | 2.750 | 2.375 |

Zero-selection cases:

| Suite | Run 1 | Run 2 | Experiment A |
|---|---:|---:|---:|
| LongMemEval-S | 2/6 | 2/6 | 3/6 |
| LoCoMo-10 | 2/8 | 3/8 | 2/8 |

The raw retrieval layer remained stable while selected context continued to vary. This supports the existing conclusion that evidence is being lost after retrieval.

## Safety and extraction

All six internal cases passed:

- project isolation;
- conversation isolation;
- supersession;
- unknown abstention;
- live gate preference extraction;
- live gate knowledge update.

The four deterministic blocking checks therefore had no regression.

The two LLM-dependent internal cases recovered from the second reference run, but this cannot be attributed to the DeepSeek fallback: the successful extraction responses were primary requests whose provider-reported model was `deepseek-v4-flash`.

## Success criteria verdict

| Criterion | Result |
|---|---|
| External judge returns a parseable result for most cases | Failed: 0/14 coverage |
| Fallback success rate is materially better than MiMo | Failed: 4.5%, below both MiMo runs |
| No regression in four blocking safety checks | Passed: 4/4, with 6/6 total internal cases passing |

Experiment A failed as a provider replacement decision. `deepseek-v4-flash-free` should not replace MiMo as the scheduled benchmark default on the strength of this run.

## Repository decision

The scheduled workflow keeps `mimo-v2.5-free` as its default fallback and judge model.

Manual benchmark runs now expose independent inputs for:

```text
fallback_model
judge_model
```

This preserves the existing baseline while allowing future isolated provider experiments without editing production agent manifests.

The role-aware telemetry should be retained. It exposed both the empty-judge failure mode and the provider-reported model alias that were invisible in the two reference runs.

## Recommended next step

Proceed to Experiment B before changing retrieval or fusion:

- calculate explicit judge coverage and empty-response rate in suite reporting;
- separate raw retrieval recall from selected recall;
- record selector empty rate on answerable cases;
- record answer language and concise-answer compliance;
- test a judge path that does not depend on the same OpenCode Zen empty-completion behavior.

Do not begin RRF-versus-RSF work as a response to Experiment A. Retrieval remained identical for the third consecutive run.
