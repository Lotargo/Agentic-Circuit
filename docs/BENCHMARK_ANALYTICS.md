# Benchmark analytics registry

Live memory benchmark runs are stored in Neon under the `ml_eval` schema.

For the current engineering state, conclusions and ordered next experiments, start with:

```text
docs/MEMORY_BENCHMARK_HANDOFF.md
```

## Stored layers

- `benchmark_runs`: one row per GitHub Actions run and attempt.
- `benchmark_suite_metrics`: aggregate metrics for LongMemEval-S, LoCoMo-10 and the internal safety suite.
- `benchmark_case_results`: every question, answer, retrieval score, latency and raw result. Manual review fields are reserved for later labeling.
- `benchmark_provider_usage`: provider-level and per-model attempts, successes, failures and fallback telemetry.
- `benchmark_suite_deltas`: comparison against the previous compatible run with the same category, seed, suite and case count.

## Reference runs

Two comparable runs are currently stored in category `live-memory-adapted-v1`:

| Role | GitHub run | Git commit | Seed |
|---|---:|---|---:|
| Initial baseline | `29304658723` | `0a1dbffd10b9760aa9268e205dd23ed28989c071` | `20260714` |
| Second reference | `29309563088` | `c419434f8f7f07a5da26b8095e458f997f21d5b6` | `20260714` |

The two runs used the same 6 LongMemEval cases, 8 LoCoMo cases and 6 internal cases.

Raw retrieval metrics were reproducible across both runs, while selector, reader, memory extraction and judge behavior were not. The detailed comparison is recorded in `docs/MEMORY_BENCHMARK_HANDOFF.md`.

## Current model decision

The current stored references used:

```text
Primary: big-pickle
Fallback: mimo-v2.5-free
Judge: mimo-v2.5-free
```

The next controlled experiment must replace every benchmark use of `mimo-v2.5-free` with:

```text
deepseek-v4-flash-free
```

Keep the first post-change run otherwise identical so the provider effect remains measurable.

## Fusion investigation

The current hybrid retrieval uses weighted RRF.

A future isolated experiment will compare it with an RSF-style normalized score fusion. This is not yet a production decision. Normalization method, raw scores and fused scores must be observable before conclusions are drawn.

Do not combine the provider replacement with the RRF-versus-RSF experiment.

## CI behavior

After deterministic safety checks pass, the workflow writes the report to Neon. `DATABASE_DIRECT_URL` is preferred; `DATABASE_URL` is used as a fallback. The connection string is never printed. A Neon delta section is appended to the Markdown artifact and therefore also appears in the GitHub job summary.

The workflow uses one shared concurrency group with `cancel-in-progress: false`. Expensive runs are serialized, but a new waiting request does not terminate an active benchmark.

`LANGSEARCH_API_KEY` and `QDRANT_API_KEY` are not required by this benchmark. It uses local Qdrant and local TEI containers, while the public datasets and OpenCode Zen calls use their own configured paths.

## Useful queries

Latest suite metrics:

```sql
SELECT
    generated_at,
    git_sha,
    suite_name,
    mean_token_f1,
    judge_accuracy,
    retrieval_recall_at_10,
    mrr_at_10,
    mean_elapsed_seconds
FROM ml_eval.benchmark_suite_deltas
WHERE run_category = 'live-memory-adapted-v1'
ORDER BY generated_at DESC, suite_name;
```

Cases where retrieval found evidence but the selector returned nothing:

```sql
SELECT
    r.generated_at,
    c.suite_name,
    c.case_key,
    c.retrieval_recall,
    c.selected_count,
    c.expected,
    c.answer
FROM ml_eval.benchmark_case_results c
JOIN ml_eval.benchmark_runs r ON r.id = c.run_id
WHERE c.retrieval_recall > 0
  AND c.selected_count = 0
ORDER BY r.generated_at DESC;
```

Provider reliability:

```sql
SELECT
    r.generated_at,
    p.provider_name,
    p.model_name,
    p.attempts,
    p.successes,
    p.failures,
    p.fallback_successes
FROM ml_eval.benchmark_provider_usage p
JOIN ml_eval.benchmark_runs r ON r.id = p.run_id
ORDER BY r.generated_at DESC, p.provider_name, p.model_name;
```

Compare the two stored reference runs:

```sql
SELECT
    generated_at,
    workflow_run_id,
    suite_name,
    mean_token_f1,
    judge_accuracy,
    retrieval_recall_at_10,
    mrr_at_10,
    mean_elapsed_seconds,
    delta_mean_token_f1,
    delta_judge_accuracy,
    delta_retrieval_recall_at_10,
    delta_mrr_at_10,
    delta_mean_elapsed_seconds
FROM ml_eval.benchmark_suite_deltas
WHERE workflow_run_id IN (29304658723, 29309563088)
ORDER BY generated_at, suite_name;
```

The database schema is also committed at `db/migrations/001_ml_eval_registry.sql` so the registry is reproducible outside the current Neon project.
