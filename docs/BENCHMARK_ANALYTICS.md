# Benchmark analytics registry

Live memory benchmark runs are stored in Neon under the `ml_eval` schema.

## Stored layers

- `benchmark_runs`: one row per GitHub Actions run and attempt.
- `benchmark_suite_metrics`: aggregate metrics for LongMemEval-S, LoCoMo-10 and the internal safety suite.
- `benchmark_case_results`: every question, answer, retrieval score, latency and raw result. Manual review fields are reserved for later labeling.
- `benchmark_provider_usage`: provider-level and per-model attempts, successes, failures and fallback telemetry.
- `benchmark_suite_deltas`: comparison against the previous compatible run with the same category, seed, suite and case count.

The initial baseline is GitHub Actions run `29304658723` in category `live-memory-adapted-v1`.

## CI behavior

After deterministic safety checks pass, the workflow writes the report to Neon. `DATABASE_DIRECT_URL` is preferred; `DATABASE_URL` is used as a fallback. The connection string is never printed. A Neon delta section is appended to the Markdown artifact and therefore also appears in the GitHub job summary.

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

The database schema is also committed at `db/migrations/001_ml_eval_registry.sql` so the registry is reproducible outside the current Neon project.
