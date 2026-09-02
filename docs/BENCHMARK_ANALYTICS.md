# Benchmark analytics registry

Agentic Circuit stores scheduled live-memory benchmark results in PostgreSQL/Neon under the `ml_eval` schema. The schema is committed at `db/migrations/001_ml_eval_registry.sql`, so the registry can be reproduced outside the current hosted database.

## Stored layers

- `benchmark_runs` — one row per GitHub Actions run and attempt.
- `benchmark_suite_metrics` — aggregate metrics for LongMemEval-S, LoCoMo-10, and the internal memory suite.
- `benchmark_case_results` — per-case question, answer, retrieval metrics, latency, errors, and raw result metadata.
- `benchmark_provider_usage` — provider/model attempts, successes, failures, and fallback telemetry.
- `benchmark_suite_deltas` — comparison against the previous compatible run with the same category, seed, suite, and case count.

## What the workflow measures

The scheduled workflow runs three different classes of checks:

1. **LongMemEval-S adapted subset** — long-term memory retrieval and answering.
2. **LoCoMo-10 QA adapted subset** — question answering over long multi-session conversations.
3. **Internal memory lifecycle and isolation** — deterministic isolation, supersession, abstention, and live memory extraction/update behavior.

The external datasets are intentionally reported as **adapted subsets**. These results are regression/evaluation signals for this runtime, not official benchmark leaderboard scores.

## Model selection

Live benchmark model IDs are configuration, not source-code constants. The workflow resolves models in this order:

1. explicit `workflow_dispatch` inputs;
2. GitHub Actions repository variables;
3. the example `AGENTIC_PRIMARY_MODEL` and `AGENTIC_FALLBACK_MODEL` values in `.env.example`.

The relevant repository variables are:

- `AGENTIC_PRIMARY_MODEL` — primary reader/agent model;
- `AGENTIC_FALLBACK_MODEL` — fallback model;
- `BENCH_JUDGE_MODEL` — optional semantic judge model; when omitted, the resolved fallback model is used.

This keeps scheduled runs working out of the box while allowing model changes without a source-code commit. The workflow deliberately does not keep a local OpenCode Zen model allowlist because availability can change independently of Agentic Circuit.

Check the current OpenCode Zen catalog and endpoint mapping before changing these values:

<https://opencode.ai/docs/ru/zen/#%D0%B4%D0%BE%D1%81%D1%82%D1%83%D0%BF-%D0%BA-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8F%D0%BC>

The bundled Zen adapter currently targets `/zen/v1/chat/completions`, so select model IDs documented for that endpoint.

## Interpreting workflow status

Two independent conditions matter:

### Memory safety

The deterministic safety checker is a hard gate. The workflow fails when any blocking isolation invariant regresses, including:

- project isolation;
- conversation isolation;
- superseded memory handling;
- unknown/unsupported-memory abstention.

### Live benchmark completeness

A quality score is meaningful only when the external provider actually completes enough cases. Provider quota exhaustion, persistent rate limits, or widespread upstream failures must not produce a green quality benchmark with mostly missing answers.

The workflow therefore applies a second completeness gate after the report has been persisted and uploaded. By default, each external suite must complete at least 80% of its configured cases. The threshold is controlled by `BENCH_MIN_COMPLETION_RATIO`.

This separation keeps two questions distinct:

- **Did the deterministic memory safety invariants pass?**
- **Was the live provider healthy enough for the quality numbers to be meaningful?**

A run can pass the first question and fail the second. The uploaded artifact remains available for diagnosis.

## Provider telemetry

Provider telemetry is stored alongside suite metrics. It includes:

- attempts and successes by model;
- failures by model;
- fallback successes;
- parameter-retry successes;
- attempts, successes, and failures by role/model;
- judge request, empty-response, and parse errors.

This makes it possible to distinguish a retrieval or memory regression from a provider outage or quota failure.

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

Cases where retrieval found evidence but selection returned nothing:

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

Comparable run deltas:

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
WHERE run_category = 'live-memory-adapted-v1'
ORDER BY generated_at DESC, suite_name;
```

## Persistence behavior

After deterministic safety checks pass, the workflow writes the report to PostgreSQL/Neon. `DATABASE_DIRECT_URL` is preferred and `DATABASE_URL` is used as a fallback. Connection strings are never printed.

The Markdown report is also appended to the GitHub Job Summary and the raw benchmark directory is uploaded as a workflow artifact. The completeness gate runs after artifact publication so provider-degraded runs remain inspectable even when the workflow ultimately fails.
