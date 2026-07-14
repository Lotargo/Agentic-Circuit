CREATE SCHEMA IF NOT EXISTS ml_eval;

CREATE TABLE IF NOT EXISTS ml_eval.benchmark_runs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_key text NOT NULL UNIQUE,
    category text NOT NULL,
    git_sha text NOT NULL,
    workflow_run_id bigint,
    artifact_id bigint,
    generated_at timestamptz NOT NULL,
    seed integer NOT NULL,
    primary_model text NOT NULL,
    fallback_models text[] NOT NULL DEFAULT ARRAY[]::text[],
    judge_model text,
    config jsonb NOT NULL DEFAULT '{}'::jsonb,
    provider_usage jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml_eval.benchmark_suite_metrics (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES ml_eval.benchmark_runs(id) ON DELETE CASCADE,
    suite_name text NOT NULL,
    case_count integer NOT NULL,
    completed_count integer NOT NULL,
    error_count integer NOT NULL,
    mean_token_f1 double precision,
    judge_accuracy double precision,
    retrieval_recall_at_10 double precision,
    mrr_at_10 double precision,
    unsupported_context_rate double precision,
    mean_elapsed_seconds double precision,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, suite_name)
);

CREATE TABLE IF NOT EXISTS ml_eval.benchmark_case_results (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES ml_eval.benchmark_runs(id) ON DELETE CASCADE,
    suite_name text NOT NULL,
    case_key text NOT NULL,
    category text NOT NULL,
    answer text NOT NULL DEFAULT '',
    expected text NOT NULL DEFAULT '',
    token_f1 double precision NOT NULL DEFAULT 0,
    judged_correct boolean,
    retrieval_recall double precision,
    reciprocal_rank double precision,
    selected_count integer NOT NULL DEFAULT 0,
    unsupported_context boolean NOT NULL DEFAULT false,
    elapsed_seconds double precision NOT NULL DEFAULT 0,
    error_text text NOT NULL DEFAULT '',
    manual_label text,
    manual_score double precision,
    manual_notes text,
    raw_result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, suite_name, case_key)
);

CREATE TABLE IF NOT EXISTS ml_eval.benchmark_provider_usage (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id bigint NOT NULL REFERENCES ml_eval.benchmark_runs(id) ON DELETE CASCADE,
    provider_name text NOT NULL,
    model_name text NOT NULL,
    attempts integer NOT NULL DEFAULT 0,
    successes integer NOT NULL DEFAULT 0,
    failures integer NOT NULL DEFAULT 0,
    fallback_successes integer NOT NULL DEFAULT 0,
    parameter_fallback_successes integer NOT NULL DEFAULT 0,
    raw_usage jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, provider_name, model_name)
);

CREATE INDEX IF NOT EXISTS benchmark_runs_category_generated_idx
    ON ml_eval.benchmark_runs (category, generated_at DESC);
CREATE INDEX IF NOT EXISTS benchmark_suite_metrics_run_idx
    ON ml_eval.benchmark_suite_metrics (run_id, suite_name);
CREATE INDEX IF NOT EXISTS benchmark_case_results_suite_category_idx
    ON ml_eval.benchmark_case_results (suite_name, category, run_id);
CREATE INDEX IF NOT EXISTS benchmark_case_results_manual_label_idx
    ON ml_eval.benchmark_case_results (manual_label)
    WHERE manual_label IS NOT NULL;
CREATE INDEX IF NOT EXISTS benchmark_provider_usage_run_idx
    ON ml_eval.benchmark_provider_usage (run_id, provider_name, model_name);

CREATE OR REPLACE VIEW ml_eval.benchmark_suite_deltas AS
WITH ordered AS (
    SELECT
        r.id AS run_id,
        r.run_key,
        r.category AS run_category,
        r.git_sha,
        r.workflow_run_id,
        r.generated_at,
        r.seed,
        s.suite_name,
        s.case_count,
        s.completed_count,
        s.error_count,
        s.mean_token_f1,
        s.judge_accuracy,
        s.retrieval_recall_at_10,
        s.mrr_at_10,
        s.unsupported_context_rate,
        s.mean_elapsed_seconds,
        lag(s.mean_token_f1) OVER metric_window AS previous_mean_token_f1,
        lag(s.judge_accuracy) OVER metric_window AS previous_judge_accuracy,
        lag(s.retrieval_recall_at_10) OVER metric_window AS previous_retrieval_recall_at_10,
        lag(s.mrr_at_10) OVER metric_window AS previous_mrr_at_10,
        lag(s.unsupported_context_rate) OVER metric_window AS previous_unsupported_context_rate,
        lag(s.mean_elapsed_seconds) OVER metric_window AS previous_mean_elapsed_seconds
    FROM ml_eval.benchmark_runs r
    JOIN ml_eval.benchmark_suite_metrics s ON s.run_id = r.id
    WINDOW metric_window AS (
        PARTITION BY r.category, r.seed, s.suite_name, s.case_count
        ORDER BY r.generated_at, r.id
    )
)
SELECT
    ordered.*,
    mean_token_f1 - previous_mean_token_f1 AS delta_mean_token_f1,
    judge_accuracy - previous_judge_accuracy AS delta_judge_accuracy,
    retrieval_recall_at_10 - previous_retrieval_recall_at_10 AS delta_retrieval_recall_at_10,
    mrr_at_10 - previous_mrr_at_10 AS delta_mrr_at_10,
    unsupported_context_rate - previous_unsupported_context_rate AS delta_unsupported_context_rate,
    mean_elapsed_seconds - previous_mean_elapsed_seconds AS delta_mean_elapsed_seconds
FROM ordered;
