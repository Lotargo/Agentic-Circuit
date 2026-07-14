"""Persist live benchmark reports to the Neon-backed ML evaluation registry."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_CATEGORY = "live-memory-adapted-v1"
SECTION_START = "<!-- neon-registry:start -->"
SECTION_END = "<!-- neon-registry:end -->"


@dataclass(frozen=True)
class RegistryContext:
    category: str
    repository: str
    workflow_run_id: int | None
    workflow_run_attempt: int
    workflow_run_number: int | None
    event_name: str
    git_ref: str
    artifact_id: int | None
    notes: str

    def run_key(self, report: Mapping[str, Any]) -> str:
        if self.repository and self.workflow_run_id is not None:
            return (
                f"github:{self.repository}:{self.workflow_run_id}:"
                f"attempt:{self.workflow_run_attempt}"
            )
        return (
            f"local:{report.get('commit', 'unknown')}:"
            f"{report.get('generated_at', 'unknown')}"
        )


def select_database_url(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    url = values.get("DATABASE_DIRECT_URL") or values.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_DIRECT_URL or DATABASE_URL must be configured; "
            "the connection string is never printed"
        )
    return url


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"benchmark report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "generated_at",
        "commit",
        "seed",
        "model_chain",
        "judge_model",
        "summaries",
        "provider_usage",
        "cases",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"benchmark report is missing fields: {', '.join(missing)}")
    if not report["model_chain"]:
        raise ValueError("benchmark report contains an empty model_chain")
    return report


def suite_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_name, summary in report["summaries"].items():
        rows.append(
            {
                "suite_name": str(suite_name),
                "case_count": int(summary["cases"]),
                "completed_count": int(summary["completed"]),
                "error_count": int(summary["errors"]),
                "mean_token_f1": summary.get("mean_token_f1"),
                "judge_accuracy": summary.get("judge_accuracy"),
                "retrieval_recall_at_10": summary.get("retrieval_recall_at_10"),
                "mrr_at_10": summary.get("mrr_at_10"),
                "unsupported_context_rate": summary.get("unsupported_context_rate"),
                "mean_elapsed_seconds": summary.get("mean_elapsed_seconds"),
            }
        )
    return rows


def case_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report["cases"]:
        rows.append(
            {
                "suite_name": str(case["benchmark"]),
                "case_key": str(case["case_id"]),
                "category": str(case.get("category", "unknown")),
                "answer": str(case.get("answer", "")),
                "expected": str(case.get("expected", "")),
                "token_f1": float(case.get("token_f1", 0.0)),
                "judged_correct": case.get("judged_correct"),
                "retrieval_recall": case.get("retrieval_recall"),
                "reciprocal_rank": case.get("reciprocal_rank"),
                "selected_count": int(case.get("selected_count", 0)),
                "unsupported_context": bool(case.get("unsupported_context", False)),
                "elapsed_seconds": float(case.get("elapsed_seconds", 0.0)),
                "error_text": str(case.get("error", "")),
                "raw_result": dict(case),
            }
        )
    return rows


def provider_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for provider_name, usage in report["provider_usage"].items():
        attempts = usage.get("attempts_by_model", {})
        successes = usage.get("successes_by_model", {})
        failures = usage.get("failures_by_model", {})
        rows.append(
            {
                "provider_name": str(provider_name),
                "model_name": "__provider__",
                "attempts": sum(int(value) for value in attempts.values()),
                "successes": sum(int(value) for value in successes.values()),
                "failures": sum(int(value) for value in failures.values()),
                "fallback_successes": int(usage.get("fallback_successes", 0)),
                "parameter_fallback_successes": int(
                    usage.get("parameter_fallback_successes", 0)
                ),
                "raw_usage": dict(usage),
            }
        )
        for model_name in sorted(set(attempts) | set(successes) | set(failures)):
            model_attempts = int(attempts.get(model_name, 0))
            model_successes = int(successes.get(model_name, 0))
            model_failures = int(failures.get(model_name, 0))
            rows.append(
                {
                    "provider_name": str(provider_name),
                    "model_name": str(model_name),
                    "attempts": model_attempts,
                    "successes": model_successes,
                    "failures": model_failures,
                    "fallback_successes": 0,
                    "parameter_fallback_successes": 0,
                    "raw_usage": {
                        "attempts": model_attempts,
                        "successes": model_successes,
                        "failures": model_failures,
                    },
                }
            )
    return rows


def registry_config(
    report: Mapping[str, Any], context: RegistryContext
) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "sources": report.get("sources", {}),
        "sample_sizes": {
            name: int(summary["cases"])
            for name, summary in report["summaries"].items()
        },
        "repository": context.repository,
        "workflow_run_attempt": context.workflow_run_attempt,
        "workflow_run_number": context.workflow_run_number,
        "event_name": context.event_name,
        "git_ref": context.git_ref,
        "protocol": "adapted-subset",
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "baseline"
    return f"{value * 100:+.1f} pp"


def _seconds(value: float | None) -> str:
    if value is None:
        return "baseline"
    return f"{value:+.1f} s"


def format_delta_markdown(
    run_key: str,
    category: str,
    deltas: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        SECTION_START,
        "## Neon benchmark registry",
        "",
        f"- Run key: `{run_key}`",
        f"- Registry category: `{category}`",
        "",
        "| Suite | Δ Token F1 | Δ Judge | Δ Recall@10 | Δ MRR@10 | Δ Unsupported | Δ Mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        lines.append(
            f"| {row['suite_name']} | "
            f"{_pct(row.get('delta_mean_token_f1'))} | "
            f"{_pct(row.get('delta_judge_accuracy'))} | "
            f"{_pct(row.get('delta_retrieval_recall_at_10'))} | "
            f"{_pct(row.get('delta_mrr_at_10'))} | "
            f"{_pct(row.get('delta_unsupported_context_rate'))} | "
            f"{_seconds(row.get('delta_mean_elapsed_seconds'))} |"
        )
    if not deltas:
        lines.append(
            "| No comparable suites | baseline | baseline | baseline | baseline | baseline | baseline |"
        )
    lines.extend(["", SECTION_END, ""])
    return "\n".join(lines)


def update_markdown_report(path: Path, section: str) -> None:
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(
        rf"\n?{re.escape(SECTION_START)}.*?{re.escape(SECTION_END)}\n?",
        flags=re.DOTALL,
    )
    cleaned = pattern.sub("\n", current).rstrip()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{cleaned}\n\n{section}" if cleaned else section, encoding="utf-8")


def persist_report(
    report: Mapping[str, Any],
    context: RegistryContext,
    *,
    database_url: str,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        import psycopg
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is required for Neon persistence; run with "
            "`uv run --with 'psycopg[binary]>=3.2,<4'`"
        ) from exc

    generated_at = datetime.fromisoformat(str(report["generated_at"]))
    model_chain = [str(model) for model in report["model_chain"]]
    run_key = context.run_key(report)
    config = registry_config(report, context)

    with psycopg.connect(
        database_url,
        connect_timeout=20,
        application_name="chat-openwebui-live-benchmark",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ml_eval.benchmark_runs (
                    run_key, category, git_sha, workflow_run_id, artifact_id,
                    generated_at, seed, primary_model, fallback_models,
                    judge_model, config, provider_usage, notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_key) DO UPDATE SET
                    category = EXCLUDED.category,
                    git_sha = EXCLUDED.git_sha,
                    workflow_run_id = EXCLUDED.workflow_run_id,
                    artifact_id = EXCLUDED.artifact_id,
                    generated_at = EXCLUDED.generated_at,
                    seed = EXCLUDED.seed,
                    primary_model = EXCLUDED.primary_model,
                    fallback_models = EXCLUDED.fallback_models,
                    judge_model = EXCLUDED.judge_model,
                    config = EXCLUDED.config,
                    provider_usage = EXCLUDED.provider_usage,
                    notes = EXCLUDED.notes
                RETURNING id
                """,
                (
                    run_key,
                    context.category,
                    str(report["commit"]),
                    context.workflow_run_id,
                    context.artifact_id,
                    generated_at,
                    int(report["seed"]),
                    model_chain[0],
                    model_chain[1:],
                    report.get("judge_model"),
                    Jsonb(config),
                    Jsonb(report["provider_usage"]),
                    context.notes,
                ),
            )
            run_id = int(cursor.fetchone()[0])

            cursor.executemany(
                """
                INSERT INTO ml_eval.benchmark_suite_metrics (
                    run_id, suite_name, case_count, completed_count, error_count,
                    mean_token_f1, judge_accuracy, retrieval_recall_at_10,
                    mrr_at_10, unsupported_context_rate, mean_elapsed_seconds
                ) VALUES (
                    %(run_id)s, %(suite_name)s, %(case_count)s,
                    %(completed_count)s, %(error_count)s, %(mean_token_f1)s,
                    %(judge_accuracy)s, %(retrieval_recall_at_10)s,
                    %(mrr_at_10)s, %(unsupported_context_rate)s,
                    %(mean_elapsed_seconds)s
                )
                ON CONFLICT (run_id, suite_name) DO UPDATE SET
                    case_count = EXCLUDED.case_count,
                    completed_count = EXCLUDED.completed_count,
                    error_count = EXCLUDED.error_count,
                    mean_token_f1 = EXCLUDED.mean_token_f1,
                    judge_accuracy = EXCLUDED.judge_accuracy,
                    retrieval_recall_at_10 = EXCLUDED.retrieval_recall_at_10,
                    mrr_at_10 = EXCLUDED.mrr_at_10,
                    unsupported_context_rate = EXCLUDED.unsupported_context_rate,
                    mean_elapsed_seconds = EXCLUDED.mean_elapsed_seconds
                """,
                [dict(row, run_id=run_id) for row in suite_rows(report)],
            )

            cursor.executemany(
                """
                INSERT INTO ml_eval.benchmark_case_results (
                    run_id, suite_name, case_key, category, answer, expected,
                    token_f1, judged_correct, retrieval_recall, reciprocal_rank,
                    selected_count, unsupported_context, elapsed_seconds,
                    error_text, raw_result
                ) VALUES (
                    %(run_id)s, %(suite_name)s, %(case_key)s, %(category)s,
                    %(answer)s, %(expected)s, %(token_f1)s,
                    %(judged_correct)s, %(retrieval_recall)s,
                    %(reciprocal_rank)s, %(selected_count)s,
                    %(unsupported_context)s, %(elapsed_seconds)s,
                    %(error_text)s, %(raw_result)s
                )
                ON CONFLICT (run_id, suite_name, case_key) DO UPDATE SET
                    category = EXCLUDED.category,
                    answer = EXCLUDED.answer,
                    expected = EXCLUDED.expected,
                    token_f1 = EXCLUDED.token_f1,
                    judged_correct = EXCLUDED.judged_correct,
                    retrieval_recall = EXCLUDED.retrieval_recall,
                    reciprocal_rank = EXCLUDED.reciprocal_rank,
                    selected_count = EXCLUDED.selected_count,
                    unsupported_context = EXCLUDED.unsupported_context,
                    elapsed_seconds = EXCLUDED.elapsed_seconds,
                    error_text = EXCLUDED.error_text,
                    raw_result = EXCLUDED.raw_result
                """,
                [
                    dict(row, run_id=run_id, raw_result=Jsonb(row["raw_result"]))
                    for row in case_rows(report)
                ],
            )

            cursor.executemany(
                """
                INSERT INTO ml_eval.benchmark_provider_usage (
                    run_id, provider_name, model_name, attempts, successes,
                    failures, fallback_successes, parameter_fallback_successes,
                    raw_usage
                ) VALUES (
                    %(run_id)s, %(provider_name)s, %(model_name)s,
                    %(attempts)s, %(successes)s, %(failures)s,
                    %(fallback_successes)s,
                    %(parameter_fallback_successes)s, %(raw_usage)s
                )
                ON CONFLICT (run_id, provider_name, model_name) DO UPDATE SET
                    attempts = EXCLUDED.attempts,
                    successes = EXCLUDED.successes,
                    failures = EXCLUDED.failures,
                    fallback_successes = EXCLUDED.fallback_successes,
                    parameter_fallback_successes =
                        EXCLUDED.parameter_fallback_successes,
                    raw_usage = EXCLUDED.raw_usage
                """,
                [
                    dict(row, run_id=run_id, raw_usage=Jsonb(row["raw_usage"]))
                    for row in provider_rows(report)
                ],
            )

            cursor.execute(
                """
                SELECT
                    suite_name,
                    delta_mean_token_f1,
                    delta_judge_accuracy,
                    delta_retrieval_recall_at_10,
                    delta_mrr_at_10,
                    delta_unsupported_context_rate,
                    delta_mean_elapsed_seconds
                FROM ml_eval.benchmark_suite_deltas
                WHERE run_id = %s
                ORDER BY suite_name
                """,
                (run_id,),
            )
            columns = [column.name for column in cursor.description]
            deltas = [dict(zip(columns, row)) for row in cursor.fetchall()]

    return run_key, deltas


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("benchmark-results/live/benchmark-results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmark-results/live/benchmark-report.md"),
    )
    parser.add_argument(
        "--category",
        default=os.environ.get("BENCH_REGISTRY_CATEGORY", DEFAULT_CATEGORY),
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        default=_optional_int(os.environ.get("GITHUB_RUN_ID")),
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
    )
    parser.add_argument(
        "--workflow-run-number",
        type=int,
        default=_optional_int(os.environ.get("GITHUB_RUN_NUMBER")),
    )
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "local"))
    parser.add_argument("--git-ref", default=os.environ.get("GITHUB_REF", ""))
    parser.add_argument("--artifact-id", type=int)
    parser.add_argument(
        "--notes",
        default="Persisted automatically by the live memory benchmark workflow",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_report(args.result)
    context = RegistryContext(
        category=args.category,
        repository=args.repository,
        workflow_run_id=args.workflow_run_id,
        workflow_run_attempt=args.workflow_run_attempt,
        workflow_run_number=args.workflow_run_number,
        event_name=args.event_name,
        git_ref=args.git_ref,
        artifact_id=args.artifact_id,
        notes=args.notes,
    )
    run_key, deltas = persist_report(
        report,
        context,
        database_url=select_database_url(),
    )
    section = format_delta_markdown(run_key, context.category, deltas)
    update_markdown_report(args.report, section)
    print(
        f"stored benchmark run {run_key}: "
        f"{len(suite_rows(report))} suites, "
        f"{len(case_rows(report))} cases, "
        f"{len(provider_rows(report))} provider rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
