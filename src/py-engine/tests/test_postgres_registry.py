from __future__ import annotations

from agentic_circuit.benchmarks.postgres_registry import (
    RegistryContext,
    case_rows,
    format_delta_markdown,
    provider_rows,
    select_database_url,
    suite_rows,
)


def _report() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-07-14T04:20:20+00:00",
        "commit": "abc123",
        "seed": 7,
        "model_chain": ["primary", "fallback"],
        "judge_model": "judge",
        "sources": {"sample": "https://example.invalid"},
        "summaries": {
            "suite-a": {
                "cases": 1,
                "completed": 1,
                "errors": 0,
                "mean_token_f1": 0.5,
                "judge_accuracy": 1.0,
                "retrieval_recall_at_10": 0.75,
                "mrr_at_10": 0.5,
                "unsupported_context_rate": 0.0,
                "mean_elapsed_seconds": 2.5,
            }
        },
        "provider_usage": {
            "provider": {
                "attempts_by_model": {"primary": 3, "fallback": 1},
                "successes_by_model": {"primary": 2, "fallback": 1},
                "failures_by_model": {"primary": 1},
                "fallback_successes": 1,
                "parameter_fallback_successes": 0,
            }
        },
        "cases": [
            {
                "benchmark": "suite-a",
                "case_id": "case-1",
                "category": "fact",
                "answer": "A",
                "expected": "A",
                "token_f1": 1.0,
                "judged_correct": True,
                "retrieval_recall": 1.0,
                "reciprocal_rank": 1.0,
                "selected_count": 1,
                "unsupported_context": False,
                "elapsed_seconds": 2.5,
                "error": "",
            }
        ],
    }


def test_registry_context_uses_github_attempt_in_key() -> None:
    context = RegistryContext(
        category="live-memory-adapted-v1",
        repository="owner/repo",
        workflow_run_id=123,
        workflow_run_attempt=2,
        workflow_run_number=10,
        event_name="workflow_dispatch",
        git_ref="refs/heads/main",
        artifact_id=None,
        notes="test",
    )
    assert context.run_key(_report()) == "github:owner/repo:123:attempt:2"


def test_report_transformers_preserve_metrics() -> None:
    report = _report()
    suites = suite_rows(report)
    cases = case_rows(report)
    providers = provider_rows(report)

    assert suites[0]["retrieval_recall_at_10"] == 0.75
    assert cases[0]["raw_result"]["case_id"] == "case-1"
    assert providers[0]["model_name"] == "__provider__"
    assert providers[0]["attempts"] == 4
    assert {row["model_name"] for row in providers[1:]} == {
        "primary",
        "fallback",
    }


def test_database_url_prefers_direct_connection() -> None:
    assert (
        select_database_url(
            {
                "DATABASE_DIRECT_URL": "postgresql://direct",
                "DATABASE_URL": "postgresql://pooled",
            }
        )
        == "postgresql://direct"
    )
    assert (
        select_database_url({"DATABASE_URL": "postgresql://pooled"})
        == "postgresql://pooled"
    )


def test_delta_markdown_formats_percentage_points() -> None:
    markdown = format_delta_markdown(
        "github:owner/repo:1:attempt:1",
        "category-v1",
        [
            {
                "suite_name": "suite-a",
                "delta_mean_token_f1": 0.1,
                "delta_judge_accuracy": None,
                "delta_retrieval_recall_at_10": -0.05,
                "delta_mrr_at_10": 0.0,
                "delta_unsupported_context_rate": None,
                "delta_mean_elapsed_seconds": -2.5,
            }
        ],
    )
    assert "category-v1" in markdown
    assert "+10.0 pp" in markdown
    assert "-5.0 pp" in markdown
    assert "-2.5 s" in markdown
