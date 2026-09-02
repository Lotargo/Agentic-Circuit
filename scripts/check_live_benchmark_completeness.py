#!/usr/bin/env python3
"""Fail when live external benchmark scores are too incomplete to trust."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXTERNAL_SUITES = (
    "LongMemEval-S adapted subset",
    "LoCoMo-10 QA adapted subset",
)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark-results/live/benchmark-results.json")
    if not path.exists():
        print(f"benchmark report is missing: {path}", file=sys.stderr)
        return 2

    try:
        minimum_ratio = float(os.environ.get("BENCH_MIN_COMPLETION_RATIO", "0.80"))
    except ValueError:
        print("BENCH_MIN_COMPLETION_RATIO must be a number", file=sys.stderr)
        return 3
    if not 0 < minimum_ratio <= 1:
        print("BENCH_MIN_COMPLETION_RATIO must be in (0, 1]", file=sys.stderr)
        return 3

    report = json.loads(path.read_text(encoding="utf-8"))
    summaries = report.get("summaries", {})
    failures: list[str] = []

    for suite in EXTERNAL_SUITES:
        summary = summaries.get(suite)
        if not isinstance(summary, dict):
            failures.append(f"{suite}: summary missing")
            continue
        cases = int(summary.get("cases") or 0)
        completed = int(summary.get("completed") or 0)
        if cases <= 0:
            failures.append(f"{suite}: configured case count is {cases}")
            continue
        ratio = completed / cases
        print(f"{suite}: completed {completed}/{cases} ({ratio:.1%})")
        if ratio < minimum_ratio:
            failures.append(
                f"{suite}: completion {ratio:.1%} is below required {minimum_ratio:.1%}"
            )

    if failures:
        print("live benchmark is provider-degraded or incomplete:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)

        provider_usage = report.get("provider_usage", {})
        for provider_name, usage in provider_usage.items():
            if not isinstance(usage, dict):
                continue
            attempts = sum((usage.get("attempts_by_model") or {}).values())
            successes = sum((usage.get("successes_by_model") or {}).values())
            failed = sum((usage.get("failures_by_model") or {}).values())
            print(
                f"- provider {provider_name}: attempts={attempts} successes={successes} failures={failed}",
                file=sys.stderr,
            )
        return 4

    print(f"live benchmark completeness passed at threshold {minimum_ratio:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
