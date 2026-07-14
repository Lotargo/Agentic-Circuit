#!/usr/bin/env python3
"""Fail a live benchmark run only on deterministic safety regressions.

External scores and LLM-dependent memory extraction remain informational until a
baseline has been collected across enough scheduled runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BLOCKING_CASES = {
    "project isolation",
    "conversation isolation",
    "supersession",
    "unknown abstention",
}


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark-results/live/benchmark-results.json")
    if not path.exists():
        print(f"benchmark report is missing: {path}", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    failures = []
    seen = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if case_id not in BLOCKING_CASES:
            continue
        seen.add(case_id)
        if case.get("error") or case.get("judged_correct") is not True:
            failures.append(case)
    missing = sorted(BLOCKING_CASES - seen)
    if missing:
        print(f"missing blocking benchmark cases: {', '.join(missing)}", file=sys.stderr)
        return 3
    if failures:
        print("deterministic memory safety regression:", file=sys.stderr)
        for case in failures:
            print(
                f"- {case.get('case_id')}: error={case.get('error')!r} "
                f"answer={case.get('answer')!r}",
                file=sys.stderr,
            )
        return 4
    print("deterministic memory safety cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
