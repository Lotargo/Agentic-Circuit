# Agentic Circuit documentation

This directory separates current public documentation from engineering history.

## Start here

- [`API_MAP.md`](API_MAP.md) — public routes, internal service boundaries, request context, streaming, and the Scalar/OpenAPI entry points.
- [`BENCHMARK_ANALYTICS.md`](BENCHMARK_ANALYTICS.md) — benchmark storage, interpretation, provider telemetry, and regression semantics.
- [`MEMORY_BENCHMARK_EXPERIMENT_A.md`](MEMORY_BENCHMARK_EXPERIMENT_A.md) — recorded controlled benchmark experiment.
- [`OPENCODE_MODEL_PROTOCOL_DIAGNOSIS.md`](OPENCODE_MODEL_PROTOCOL_DIAGNOSIS.md) — protocol compatibility notes for the current provider family.

## Engineering history

The following documents are retained as implementation history rather than current operating instructions:

- `LOCAL_PROVIDER_DEBUG_HANDOFF.md`
- `MEMORY_BENCHMARK_HANDOFF.md`
- `superpowers/specs/*`

Handoff documents describe the repository at the time they were written and may reference commits, paths, experiments, or next steps that have already been superseded. They should not be used as the primary source for the current architecture.

For the current system surface, use the root [`README.md`](../README.md), this index, and [`API_MAP.md`](API_MAP.md).
