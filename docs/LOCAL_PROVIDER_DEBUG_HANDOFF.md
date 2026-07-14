# Local provider debugging handoff

Last updated: 2026-07-14

## Purpose

This is the primary continuation document for the next development session.

The immediate goal is not to run another full memory benchmark. The goal is to move provider and model debugging to a local, reproducible workflow and determine whether the failures observed in Experiment A came from:

- the model;
- the OpenCode gateway;
- an OpenAI-compatible protocol mismatch;
- output-token exhaustion in a reasoning model;
- OpenAI SDK response parsing;
- or the project's provider client.

Give this file to the next AI assistant and ask it to continue from here.

## Repository state

Repository:

```text
Lotargo/chat-openwebui
```

Working branch:

```text
agent/experiment-a-deepseek-provider
```

Open draft pull request:

```text
PR #17: Add provider telemetry and OpenCode protocol diagnostics
```

Do not assume the PR is merged. Continue on the branch unless the user explicitly chooses another branch.

Read these files before changing code:

```text
docs/LOCAL_PROVIDER_DEBUG_HANDOFF.md
docs/MEMORY_BENCHMARK_HANDOFF.md
docs/MEMORY_BENCHMARK_EXPERIMENT_A.md
docs/OPENCODE_MODEL_PROTOCOL_DIAGNOSIS.md
src/py-engine/src/agentic_circuit/providers/client.py
src/py-engine/src/agentic_circuit/config/schema.py
src/py-engine/src/agentic_circuit/benchmarks/live.py
src/py-engine/scripts/probe_opencode_protocols.py
config/providers.yaml
config/agents/memory.yaml
config/agents/synthesis.yaml
```

## Current decision

Development now becomes local-first.

GitHub Actions remains useful for final regression checks, Docker builds and a controlled benchmark after local verification. It must not remain the primary way to discover provider response shapes, token-limit behavior or parsing errors.

The next candidate models should use an OpenAI-compatible chat-completions endpoint. The user will either provide the current OpenCode model list or ask the assistant to verify it on the web.

Do not select the final primary, fallback or judge model before the local protocol matrix is complete.

## Facts already established

### LangGraph is not the cause of Experiment A provider failures

The live benchmark does not send model calls through the LangGraph graph. It creates the project provider registry directly and calls:

```python
client.acomplete(...)
```

The provider client then calls:

```python
AsyncOpenAI.chat.completions.create(...)
```

LangChain is not in this request path.

LangGraph remains part of the production orchestration graph, but it did not transform the benchmark judge responses that were classified as empty.

### Experiment A did not prove that DeepSeek itself was broken

Experiment A used:

```text
primary: big-pickle
fallback: deepseek-v4-flash-free
judge: deepseek-v4-flash-free
```

Observed provider telemetry:

```text
DeepSeek requested attempts: 22
DeepSeek successful visible completions: 1
DeepSeek failures: 21
External judge coverage: 0 / 14
Judge empty completions: 14
Judge request exceptions: 0
Judge parse failures: 0
```

The judge used only:

```text
max_tokens = 80
```

The old client inspected `message.content`, but did not retain enough information from an empty visible completion:

- `finish_reason`;
- provider-specific reasoning fields;
- complete message field names;
- raw successful response shape;
- detailed token usage for the failed visible completion.

A reasoning model may consume a small output allowance before producing visible final text. Therefore the correct current statement is:

> OpenCode returned no visible `message.content` that the project accepted for those judge calls.

It is not yet proven whether the model, gateway, token budget, SDK or project wrapper caused that result.

### Big Pickle may be related to DeepSeek, but this is not proven

Successful calls requested as:

```text
big-pickle
```

were reported in provider response metadata as:

```text
deepseek-v4-flash
```

This is compatible with an alias, router mapping, shared model family or incorrect provider metadata. It does not prove that Big Pickle and DeepSeek V4 Flash are identical weights or identical deployments.

Treat them as potentially correlated until independent behavior is demonstrated.

### Qwen is not part of the immediate migration plan

The previously considered Qwen model used a different API protocol from the current OpenAI-compatible client. The user has decided to choose another model that fits the OpenAI-compatible endpoint instead.

Do not implement an Anthropic-compatible transport or cross-protocol fallback in this task unless the user changes this decision.

## Existing diagnostic work

The branch currently contains:

```text
src/py-engine/scripts/probe_opencode_protocols.py
.github/workflows/opencode-protocol-probe.yml
docs/OPENCODE_MODEL_PROTOCOL_DIAGNOSIS.md
```

The existing script is a useful starting point, but it was designed during the earlier Qwen/OpenCode Go investigation. The next assistant should adapt or replace it with a local OpenAI-compatible provider probe.

The failed GitHub workflow attempts did not reach the provider because the required secret was unavailable in those runs. They are not model test results.

## Local development setup

The repository already supports full local startup.

From the repository root:

```bash
cp .env.example .env
docker compose up --build
```

NVIDIA variant:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Local service addresses:

```text
OpenWebUI:    http://localhost:3200
TS gateway:  http://localhost:9191
Python API:  http://localhost:8823
Qdrant:      http://localhost:6633
Embedding:   http://localhost:8899
Reranker:    http://localhost:8898
```

For provider debugging, do not start the full stack first. The first diagnostic phases should require only Python, the provider key and internet access.

Python setup:

```bash
cd src/py-engine
uv sync --frozen --extra test
```

Set the provider key in the local shell or `.env`. Never commit the key.

Example for bash/WSL:

```bash
export OPENCODE_ZEN_API_KEY='...'
```

Example for PowerShell:

```powershell
$env:OPENCODE_ZEN_API_KEY='...'
```

The local probe must read credentials only from environment variables.

## Local-first investigation order

Follow this order. Do not begin with the full LongMemEval/LoCoMo workflow.

```text
1. Raw HTTP request
2. Official OpenAI Python SDK request
3. Project OpenAICompatibleClient request
4. Small provider-role matrix
5. Small memory benchmark subset
6. Full fixed benchmark subset
7. CI regression validation
```

Each stage must explain the failure before the next layer is introduced.

## Phase 1: build a local OpenAI-compatible raw probe

Create or refactor a script intended to run locally, for example:

```text
src/py-engine/scripts/probe_openai_compatible_provider.py
```

The script should accept configuration through CLI arguments and environment variables rather than hard-coded model names.

Suggested interface:

```bash
uv run python scripts/probe_openai_compatible_provider.py \
  --base-url <openai-compatible-base-url> \
  --api-key-env OPENCODE_ZEN_API_KEY \
  --model <model-id> \
  --max-tokens 80 256 1024
```

Support repeated `--model` arguments or a comma-separated model list.

For each request, record a redacted JSON result containing:

- transport: `raw_httpx`, `openai_sdk` or `project_client`;
- requested model;
- provider-reported model;
- endpoint;
- HTTP status;
- latency;
- choice count;
- message field names;
- visible content length;
- a bounded visible-content sample;
- reasoning-field names and lengths without exposing full chain-of-thought;
- `finish_reason`;
- prompt tokens;
- completion tokens;
- total tokens;
- exception type and redacted message;
- classification described below.

Do not store API keys, authorization headers, private conversations or full reasoning traces.

Write local output to an ignored directory such as:

```text
.local-results/provider-probe/
```

Add the directory to `.gitignore` if needed.

## Phase 2: test two prompt classes

Every candidate model must receive the same two prompt classes.

### Minimal deterministic prompt

Purpose: verify basic visible completion and JSON capability without project prompts.

Example:

```text
Return JSON only: {"ok":true}
```

### Real benchmark judge prompt

Use the same semantic-judge prompt structure used by:

```text
src/py-engine/src/agentic_circuit/benchmarks/live.py
```

Use a fixed simple question/reference/candidate trio whose correct result is obvious.

Run both prompt classes at no less than:

```text
80
256
1024
```

output tokens.

Keep temperature and top-p fixed across candidates. Prefer temperature `0.0` and a low top-p for judge tests.

## Phase 3: classify outcomes correctly

Do not collapse every non-visible answer into one generic empty-completion error.

Use explicit categories:

```text
success_visible_content
success_parseable_json
http_error
sdk_error
provider_error
empty_no_reasoning
reasoning_only
output_budget_exhausted
content_filtered
missing_choices
missing_message
invalid_json
unknown_response_shape
```

Suggested rules:

- visible non-empty content and valid expected JSON: `success_parseable_json`;
- visible non-empty content but invalid expected JSON: `invalid_json`;
- empty content with a reasoning field and `finish_reason=length`: `output_budget_exhausted`;
- empty content with a reasoning field and another stop reason: `reasoning_only`;
- empty content without reasoning and otherwise valid response: `empty_no_reasoning`;
- zero choices: `missing_choices`;
- transport or provider status failure: use the corresponding error category.

Do not use hidden reasoning text as the user-visible answer. Reasoning metadata is diagnostic only.

## Phase 4: compare three transport layers

For the same model, prompt and token allowance, compare:

1. raw `httpx` request to the OpenAI-compatible endpoint;
2. `AsyncOpenAI.chat.completions.create`;
3. `OpenAICompatibleClient.acomplete` from this project.

Interpretation:

- raw fails and SDK fails: likely provider/model/request issue;
- raw succeeds but SDK loses fields: SDK compatibility issue;
- SDK succeeds but project client fails: project wrapper issue;
- only 80 tokens fail while 256/1024 succeed: output-budget issue;
- all token budgets return empty content with valid choices: provider/model behavior or unsupported response field;
- requested and reported model IDs differ: record alias/router evidence but do not guess hidden weights.

## Phase 5: improve the project provider client only after evidence

Do not patch the client based only on speculation.

Once the local matrix demonstrates the actual response shapes, update telemetry and error classification in:

```text
src/py-engine/src/agentic_circuit/providers/client.py
```

Likely useful additions, subject to evidence:

- actual `finish_reason`;
- visible-content length;
- reasoning-field presence and length;
- token usage for empty visible responses;
- explicit empty-response outcome category;
- distinction between output exhaustion and a genuinely empty provider response;
- bounded, redacted raw shape metadata for unknown response formats.

Keep the normalized final result free of hidden chain-of-thought.

Add unit tests with mocked provider responses for every response shape found locally.

## Phase 6: choose OpenAI-compatible models

The user will provide candidate model IDs or request a current web lookup.

All candidates in the immediate comparison must:

- use an OpenAI-compatible chat-completions endpoint;
- accept the same basic message structure;
- be callable with the available subscription or API key;
- support enough output tokens for selector, extraction, reader and judge roles;
- return a visible completion or a clearly documented compatible field;
- be tested independently before being used as fallback.

Evaluate primary, fallback and judge roles separately. A strong reader is not automatically a reliable strict-JSON judge.

Do not assume a primary and fallback are independent merely because their public model names differ.

## Phase 7: local provider-role matrix

Before Qdrant or datasets, run a small role matrix:

```text
memory_select
memory_extract
benchmark_reader
benchmark_judge
```

Use representative fixed prompts from the project.

For every model and role, record:

- parse success;
- visible completion success;
- average latency;
- requested and reported model ID;
- fallback activation;
- parameter retry activation;
- output-token exhaustion;
- response classification.

A candidate should not enter the full benchmark if its required structured-output role is unreliable in this small matrix.

## Phase 8: small local memory benchmark

After provider behavior is understood, start the retrieval services locally and run a very small subset first.

Suggested first subset:

```text
LongMemEval: 2
LoCoMo: 2
Internal: 6
Seed: 20260714
```

Keep unchanged:

- embeddings;
- Qdrant;
- BM25;
- weighted RRF;
- reranker;
- selector prompt;
- reader prompt;
- datasets;
- safety checks.

The first local benchmark is for plumbing and diagnostics, not model ranking.

Do not write exploratory local runs to the comparable Neon category by default. Save them locally or use a separate development category.

## Phase 9: resume the fixed benchmark and Experiment B

Only after the local provider chain is stable, run the fixed comparison:

```text
LongMemEval: 6
LoCoMo: 8
Internal: 6
Seed: 20260714
Category: live-memory-adapted-v1
```

Then continue Experiment B from `docs/MEMORY_BENCHMARK_HANDOFF.md`:

- raw retrieval recall;
- selected recall;
- selector empty rate on answerable cases;
- judge coverage;
- judge empty-response rate;
- judge parse-failure rate;
- answer language match;
- concise-answer compliance;
- requested and provider-reported model by role.

Experiment B must not silently change retrieval, selector, reader and provider at the same time.

## Acceptance criteria before changing production defaults

A provider/model chain can replace the current defaults only when all of the following are true:

1. Raw HTTP, OpenAI SDK and project client results are understood and consistent.
2. Empty responses have a specific evidenced classification.
3. Judge JSON succeeds reliably at a justified output-token budget.
4. Selector and extraction structured outputs are locally reproducible.
5. Primary and fallback behavior is measured independently.
6. Requested and provider-reported model IDs are recorded.
7. Unit tests cover every discovered compatibility case.
8. The small local benchmark completes without unexplained provider failures.
9. The fixed benchmark uses the same seed, cases and retrieval stack as the three stored reference runs.
10. All deterministic memory safety checks still pass.
11. Normal CI passes after the local work is complete.

## Do not do yet

Do not:

- blame LangGraph for the Experiment A empty judge results;
- switch all agent manifests to new models before local protocol testing;
- add Qwen/Anthropic transport in the current task;
- implement cross-protocol fallback;
- change embeddings, Qdrant, reranker or RRF;
- start RRF versus RSF work;
- add deterministic selector fallback;
- treat Token F1 as semantic correctness;
- publish local exploratory runs as official benchmark results;
- commit provider keys or raw private prompts;
- use CI as the main interactive debugger.

## Expected deliverables from the next assistant

The next assistant should produce, in order:

1. A configurable local OpenAI-compatible raw/SDK/project-client probe.
2. A redacted local report comparing token budgets and response shapes.
3. A root-cause conclusion supported by captured metadata.
4. Provider-client fixes only where the report demonstrates a need.
5. Unit tests for discovered response shapes.
6. A provider-role matrix for candidate models.
7. A small local benchmark run.
8. A recommendation for primary, fallback and judge models.
9. Only then, the full fixed benchmark and Experiment B metrics.

## Instruction to the next AI assistant

Start by checking out `agent/experiment-a-deepseek-provider`, reading this file and inspecting the current provider client and probe script.

Do not begin with GitHub Actions and do not change model defaults immediately.

First make one simple OpenAI-compatible model request locally through raw HTTP, the OpenAI SDK and the project client. Capture the response shape, token usage and finish reason without storing secrets or hidden reasoning. Then reproduce the benchmark judge prompt at 80, 256 and 1024 output tokens.

Report the first proven divergence between the three layers before modifying production code.
