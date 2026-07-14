# OpenCode model protocol diagnosis

Date: 2026-07-14

## Question

Before replacing the current benchmark and application models with OpenCode Go models, determine whether the empty completions observed with Big Pickle, MiMo and DeepSeek came from:

1. LangChain or LangGraph compatibility;
2. the OpenAI Python SDK;
3. the OpenCode gateway protocol;
4. model-side reasoning and token-budget behavior;
5. an actual model reliability failure.

## LangGraph is not in the failing benchmark request path

The live memory benchmark constructs `ClientRegistry` directly and calls:

```python
client.acomplete(...)
```

The provider implementation calls:

```python
AsyncOpenAI.chat.completions.create(...)
```

The benchmark runner does not build or execute the LangGraph state graph. LangGraph is used by the production multi-agent circuit, but it does not transform the benchmark judge request or response.

Therefore the `0 / 14` empty judge result from Experiment A cannot be attributed to LangGraph handling a Chinese model.

The project does not depend on LangChain for model transport.

## Strong token-budget hypothesis

Experiment A used different output limits by role:

```text
benchmark reader: up to 4096 tokens
memory selector/extractor: up to 1800 tokens
benchmark judge: 80 tokens
```

Observed behavior:

- the reader succeeded on all 14 external cases;
- selector and extraction were intermittent;
- every judge request returned an empty visible completion;
- successful primary responses requested as `big-pickle` were reported by the gateway as `deepseek-v4-flash`.

This pattern is consistent with a reasoning model consuming a small output budget before producing final visible content. It is not proof, because the current telemetry does not record:

- `finish_reason`;
- raw successful response bodies;
- `reasoning_content` or equivalent provider fields;
- detailed completion-token usage for empty responses.

The current client converts an empty `message.content` into `RuntimeError("provider returned an empty completion")`, losing the distinction between:

- genuine empty provider output;
- `finish_reason=length` after hidden reasoning;
- final text stored in a nonstandard reasoning field;
- a gateway response with another content shape.

## OpenCode Zen and Go protocols are model-specific

OpenCode documents different endpoints for different model families.

### OpenAI-compatible chat completions

These models use chat completions:

```text
DeepSeek V4 Pro
DeepSeek V4 Flash
Big Pickle
MiMo
GLM
Kimi
```

Go endpoint:

```text
https://opencode.ai/zen/go/v1/chat/completions
```

Zen endpoint:

```text
https://opencode.ai/zen/v1/chat/completions
```

### Anthropic-compatible messages

These models use the Anthropic Messages protocol:

```text
Qwen3.7 Max
Qwen3.7 Plus
Qwen3.6 Plus
MiniMax M3
MiniMax M2.7
```

Go endpoint:

```text
https://opencode.ai/zen/go/v1/messages
```

Consequently, `qwen3.7-plus` cannot be inserted into the current `openai-compatible` provider chain. The current `ClientRegistry` always instantiates `OpenAICompatibleClient`, so Qwen support requires a second provider transport or an adapter that speaks the Messages API.

DeepSeek V4 Pro can use the existing OpenAI-compatible transport, but it still requires a separate Go provider base URL and Go credentials.

## Big Pickle identity

Experiment A requested `big-pickle`, while successful responses reported:

```text
deepseek-v4-flash
```

This is strong evidence that Big Pickle is routed to, aliased to, or at least metadata-compatible with DeepSeek V4 Flash at the OpenCode gateway.

It does not prove that the two public model IDs always resolve to identical weights or serving configurations. They may use:

- different snapshots;
- different quantization;
- different system templates;
- different routing pools;
- the same model family with different provider settings.

Treat them as related until a raw protocol probe shows otherwise.

## Diagnostic probe added

Files:

```text
src/py-engine/scripts/probe_opencode_protocols.py
.github/workflows/opencode-protocol-probe.yml
```

The probe bypasses:

- LangGraph;
- memory retrieval;
- selector code;
- application prompt assembly;
- the project provider wrapper for raw requests.

It tests:

```text
Zen: big-pickle
Zen: deepseek-v4-flash-free
Go: deepseek-v4-pro
Go: qwen3.7-plus
```

For OpenAI-compatible models it compares:

- raw `httpx` response;
- OpenAI SDK parsed response;
- 80-token judge request;
- 1024-token judge request;
- minimal simple request.

For Qwen it calls the native Messages endpoint and records content block types.

Recorded fields include:

- HTTP status;
- provider-reported model ID;
- finish or stop reason;
- visible content length;
- reasoning content length;
- response keys;
- usage metadata;
- bounded raw response excerpts;
- transport errors.

No API keys are written to artifacts.

## Current execution blocker

The automated probe could not call OpenCode because both pull-request and internal branch push runs received an empty:

```text
OPENCODE_ZEN_API_KEY
```

The earlier Experiment A run had access to this secret, so the repository secret is currently missing, renamed or no longer available.

Before running the manual workflow, configure:

```text
OPENCODE_ZEN_API_KEY
OPENCODE_GO_API_KEY
```

The first key is needed to reproduce Big Pickle and free DeepSeek behavior. The second key is needed for Qwen3.7 Plus and DeepSeek V4 Pro on the Go endpoints.

## Decision before model migration

Do not yet replace every role with Qwen3.7 Plus and DeepSeek V4 Pro.

First run the protocol probe and classify each failure:

1. **80 empty, 1024 non-empty, finish reason length**  
   Increase role-specific output budgets or disable reasoning where the gateway supports it.

2. **Raw response contains text, OpenAI SDK does not**  
   Fix SDK field extraction or use raw transport for that provider.

3. **Raw response stores reasoning but no final text**  
   Add explicit reasoning-field support and a minimum final-answer budget.

4. **Qwen Messages request succeeds**  
   Implement an Anthropic-compatible provider client before migrating primary roles.

5. **Raw requests are also empty at large budgets**  
   Treat it as a gateway/model reliability issue and compare another Go model.

6. **Authentication fails only on Go endpoints**  
   Correct the Go secret or account key before drawing model conclusions.

The most likely current explanation is not LangGraph. The leading hypotheses are the 80-token reasoning budget and incomplete observation of the raw OpenCode response shape.
