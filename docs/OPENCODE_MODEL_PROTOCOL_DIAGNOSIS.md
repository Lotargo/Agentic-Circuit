# OpenCode model protocol diagnosis

Last updated: 2026-07-14

The active continuation plan has moved to:

```text
docs/LOCAL_PROVIDER_DEBUG_HANDOFF.md
```

That document supersedes the earlier CI-first protocol investigation and defines a local-first workflow:

```text
raw HTTP -> OpenAI SDK -> project client -> provider-role matrix -> small local benchmark -> full benchmark
```

## Conclusions retained from the earlier investigation

- The benchmark provider calls do not pass through LangGraph.
- Experiment A did not preserve enough response metadata to prove that the model itself caused empty judge completions.
- The judge used only `max_tokens=80`, so output-budget exhaustion remains a leading hypothesis.
- Successful `big-pickle` requests were reported by the provider as `deepseek-v4-flash`; this is alias/router evidence, not proof of identical hidden weights.
- The current immediate model search is limited to OpenAI-compatible chat-completions endpoints.
- Qwen and Anthropic-compatible transport are not part of the next task unless the user changes direction.
- GitHub Actions must be used for final validation, not as the primary interactive provider debugger.

## Existing probe

The branch still contains:

```text
src/py-engine/scripts/probe_opencode_protocols.py
.github/workflows/opencode-protocol-probe.yml
```

The script may be reused for code or response-normalization ideas, but it should not dictate the next design. The next implementation should be a configurable local OpenAI-compatible probe with no hard dependency on Qwen or GitHub secrets.

See `docs/LOCAL_PROVIDER_DEBUG_HANDOFF.md` for the full task order, local commands, output classifications, acceptance criteria and instructions for the next AI assistant.
