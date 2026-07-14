"""Probe OpenCode Zen/Go model protocols without LangGraph or application prompts.

The probe deliberately uses both raw HTTP and the OpenAI Python SDK for
OpenAI-compatible endpoints. It records response shape and bounded excerpts,
never API keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

OUTPUT_DIR = Path(os.environ.get("PROBE_OUTPUT_DIR", "protocol-probe-results"))
ZEN_CHAT_URL = "https://opencode.ai/zen/v1/chat/completions"
GO_CHAT_URL = "https://opencode.ai/zen/go/v1/chat/completions"
GO_MESSAGES_URL = "https://opencode.ai/zen/go/v1/messages"

JUDGE_PROMPT = (
    "Judge whether the candidate answer is semantically correct for the reference answer. "
    "Allow equivalent wording. Return JSON only: {\"correct\":true}.\n\n"
    "Question: Which database was selected?\n"
    "Reference: Neon\n"
    "Candidate: The selected database is Neon."
)
SIMPLE_PROMPT = "Reply with exactly OK and nothing else."


@dataclass
class ProbeResult:
    provider: str
    protocol: str
    transport: str
    model: str
    case: str
    max_tokens: int
    request_profile: str
    status_code: int | None = None
    elapsed_ms: int = 0
    response_model: str = ""
    finish_reason: str = ""
    content: str = ""
    content_length: int = 0
    reasoning_content: str = ""
    reasoning_length: int = 0
    message_keys: list[str] | None = None
    top_level_keys: list[str] | None = None
    usage: dict[str, Any] | None = None
    error: str = ""
    raw_excerpt: str = ""


def _safe_excerpt(value: object, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:limit]


def _first_choice(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}, {}
    choice = choices[0]
    message = choice.get("message")
    return choice, message if isinstance(message, dict) else {}


def _extract_reasoning(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return _safe_excerpt(value, limit=2000)
    return ""


async def raw_openai_probe(
    client: httpx.AsyncClient,
    *,
    provider: str,
    url: str,
    api_key: str,
    model: str,
    case: str,
    prompt: str,
    max_tokens: int,
    minimal: bool,
) -> ProbeResult:
    profile = "minimal" if minimal else "benchmark-like"
    result = ProbeResult(
        provider=provider,
        protocol="chat/completions",
        transport="raw-httpx",
        model=model,
        case=case,
        max_tokens=max_tokens,
        request_profile=profile,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if not minimal:
        payload.update({"temperature": 0.0, "top_p": 0.1})
    started = time.monotonic()
    try:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        result.status_code = response.status_code
        result.raw_excerpt = _safe_excerpt(response.text)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("response JSON is not an object")
        choice, message = _first_choice(data)
        content = message.get("content")
        result.content = content if isinstance(content, str) else _safe_excerpt(content or "")
        result.content_length = len(result.content)
        result.reasoning_content = _extract_reasoning(message)
        result.reasoning_length = len(result.reasoning_content)
        result.response_model = str(data.get("model") or "")
        result.finish_reason = str(choice.get("finish_reason") or "")
        result.message_keys = sorted(message)
        result.top_level_keys = sorted(data)
        result.usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    except Exception as exc:
        result.elapsed_ms = result.elapsed_ms or int((time.monotonic() - started) * 1000)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


async def sdk_openai_probe(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> ProbeResult:
    result = ProbeResult(
        provider=provider,
        protocol="chat/completions",
        transport="openai-sdk",
        model=model,
        case="judge",
        max_tokens=max_tokens,
        request_profile="benchmark-like",
    )
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    started = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT}],
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.1,
        )
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        result.response_model = response.model or ""
        if response.choices:
            choice = response.choices[0]
            result.finish_reason = str(choice.finish_reason or "")
            message = choice.message
            result.content = message.content or ""
            result.content_length = len(result.content)
            extra = getattr(message, "model_extra", None) or {}
            if isinstance(extra, dict):
                result.message_keys = sorted(set(message.model_fields_set) | set(extra))
                result.reasoning_content = _extract_reasoning(extra)
                result.reasoning_length = len(result.reasoning_content)
        usage = response.usage
        result.usage = usage.model_dump() if usage is not None else None
        result.top_level_keys = sorted(response.model_fields_set)
        result.raw_excerpt = _safe_excerpt(response.model_dump(mode="json"))
    except Exception as exc:
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        await client.close()
    return result


async def raw_anthropic_probe(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model: str,
    case: str,
    prompt: str,
    max_tokens: int,
) -> ProbeResult:
    result = ProbeResult(
        provider="opencode-go",
        protocol="messages",
        transport="raw-httpx",
        model=model,
        case=case,
        max_tokens=max_tokens,
        request_profile="native-anthropic",
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    started = time.monotonic()
    try:
        response = await client.post(
            GO_MESSAGES_URL,
            headers={
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        result.status_code = response.status_code
        result.raw_excerpt = _safe_excerpt(response.text)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("response JSON is not an object")
        blocks = data.get("content")
        texts: list[str] = []
        reasoning: list[str] = []
        block_types: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "")
                block_types.append(block_type)
                text = block.get("text")
                if isinstance(text, str):
                    if block_type in {"thinking", "reasoning"}:
                        reasoning.append(text)
                    else:
                        texts.append(text)
        result.content = "".join(texts)
        result.content_length = len(result.content)
        result.reasoning_content = "".join(reasoning)
        result.reasoning_length = len(result.reasoning_content)
        result.response_model = str(data.get("model") or "")
        result.finish_reason = str(data.get("stop_reason") or "")
        result.message_keys = block_types
        result.top_level_keys = sorted(data)
        result.usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
    except Exception as exc:
        result.elapsed_ms = result.elapsed_ms or int((time.monotonic() - started) * 1000)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def render_markdown(results: list[ProbeResult]) -> str:
    lines = [
        "# OpenCode model protocol probe",
        "",
        "This probe bypasses LangGraph and the application provider wrapper.",
        "",
        "| Provider | Protocol | Transport | Model | Case | Max tokens | Profile | HTTP | Response model | Finish | Content | Reasoning | Error |",
        "|---|---|---|---|---|---:|---|---:|---|---|---:|---:|---|",
    ]
    for item in results:
        error = item.error.replace("|", "\\|")[:120]
        lines.append(
            f"| {item.provider} | {item.protocol} | {item.transport} | `{item.model}` | "
            f"{item.case} | {item.max_tokens} | {item.request_profile} | "
            f"{item.status_code if item.status_code is not None else '-'} | "
            f"`{item.response_model or '-'}` | `{item.finish_reason or '-'}` | "
            f"{item.content_length} | {item.reasoning_length} | {error or '-'} |"
        )
    lines.extend(["", "## Response shapes", ""])
    for index, item in enumerate(results, 1):
        lines.extend(
            [
                f"### {index}. {item.provider} / {item.model} / {item.transport} / {item.case} / {item.max_tokens}",
                "",
                f"- Message or block keys: `{item.message_keys or []}`",
                f"- Top-level keys: `{item.top_level_keys or []}`",
                f"- Usage: `{json.dumps(item.usage, ensure_ascii=False)}`",
                f"- Content excerpt: `{_safe_excerpt(item.content, 300)}`",
                f"- Reasoning excerpt: `{_safe_excerpt(item.reasoning_content, 300)}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


async def main() -> int:
    zen_key = os.environ.get("OPENCODE_ZEN_API_KEY", "")
    go_key = os.environ.get("OPENCODE_GO_API_KEY", "") or zen_key
    if not zen_key:
        raise RuntimeError("OPENCODE_ZEN_API_KEY is required")

    results: list[ProbeResult] = []
    async with httpx.AsyncClient(timeout=120.0) as http:
        for model in ("big-pickle", "deepseek-v4-flash-free"):
            results.append(
                await raw_openai_probe(
                    http,
                    provider="opencode-zen",
                    url=ZEN_CHAT_URL,
                    api_key=zen_key,
                    model=model,
                    case="simple",
                    prompt=SIMPLE_PROMPT,
                    max_tokens=80,
                    minimal=True,
                )
            )
            results.append(
                await raw_openai_probe(
                    http,
                    provider="opencode-zen",
                    url=ZEN_CHAT_URL,
                    api_key=zen_key,
                    model=model,
                    case="judge",
                    prompt=JUDGE_PROMPT,
                    max_tokens=80,
                    minimal=False,
                )
            )
            results.append(
                await sdk_openai_probe(
                    provider="opencode-zen",
                    base_url="https://opencode.ai/zen/v1",
                    api_key=zen_key,
                    model=model,
                    max_tokens=80,
                )
            )
            results.append(
                await raw_openai_probe(
                    http,
                    provider="opencode-zen",
                    url=ZEN_CHAT_URL,
                    api_key=zen_key,
                    model=model,
                    case="judge",
                    prompt=JUDGE_PROMPT,
                    max_tokens=1024,
                    minimal=True,
                )
            )

        results.append(
            await raw_openai_probe(
                http,
                provider="opencode-go",
                url=GO_CHAT_URL,
                api_key=go_key,
                model="deepseek-v4-pro",
                case="simple",
                prompt=SIMPLE_PROMPT,
                max_tokens=80,
                minimal=True,
            )
        )
        results.append(
            await raw_openai_probe(
                http,
                provider="opencode-go",
                url=GO_CHAT_URL,
                api_key=go_key,
                model="deepseek-v4-pro",
                case="judge",
                prompt=JUDGE_PROMPT,
                max_tokens=80,
                minimal=False,
            )
        )
        results.append(
            await sdk_openai_probe(
                provider="opencode-go",
                base_url="https://opencode.ai/zen/go/v1",
                api_key=go_key,
                model="deepseek-v4-pro",
                max_tokens=80,
            )
        )
        results.append(
            await raw_openai_probe(
                http,
                provider="opencode-go",
                url=GO_CHAT_URL,
                api_key=go_key,
                model="deepseek-v4-pro",
                case="judge",
                prompt=JUDGE_PROMPT,
                max_tokens=1024,
                minimal=True,
            )
        )

        for case, prompt, max_tokens in (
            ("simple", SIMPLE_PROMPT, 80),
            ("judge", JUDGE_PROMPT, 80),
            ("judge", JUDGE_PROMPT, 1024),
        ):
            results.append(
                await raw_anthropic_probe(
                    http,
                    api_key=go_key,
                    model="qwen3.7-plus",
                    case=case,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "go_key_source": (
            "OPENCODE_GO_API_KEY"
            if os.environ.get("OPENCODE_GO_API_KEY")
            else "OPENCODE_ZEN_API_KEY fallback"
        ),
        "results": [asdict(item) for item in results],
    }
    (OUTPUT_DIR / "opencode-protocol-probe.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(results)
    (OUTPUT_DIR / "opencode-protocol-probe.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
