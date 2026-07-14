"""OpenAI-compatible provider client with completion and streaming APIs."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from typing import Optional, Protocol

from openai import AsyncOpenAI

from ..config.schema import ModelConfig, Provider


class LLMResult:
    def __init__(
        self,
        content: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: Optional[str] = None,
        latency_ms: int = 0,
        attempted_models: Optional[list[str]] = None,
        fallback_used: bool = False,
        parameter_fallback_used: bool = False,
        fallback_reason: str = "",
        parameter_retry_reason: str = "",
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.error = error
        self.latency_ms = latency_ms
        self.attempted_models = list(attempted_models or [model])
        self.fallback_used = fallback_used
        self.parameter_fallback_used = parameter_fallback_used
        self.fallback_reason = fallback_reason
        self.parameter_retry_reason = parameter_retry_reason


_THINKING_TO_EXTRA = {
    "off": None,
    "low": {"thinking": {"type": "enabled", "budget": 2048}},
    "medium": {"thinking": {"type": "enabled", "budget": 8192}},
    "high": {"thinking": {"type": "enabled", "budget": 16384}},
}


def _openai_sdk_base_url(configured_url: str) -> str:
    url = configured_url.rstrip("/")
    suffix = "/chat/completions"
    return url[: -len(suffix)] if url.endswith(suffix) else url


def _completion_params(
    messages: list[dict],
    model_cfg: ModelConfig,
    tools: Optional[list[dict]] = None,
    *,
    model: str | None = None,
) -> tuple[dict, dict]:
    params: dict = {
        "model": model or model_cfg.model,
        "messages": messages,
        "temperature": model_cfg.temperature,
        "max_tokens": model_cfg.max_tokens,
        "top_p": model_cfg.top_p,
    }
    if tools:
        params["tools"] = tools
    extra: dict = {}
    thinking = _THINKING_TO_EXTRA.get(model_cfg.thinking_level)
    if thinking:
        extra["extra_body"] = thinking
    return params, extra


def _request_profiles(extra: dict) -> list[tuple[str, dict]]:
    """Try configured provider extensions first, then plain OpenAI parameters."""
    profiles = [("configured", extra)]
    if extra:
        profiles.append(("plain", {}))
    return profiles


def _request_role(messages: list[dict]) -> str:
    content = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )
    if "MEMORY_SELECT mode." in content:
        return "memory_select"
    if "MEMORY_EXTRACT mode." in content:
        return "memory_extract"
    if "Benchmark protocol:" in content:
        return "benchmark_reader"
    if "Judge whether the candidate answer is semantically correct" in content:
        return "benchmark_judge"
    return "other"


def _judge_parse_error(content: str) -> str:
    text = content.strip()
    if not text:
        return "empty_response"
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return "missing_json_object"
    try:
        parsed = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"invalid_json:{type(exc).__name__}"
    if not isinstance(parsed, dict):
        return "json_not_object"
    if not isinstance(parsed.get("correct"), bool):
        return "correct_not_boolean"
    return ""


class ProviderClient(Protocol):
    async def acomplete(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> LLMResult: ...

    def astream(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleClient:
    def __init__(self, provider: Provider):
        api_key = os.environ.get(provider.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Provider API key is missing: set environment variable {provider.api_key_env}"
            )
        self.provider = provider
        self._api_key = api_key
        self._client = AsyncOpenAI(
            base_url=_openai_sdk_base_url(provider.base_url),
            api_key=api_key,
            timeout=120.0,
        )
        self._initialize_telemetry()

    def _initialize_telemetry(self) -> None:
        self._attempts: Counter[str] = Counter()
        self._successes: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        self._fallback_successes = 0
        self._parameter_fallback_successes = 0
        self._role_attempts: dict[str, Counter[str]] = defaultdict(Counter)
        self._role_successes: dict[str, Counter[str]] = defaultdict(Counter)
        self._role_failures: dict[str, Counter[str]] = defaultdict(Counter)
        self._role_fallback_successes: Counter[str] = Counter()
        self._role_parameter_fallback_successes: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._parameter_retry_reasons: Counter[str] = Counter()
        self._judge_request_errors: Counter[str] = Counter()
        self._judge_empty_responses = 0
        self._judge_parse_errors: Counter[str] = Counter()
        self._judge_parse_failure_samples: list[dict] = []

    def _ensure_telemetry(self) -> None:
        if not hasattr(self, "_role_attempts"):
            self._initialize_telemetry()

    def _safe_text(self, value: object, limit: int = 1200) -> str:
        text = str(value)
        secret = getattr(self, "_api_key", "")
        if secret:
            text = text.replace(secret, "[redacted]")
        return text[:limit]

    def _failure_text(self, model_name: str, profile_name: str, exc: Exception) -> str:
        return self._safe_text(
            f"{model_name}/{profile_name}: {type(exc).__name__}: {exc}"
        )

    def _record_judge_response(
        self,
        *,
        content: str,
        model: str,
        attempted_models: list[str],
        fallback_reason: str,
        parameter_retry_reason: str,
    ) -> None:
        error = _judge_parse_error(content)
        if not error:
            return
        if error == "empty_response":
            self._judge_empty_responses += 1
            return
        self._judge_parse_errors[error] += 1
        if len(self._judge_parse_failure_samples) < 12:
            self._judge_parse_failure_samples.append(
                {
                    "error": error,
                    "model": model,
                    "attempted_models": list(attempted_models),
                    "fallback_reason": fallback_reason,
                    "parameter_retry_reason": parameter_retry_reason,
                    "raw_response": self._safe_text(content, limit=2000),
                }
            )

    def usage_snapshot(self) -> dict:
        self._ensure_telemetry()
        return {
            "attempts_by_model": dict(self._attempts),
            "successes_by_model": dict(self._successes),
            "failures_by_model": dict(self._failures),
            "fallback_successes": self._fallback_successes,
            "parameter_fallback_successes": self._parameter_fallback_successes,
            "attempts_by_role_and_model": {
                role: dict(counts) for role, counts in self._role_attempts.items()
            },
            "successes_by_role_and_model": {
                role: dict(counts) for role, counts in self._role_successes.items()
            },
            "failures_by_role_and_model": {
                role: dict(counts) for role, counts in self._role_failures.items()
            },
            "fallback_successes_by_role": dict(self._role_fallback_successes),
            "parameter_fallback_successes_by_role": dict(
                self._role_parameter_fallback_successes
            ),
            "fallback_reasons": dict(self._fallback_reasons),
            "parameter_retry_reasons": dict(self._parameter_retry_reasons),
            "judge_request_errors": dict(self._judge_request_errors),
            "judge_empty_responses": self._judge_empty_responses,
            "judge_parse_errors": dict(self._judge_parse_errors),
            "judge_parse_failure_samples": list(self._judge_parse_failure_samples),
        }

    async def acomplete(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> LLMResult:
        self._ensure_telemetry()
        start = time.monotonic()
        role = _request_role(messages)
        errors: list[str] = []
        attempted: list[str] = []
        model_errors: dict[str, list[str]] = defaultdict(list)
        for model_index, model_name in enumerate(model_cfg.model_chain):
            attempted.append(model_name)
            params, configured_extra = _completion_params(
                messages,
                model_cfg,
                tools,
                model=model_name,
            )
            configured_failure = ""
            for profile_index, (profile_name, extra) in enumerate(
                _request_profiles(configured_extra)
            ):
                self._attempts[model_name] += 1
                self._role_attempts[role][model_name] += 1
                try:
                    response = await self._client.chat.completions.create(**params, **extra)
                    choice = response.choices[0].message
                    content = choice.content or ""
                    if not content.strip() and not tools:
                        raise RuntimeError("provider returned an empty completion")
                    usage = response.usage
                    actual_model = response.model or model_name
                    fallback_reason = " | ".join(
                        error
                        for previous_model in model_cfg.model_chain[:model_index]
                        for error in model_errors.get(previous_model, [])
                    )
                    parameter_retry_reason = (
                        configured_failure if profile_index > 0 else ""
                    )
                    self._successes[model_name] += 1
                    self._role_successes[role][actual_model] += 1
                    if model_index > 0:
                        self._fallback_successes += 1
                        self._role_fallback_successes[role] += 1
                        if fallback_reason:
                            self._fallback_reasons[fallback_reason] += 1
                    if profile_index > 0:
                        self._parameter_fallback_successes += 1
                        self._role_parameter_fallback_successes[role] += 1
                        if parameter_retry_reason:
                            self._parameter_retry_reasons[parameter_retry_reason] += 1
                    if role == "benchmark_judge":
                        self._record_judge_response(
                            content=content,
                            model=actual_model,
                            attempted_models=attempted,
                            fallback_reason=fallback_reason,
                            parameter_retry_reason=parameter_retry_reason,
                        )
                    return LLMResult(
                        content=content,
                        model=actual_model,
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        latency_ms=int((time.monotonic() - start) * 1000),
                        attempted_models=attempted,
                        fallback_used=model_index > 0,
                        parameter_fallback_used=profile_index > 0,
                        fallback_reason=fallback_reason,
                        parameter_retry_reason=parameter_retry_reason,
                    )
                except Exception as exc:
                    failure = self._failure_text(model_name, profile_name, exc)
                    self._failures[model_name] += 1
                    self._role_failures[role][model_name] += 1
                    errors.append(failure)
                    model_errors[model_name].append(failure)
                    if profile_name == "configured":
                        configured_failure = failure
                    if role == "benchmark_judge":
                        if "empty completion" in failure:
                            self._judge_empty_responses += 1
                        else:
                            self._judge_request_errors[type(exc).__name__] += 1

        fallback_reason = " | ".join(
            error
            for previous_model in model_cfg.model_chain[:-1]
            for error in model_errors.get(previous_model, [])
        )
        last_model = attempted[-1] if attempted else model_cfg.model
        parameter_retry_reason = next(
            (
                error
                for error in reversed(model_errors.get(last_model, []))
                if "/configured:" in error
            ),
            "",
        )
        return LLMResult(
            content="",
            model=last_model,
            error="; ".join(errors) or "all configured models failed",
            latency_ms=int((time.monotonic() - start) * 1000),
            attempted_models=attempted,
            fallback_used=len(attempted) > 1,
            parameter_fallback_used=any("/plain:" in error for error in errors),
            fallback_reason=fallback_reason,
            parameter_retry_reason=parameter_retry_reason,
        )

    async def astream(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Fall back only before the first emitted token to avoid duplicate output."""
        self._ensure_telemetry()
        role = _request_role(messages)
        errors: list[str] = []
        model_errors: dict[str, list[str]] = defaultdict(list)
        for model_index, model_name in enumerate(model_cfg.model_chain):
            params, configured_extra = _completion_params(
                messages,
                model_cfg,
                tools,
                model=model_name,
            )
            configured_failure = ""
            for profile_index, (profile_name, extra) in enumerate(
                _request_profiles(configured_extra)
            ):
                self._attempts[model_name] += 1
                self._role_attempts[role][model_name] += 1
                emitted = False
                success_recorded = False
                try:
                    stream = await self._client.chat.completions.create(
                        **params,
                        **extra,
                        stream=True,
                    )
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if not content:
                            continue
                        if not success_recorded:
                            self._successes[model_name] += 1
                            self._role_successes[role][model_name] += 1
                            if model_index > 0:
                                self._fallback_successes += 1
                                self._role_fallback_successes[role] += 1
                                fallback_reason = " | ".join(
                                    error
                                    for previous_model in model_cfg.model_chain[:model_index]
                                    for error in model_errors.get(previous_model, [])
                                )
                                if fallback_reason:
                                    self._fallback_reasons[fallback_reason] += 1
                            if profile_index > 0:
                                self._parameter_fallback_successes += 1
                                self._role_parameter_fallback_successes[role] += 1
                                if configured_failure:
                                    self._parameter_retry_reasons[configured_failure] += 1
                            success_recorded = True
                        emitted = True
                        yield content
                    if not emitted:
                        raise RuntimeError("provider returned an empty stream")
                    return
                except Exception as exc:
                    failure = self._failure_text(model_name, profile_name, exc)
                    self._failures[model_name] += 1
                    self._role_failures[role][model_name] += 1
                    if emitted:
                        raise
                    errors.append(failure)
                    model_errors[model_name].append(failure)
                    if profile_name == "configured":
                        configured_failure = failure

        raise RuntimeError("; ".join(errors) or "all configured models failed")

    async def aclose(self) -> None:
        await self._client.close()


class ClientRegistry:
    def __init__(self, providers: dict[str, Provider]):
        self._providers = providers
        self._cache: dict[str, OpenAICompatibleClient] = {}

    def get(self, name: str) -> OpenAICompatibleClient:
        if name not in self._cache:
            if name not in self._providers:
                raise KeyError(f"Unknown provider: {name}")
            self._cache[name] = OpenAICompatibleClient(self._providers[name])
        return self._cache[name]

    def usage_snapshot(self) -> dict[str, dict]:
        return {
            provider_name: client.usage_snapshot()
            for provider_name, client in self._cache.items()
        }

    async def aclose(self) -> None:
        clients = list(self._cache.values())
        self._cache.clear()
        for client in clients:
            await client.aclose()
