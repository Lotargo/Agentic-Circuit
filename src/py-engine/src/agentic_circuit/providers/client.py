"""OpenAI-compatible provider client with completion and streaming APIs."""

from __future__ import annotations

import os
import time
from collections import Counter
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
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.error = error
        self.latency_ms = latency_ms
        self.attempted_models = list(attempted_models or [model])
        self.fallback_used = fallback_used


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
        self._client = AsyncOpenAI(
            base_url=_openai_sdk_base_url(provider.base_url),
            api_key=api_key,
            timeout=120.0,
        )
        self._attempts: Counter[str] = Counter()
        self._successes: Counter[str] = Counter()
        self._failures: Counter[str] = Counter()
        self._fallback_successes = 0

    def usage_snapshot(self) -> dict:
        return {
            "attempts_by_model": dict(self._attempts),
            "successes_by_model": dict(self._successes),
            "failures_by_model": dict(self._failures),
            "fallback_successes": self._fallback_successes,
        }

    async def acomplete(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> LLMResult:
        start = time.monotonic()
        errors: list[str] = []
        attempted: list[str] = []
        for index, model_name in enumerate(model_cfg.model_chain):
            attempted.append(model_name)
            self._attempts[model_name] += 1
            params, extra = _completion_params(
                messages,
                model_cfg,
                tools,
                model=model_name,
            )
            try:
                response = await self._client.chat.completions.create(**params, **extra)
                choice = response.choices[0].message
                content = choice.content or ""
                if not content and not tools:
                    raise RuntimeError("provider returned an empty completion")
                usage = response.usage
                self._successes[model_name] += 1
                if index > 0:
                    self._fallback_successes += 1
                return LLMResult(
                    content=content,
                    model=response.model or model_name,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=int((time.monotonic() - start) * 1000),
                    attempted_models=attempted,
                    fallback_used=index > 0,
                )
            except Exception as exc:
                self._failures[model_name] += 1
                errors.append(f"{model_name}: {type(exc).__name__}: {exc}")

        return LLMResult(
            content="",
            model=attempted[-1] if attempted else model_cfg.model,
            error="; ".join(errors) or "all configured models failed",
            latency_ms=int((time.monotonic() - start) * 1000),
            attempted_models=attempted,
            fallback_used=len(attempted) > 1,
        )

    async def astream(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Yield upstream deltas and fall back only before the first emitted token."""
        errors: list[str] = []
        for index, model_name in enumerate(model_cfg.model_chain):
            self._attempts[model_name] += 1
            params, extra = _completion_params(
                messages,
                model_cfg,
                tools,
                model=model_name,
            )
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
                        if index > 0:
                            self._fallback_successes += 1
                        success_recorded = True
                    emitted = True
                    yield content
                if not emitted:
                    raise RuntimeError("provider returned an empty stream")
                return
            except Exception as exc:
                self._failures[model_name] += 1
                if emitted:
                    raise
                errors.append(f"{model_name}: {type(exc).__name__}: {exc}")

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
