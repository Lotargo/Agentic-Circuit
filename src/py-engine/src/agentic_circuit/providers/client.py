"""OpenAI-compatible provider client with agent parameter mapping."""

from __future__ import annotations

import os
import time
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
    ):
        self.content = content
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.error = error
        self.latency_ms = latency_ms

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<LLMResult model={self.model} tokens={self.completion_tokens} err={self.error}>"


_THINKING_TO_EXTRA = {
    "off": None,
    "low": {"thinking": {"type": "enabled", "budget": 2048}},
    "medium": {"thinking": {"type": "enabled", "budget": 8192}},
    "high": {"thinking": {"type": "enabled", "budget": 16384}},
}


class ProviderClient(Protocol):
    async def acomplete(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> LLMResult: ...


class OpenAICompatibleClient:
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, provider: Provider):
        api_key = os.environ.get(provider.api_key_env, provider.api_key_env)
        self.provider = provider
        self._client = AsyncOpenAI(
            base_url=provider.base_url,
            api_key=api_key,
            timeout=120.0,
        )

    async def acomplete(
        self,
        messages: list[dict],
        model_cfg: ModelConfig,
        tools: Optional[list[dict]] = None,
    ) -> LLMResult:
        extra: dict = {}
        thinking = _THINKING_TO_EXTRA.get(model_cfg.thinking_level)
        if thinking:
            extra["extra_body"] = thinking
        start = time.monotonic()
        try:
            params = {
                "model": model_cfg.model,
                "messages": messages,
                "temperature": model_cfg.temperature,
                "max_tokens": model_cfg.max_tokens,
                "top_p": model_cfg.top_p,
            }
            if tools:
                params["tools"] = tools
            resp = await self._client.chat.completions.create(**params, **extra)
            choice = resp.choices[0].message
            usage = resp.usage
            return LLMResult(
                content=choice.content or "",
                model=resp.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=int((time.monotonic() - start) * 1000),
            )
        except Exception as exc:  # surface as structured error, do not crash graph
            return LLMResult(
                content="",
                model=model_cfg.model,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.monotonic() - start) * 1000),
            )


class ClientRegistry:
    """Lazily builds and caches one client per provider."""

    def __init__(self, providers: dict[str, Provider]):
        self._providers = providers
        self._cache: dict[str, OpenAICompatibleClient] = {}

    def get(self, name: str) -> OpenAICompatibleClient:
        if name not in self._cache:
            if name not in self._providers:
                raise KeyError(f"Unknown provider: {name}")
            self._cache[name] = OpenAICompatibleClient(self._providers[name])
        return self._cache[name]
