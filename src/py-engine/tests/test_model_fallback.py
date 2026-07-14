from collections import Counter
from types import SimpleNamespace

import pytest

from agentic_circuit.config.schema import ModelConfig, Provider
from agentic_circuit.providers.client import OpenAICompatibleClient


class FakeCompletions:
    def __init__(self, values):
        self.values = list(values)
        self.models = []

    async def create(self, **kwargs):
        self.models.append(kwargs["model"])
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class AsyncChunks:
    def __init__(self, values):
        self.values = values

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for value in self.values:
            if isinstance(value, Exception):
                raise value
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=value))]
            )


def response(content: str, model: str):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def client_with(values):
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.provider = Provider(
        base_url="https://example.invalid/v1",
        api_key_env="TEST_KEY",
        models=["primary", "fallback"],
    )
    completions = FakeCompletions(values)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client._attempts = Counter()
    client._successes = Counter()
    client._failures = Counter()
    client._fallback_successes = 0
    return client, completions


def config():
    return ModelConfig(
        provider="provider",
        model="primary",
        fallback_models=["fallback"],
        thinking_level="off",
    )


async def test_completion_uses_fallback_after_primary_error():
    client, completions = client_with(
        [RuntimeError("primary unavailable"), response("ok", "fallback")]
    )

    result = await client.acomplete([{"role": "user", "content": "hello"}], config())

    assert result.content == "ok"
    assert result.fallback_used is True
    assert result.attempted_models == ["primary", "fallback"]
    assert completions.models == ["primary", "fallback"]
    assert client.usage_snapshot()["fallback_successes"] == 1


async def test_stream_uses_fallback_only_before_first_token():
    client, completions = client_with(
        [RuntimeError("primary unavailable"), AsyncChunks(["a", "b"])]
    )

    chunks = [
        chunk
        async for chunk in client.astream(
            [{"role": "user", "content": "hello"}], config()
        )
    ]

    assert chunks == ["a", "b"]
    assert completions.models == ["primary", "fallback"]
    assert client.usage_snapshot()["fallback_successes"] == 1


async def test_stream_does_not_restart_after_partial_output():
    client, completions = client_with(
        [AsyncChunks(["partial", RuntimeError("connection lost")]), AsyncChunks(["new"])]
    )
    received = []

    with pytest.raises(RuntimeError, match="connection lost"):
        async for chunk in client.astream(
            [{"role": "user", "content": "hello"}], config()
        ):
            received.append(chunk)

    assert received == ["partial"]
    assert completions.models == ["primary"]
