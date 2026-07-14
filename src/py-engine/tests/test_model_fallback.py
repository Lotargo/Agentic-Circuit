from collections import Counter
from types import SimpleNamespace

import pytest

from agentic_circuit.config.schema import ModelConfig, Provider
from agentic_circuit.providers.client import OpenAICompatibleClient


class FakeCompletions:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
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
    client._parameter_fallback_successes = 0
    return client, completions


def config(*, thinking_level="off"):
    return ModelConfig(
        provider="provider",
        model="primary",
        fallback_models=["fallback"],
        thinking_level=thinking_level,
    )


async def test_completion_uses_fallback_after_primary_error():
    client, completions = client_with(
        [RuntimeError("primary unavailable"), response("ok", "fallback")]
    )

    result = await client.acomplete([{"role": "user", "content": "hello"}], config())

    assert result.content == "ok"
    assert result.fallback_used is True
    assert result.attempted_models == ["primary", "fallback"]
    assert "primary unavailable" in result.fallback_reason
    assert [call["model"] for call in completions.calls] == ["primary", "fallback"]
    usage = client.usage_snapshot()
    assert usage["fallback_successes"] == 1
    assert usage["fallback_reasons"] == {result.fallback_reason: 1}
    assert usage["successes_by_role_and_model"]["other"] == {"fallback": 1}


async def test_completion_retries_same_model_without_thinking_before_fallback():
    client, completions = client_with(
        [RuntimeError("unsupported thinking"), response("ok", "primary")]
    )

    result = await client.acomplete(
        [{"role": "user", "content": "hello"}],
        config(thinking_level="low"),
    )

    assert result.content == "ok"
    assert result.fallback_used is False
    assert result.parameter_fallback_used is True
    assert "unsupported thinking" in result.parameter_retry_reason
    assert [call["model"] for call in completions.calls] == ["primary", "primary"]
    assert "extra_body" in completions.calls[0]
    assert "extra_body" not in completions.calls[1]
    usage = client.usage_snapshot()
    assert usage["parameter_fallback_successes"] == 1
    assert usage["parameter_retry_reasons"] == {result.parameter_retry_reason: 1}


async def test_judge_role_records_model_and_parse_failure_sample():
    client, _ = client_with([response("not-json", "fallback")])
    judge_messages = [
        {
            "role": "user",
            "content": "Judge whether the candidate answer is semantically correct. Return JSON only.",
        }
    ]
    judge_config = ModelConfig(provider="provider", model="fallback")

    result = await client.acomplete(judge_messages, judge_config)

    assert result.content == "not-json"
    usage = client.usage_snapshot()
    assert usage["successes_by_role_and_model"]["benchmark_judge"] == {"fallback": 1}
    assert usage["judge_parse_errors"] == {"missing_json_object": 1}
    assert usage["judge_parse_failure_samples"] == [
        {
            "error": "missing_json_object",
            "model": "fallback",
            "attempted_models": ["fallback"],
            "fallback_reason": "",
            "parameter_retry_reason": "",
            "raw_response": "not-json",
        }
    ]


async def test_judge_role_counts_empty_response_without_storing_raw_response():
    client, _ = client_with([response("   ", "fallback")])
    judge_messages = [
        {
            "role": "user",
            "content": "Judge whether the candidate answer is semantically correct. Return JSON only.",
        }
    ]
    judge_config = ModelConfig(provider="provider", model="fallback")

    result = await client.acomplete(judge_messages, judge_config)

    assert result.error
    usage = client.usage_snapshot()
    assert usage["judge_empty_responses"] == 1
    assert usage["judge_parse_failure_samples"] == []


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
    assert [call["model"] for call in completions.calls] == ["primary", "fallback"]
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
    assert [call["model"] for call in completions.calls] == ["primary"]
