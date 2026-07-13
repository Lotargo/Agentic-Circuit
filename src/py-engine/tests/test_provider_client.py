import pytest

from agentic_circuit.config.schema import Provider
from agentic_circuit.providers.client import (
    OpenAICompatibleClient,
    _openai_sdk_base_url,
)


def test_full_chat_endpoint_is_normalized_for_openai_sdk():
    assert (
        _openai_sdk_base_url("https://opencode.ai/zen/v1/chat/completions")
        == "https://opencode.ai/zen/v1"
    )
    assert _openai_sdk_base_url("https://example.test/v1") == "https://example.test/v1"


def test_missing_api_key_fails_with_clear_error(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    provider = Provider(
        base_url="https://example.test/v1/chat/completions",
        api_key_env="MISSING_TEST_KEY",
        models=["test-model"],
    )
    with pytest.raises(RuntimeError, match="MISSING_TEST_KEY"):
        OpenAICompatibleClient(provider)
