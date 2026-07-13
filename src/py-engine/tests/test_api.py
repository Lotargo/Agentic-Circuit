"""Contract tests for the FastAPI engine surface."""

from fastapi.testclient import TestClient

from agentic_circuit import main


async def _fake_run(
    conversation: list[dict],
    prism: str,
    memory_scope: str,
) -> str:
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    assert memory_scope.startswith("user:")
    return "готовый ответ"


async def _fake_stream(
    conversation: list[dict],
    prism: str,
    memory_scope: str,
):
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    assert memory_scope.startswith("user:")
    yield "готовый "
    yield "ответ"


def test_models_endpoint_exposes_logical_model():
    client = TestClient(main.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "agentic-circuit"


def test_non_stream_completion_preserves_history_prism_and_user_scope(monkeypatch):
    monkeypatch.setattr(main, "_run", _fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        headers={"X-OpenWebUI-User-Id": "oleg"},
        json={
            "model": "agentic-circuit",
            "prism": "joy",
            "messages": [
                {"role": "user", "content": "Меня зовут Олег"},
                {"role": "assistant", "content": "Запомнила"},
                {"role": "user", "content": "Как меня зовут?"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "готовый ответ"


def test_stream_forwards_each_token_as_its_own_sse_chunk(monkeypatch):
    monkeypatch.setattr(main, "_stream", _fake_stream)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        headers={"X-OpenWebUI-User-Id": "oleg"},
        json={
            "stream": True,
            "prism": "joy",
            "messages": [
                {"role": "user", "content": "Меня зовут Олег"},
                {"role": "assistant", "content": "Запомнила"},
                {"role": "user", "content": "Как меня зовут?"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: data:" not in response.text
    assert "data: [DONE]" in response.text
    assert response.text.count("chat.completion.chunk") == 3
    assert "готовый " in response.text
    assert "ответ" in response.text


def test_missing_user_identity_disables_persistent_memory(monkeypatch):
    async def fake_run(conversation, prism, memory_scope):
        assert memory_scope == ""
        return "без памяти"

    monkeypatch.setattr(main, "_run", fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200


def test_standard_openai_user_field_creates_stable_scope():
    client = TestClient(main.app)
    request = client.build_request("POST", "/", json={})
    # Test the helper through lightweight Starlette requests is unnecessary here;
    # deterministic hashing is covered by equal calls with equivalent metadata.
    scope_a = main._memory_scope(
        request,
        {"user": "same-user"},
    )
    scope_b = main._memory_scope(
        request,
        {"metadata": {"user_id": "same-user"}},
    )
    assert scope_a == scope_b
    assert scope_a.startswith("user:")
    assert "same-user" not in scope_a


def test_invalid_final_message_is_rejected():
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "assistant", "content": "hello"}]},
    )
    assert response.status_code == 400


def test_invalid_prism_is_rejected():
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "prism": "unknown",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 400
