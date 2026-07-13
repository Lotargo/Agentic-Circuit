"""Contract tests for the FastAPI engine surface."""

from fastapi.testclient import TestClient

from agentic_circuit import main


async def _fake_run(_user_input: str) -> str:
    return "готовый ответ"


def test_models_endpoint_exposes_logical_model():
    client = TestClient(main.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "agentic-circuit"


def test_non_stream_completion(monkeypatch):
    monkeypatch.setattr(main, "_run", _fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "agentic-circuit",
            "messages": [{"role": "user", "content": "Привет"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "готовый ответ"


def test_stream_uses_single_sse_prefix_and_done_marker(monkeypatch):
    monkeypatch.setattr(main, "_run", _fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Привет"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: data:" not in response.text
    assert "data: [DONE]" in response.text
    assert "chat.completion.chunk" in response.text


def test_empty_user_message_is_rejected():
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "assistant", "content": "hello"}]},
    )
    assert response.status_code == 400
