"""Contract tests for the FastAPI engine surface."""

from fastapi.testclient import TestClient

from agentic_circuit import main


async def _fake_run(conversation: list[dict], prism: str) -> str:
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    return "готовый ответ"


async def _fake_stream(conversation: list[dict], prism: str):
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    yield "готовый "
    yield "ответ"


def test_models_endpoint_exposes_logical_model():
    client = TestClient(main.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "agentic-circuit"


def test_non_stream_completion_preserves_history_and_prism(monkeypatch):
    monkeypatch.setattr(main, "_run", _fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
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
