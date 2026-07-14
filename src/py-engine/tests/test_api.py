"""Contract tests for the FastAPI engine surface."""

from fastapi.testclient import TestClient

from agentic_circuit import main
from agentic_circuit.memory import MemoryContext


async def _fake_run(
    conversation: list[dict],
    prism: str,
    memory_context: MemoryContext,
) -> str:
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    assert memory_context.scope.startswith("user:")
    assert memory_context.workspace_id.startswith("workspace:")
    assert memory_context.project_id.startswith("project:")
    assert memory_context.conversation_id.startswith("conversation:")
    return "готовый ответ"


async def _fake_stream(
    conversation: list[dict],
    prism: str,
    memory_context: MemoryContext,
):
    assert conversation[0]["content"] == "Меня зовут Олег"
    assert conversation[-1]["content"] == "Как меня зовут?"
    assert prism == "joy"
    assert memory_context.scope.startswith("user:")
    yield "готовый "
    yield "ответ"


def test_models_endpoint_exposes_logical_model():
    client = TestClient(main.app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "agentic-circuit"


def test_non_stream_completion_preserves_history_and_namespaces(monkeypatch):
    monkeypatch.setattr(main, "_run", _fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-OpenWebUI-User-Id": "oleg",
            "X-OpenWebUI-Workspace-Id": "portfolio",
            "X-Project-Id": "chat-openwebui",
            "X-OpenWebUI-Chat-Id": "chat-42",
        },
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
    async def fake_run(conversation, prism, memory_context):
        assert memory_context == MemoryContext()
        return "без памяти"

    monkeypatch.setattr(main, "_run", fake_run)
    client = TestClient(main.app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200


def test_memory_false_disables_all_namespaces():
    client = TestClient(main.app)
    request = client.build_request(
        "POST",
        "/",
        headers={"X-OpenWebUI-User-Id": "oleg", "X-Project-Id": "secret"},
        json={},
    )
    assert main._memory_context(request, {"memory": False}) == MemoryContext()


def test_equivalent_metadata_creates_stable_opaque_namespaces():
    client = TestClient(main.app)
    request = client.build_request("POST", "/", json={})
    context_a = main._memory_context(
        request,
        {
            "user": "same-user",
            "project_id": "project-a",
            "conversation_id": "chat-a",
        },
    )
    context_b = main._memory_context(
        request,
        {
            "metadata": {
                "user_id": "same-user",
                "project_id": "project-a",
                "conversation_id": "chat-a",
            }
        },
    )
    assert context_a == context_b
    assert context_a.scope.startswith("user:")
    assert context_a.project_id.startswith("project:")
    assert context_a.conversation_id.startswith("conversation:")
    assert "same-user" not in repr(context_a)
    assert "project-a" not in repr(context_a)


def test_projects_are_isolated_inside_same_user_scope():
    client = TestClient(main.app)
    request = client.build_request("POST", "/", json={})
    first = main._memory_context(request, {"user": "oleg", "project_id": "one"})
    second = main._memory_context(request, {"user": "oleg", "project_id": "two"})
    assert first.scope == second.scope
    assert first.project_id != second.project_id


def test_same_user_is_isolated_between_workspaces():
    client = TestClient(main.app)
    first_request = client.build_request(
        "POST",
        "/",
        headers={
            "X-OpenWebUI-User-Id": "oleg",
            "X-OpenWebUI-Workspace-Id": "workspace-a",
        },
        json={},
    )
    second_request = client.build_request(
        "POST",
        "/",
        headers={
            "X-OpenWebUI-User-Id": "oleg",
            "X-OpenWebUI-Workspace-Id": "workspace-b",
        },
        json={},
    )
    first = main._memory_context(first_request, {})
    second = main._memory_context(second_request, {})
    assert first.workspace_id != second.workspace_id
    assert first.scope != second.scope


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
