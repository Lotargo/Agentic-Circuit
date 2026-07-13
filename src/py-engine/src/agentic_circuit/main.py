"""FastAPI entrypoint for the Python LangGraph engine.

The TS gateway forwards OpenAI-compatible requests here. The engine returns a
small but valid subset of the OpenAI chat-completions contract, including SSE
frames for streaming clients such as OpenWebUI.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import get_config
from .graph import CompositeMemory, EngineContext, build_graph
from .providers import ClientRegistry
from .rag import EmbeddingClient, RerankClient, VectorMemory
from .tools import WebSearchTool

app = FastAPI(title="agentic-circuit-engine", version="0.1.0")
LOGICAL_MODEL_ID = "agentic-circuit"


def _build_context() -> EngineContext:
    config = get_config()
    clients = ClientRegistry(dict(config.providers.providers))
    embeddings = EmbeddingClient()
    rerank = RerankClient() if os.environ.get("RERANK_SIDECAR_URL") else None
    web = WebSearchTool() if os.environ.get("LANGSEARCH_API_KEY") else None

    circuits = sorted({a.circuit for a in config.agents.values() if a.circuit})
    memories = {
        circuit: VectorMemory(
            collection=circuit,
            embedding_client=embeddings,
            rerank_client=rerank,
        )
        for circuit in circuits
    }
    return EngineContext(
        config=config,
        clients=clients,
        embeddings=embeddings,
        rerank=rerank,
        web=web,
        memories=memories,
        synthesis_memory=CompositeMemory(list(memories.values())),
    )


CTX = _build_context()
GRAPH = build_graph(CTX)


async def _ensure_memories() -> None:
    for memory in CTX.memories.values():
        try:
            await memory.ensure_collection()
        except Exception:
            # RAG is optional for availability: provider calls may still work.
            pass


@app.on_event("startup")
async def _startup() -> None:
    await _ensure_memories()


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _last_user(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_to_text(message.get("content"))
    return ""


async def _run(user_input: str) -> str:
    result = await GRAPH.ainvoke({"user_input": user_input})
    answer = result.get("synthesis_output", "")
    if not answer:
        errors = result.get("errors") or ["synthesis returned an empty response"]
        raise RuntimeError("; ".join(str(error) for error in errors))
    return answer


def _chunk_payload(
    *,
    request_id: str,
    created: int,
    model: str,
    text: str = "",
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": text} if text else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/v1/models")
async def models() -> dict:
    """Expose the logical model expected by OpenAI-compatible UIs."""
    return {
        "object": "list",
        "data": [
            {
                "id": LOGICAL_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be an array")

    user_input = _last_user(messages).strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="a non-empty user message is required")

    try:
        answer = await _run(user_input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"engine failed: {exc}") from exc

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = str(body.get("model") or LOGICAL_MODEL_ID)

    if bool(body.get("stream", False)):
        async def event_gen() -> AsyncIterator[str]:
            for offset in range(0, len(answer), 48):
                yield _chunk_payload(
                    request_id=request_id,
                    created=created,
                    model=model,
                    text=answer[offset : offset + 48],
                )
            yield _chunk_payload(
                request_id=request_id,
                created=created,
                model=model,
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "circuits": sorted(CTX.memories.keys())}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PY_ENGINE_PORT", "8823"))
    uvicorn.run(app, host="0.0.0.0", port=port)
