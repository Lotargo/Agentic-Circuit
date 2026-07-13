"""FastAPI entrypoint for the Python LangGraph engine (langserve-style).

Exposes an OpenAI-compatible POST /v1/chat/completions endpoint that the
TS gateway (and OpenWebUI) call. Supports streaming via Server-Sent Events.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .config import get_config
from .graph import CompositeMemory, EngineContext, build_graph
from .providers import ClientRegistry
from .rag import EmbeddingClient, RerankClient, VectorMemory
from .tools import WebSearchTool

app = FastAPI(title="agentic-circuit-engine", version="0.1.0")


def _build_context() -> EngineContext:
    config = get_config()
    clients = ClientRegistry({k: v for k, v in config.providers.providers.items()})
    embeddings = EmbeddingClient()
    rerank = RerankClient() if os.environ.get("RERANK_SIDECAR_URL") else None
    web = WebSearchTool() if os.environ.get("LANGSEARCH_API_KEY") else None

    circuits = sorted({a.circuit for a in config.agents.values() if a.circuit})
    memories = {c: VectorMemory(collection=c, embedding_client=embeddings, rerank_client=rerank) for c in circuits}
    synth_mem = CompositeMemory(list(memories.values()))
    return EngineContext(
        config=config,
        clients=clients,
        embeddings=embeddings,
        rerank=rerank,
        web=web,
        memories=memories,
        synthesis_memory=synth_mem,
    )


CTX = _build_context()
GRAPH = build_graph(CTX)


async def _ensure_memories() -> None:
    for m in CTX.memories.values():
        try:
            await m.ensure_collection()
        except Exception:
            pass
    try:
        if CTX.synthesis_memory:
            await CTX.synthesis_memory.ensure_collection()
    except Exception:
        pass


@app.on_event("startup")
async def _startup() -> None:
    await _ensure_memories()


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


async def _run(user_input: str) -> str:
    result = await GRAPH.ainvoke({"user_input": user_input})
    return result.get("synthesis_output", "")


def _sse_chunk(text: str, finish: bool = False) -> str:
    payload = {
        "choices": [
            {
                "delta": {"content": text} if not finish else {},
                "finish_reason": "stop" if finish else None,
            }
        ]
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    user_input = _last_user(body.get("messages", []))
    stream = bool(body.get("stream", False))

    async def event_gen() -> AsyncIterator[dict]:
        answer = await _run(user_input)
        # chunk the final answer for an SSE-style stream
        for chunk in [answer[i : i + 24] for i in range(0, len(answer), 24)]:
            yield {"data": _sse_chunk(chunk)}
        yield {"data": _sse_chunk("", finish=True)}
        yield {"data": "data: [DONE]\n\n"}

    if stream:
        return EventSourceResponse(event_gen())

    answer = await _run(user_input)
    return {
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "circuits": sorted(CTX.memories.keys())}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PY_ENGINE_PORT", "8823"))
    uvicorn.run(app, host="0.0.0.0", port=port)
