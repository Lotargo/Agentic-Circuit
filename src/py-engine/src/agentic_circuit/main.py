"""FastAPI entrypoint for the Python LangGraph engine."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import CircuitConfig, config_fingerprint
from .graph import CompositeMemory, EngineContext, build_graph
from .providers import ClientRegistry
from .rag import EmbeddingClient, RerankClient, VectorMemory
from .tools import WebSearchTool

app = FastAPI(title="agentic-circuit-engine", version="0.2.0")
LOGICAL_MODEL_ID = "agentic-circuit"
ALLOWED_ROLES = {"user", "assistant"}
ALLOWED_PRISMS = {"joy", "flirt", "resentment", "arousal", "anger", "apathy", "neutral", "sadness"}


def _build_context() -> EngineContext:
    config = CircuitConfig.from_disk()
    clients = ClientRegistry(dict(config.providers.providers))
    embeddings = EmbeddingClient()
    rerank = RerankClient() if os.environ.get("RERANK_SIDECAR_URL") else None
    web = WebSearchTool() if os.environ.get("LANGSEARCH_API_KEY") else None
    circuits = sorted({agent.circuit for agent in config.agents.values() if agent.circuit})
    memories = {circuit: VectorMemory(circuit, embeddings, rerank) for circuit in circuits}
    return EngineContext(
        config=config,
        clients=clients,
        embeddings=embeddings,
        rerank=rerank,
        web=web,
        memories=memories,
        synthesis_memory=CompositeMemory(list(memories.values())),
    )


async def _initialize_memories(context: EngineContext) -> None:
    for memory in context.memories.values():
        try:
            await memory.ensure_collection()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


class Runtime:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._init_task: asyncio.Task | None = None
        self.fingerprint = ""
        self.context: EngineContext | None = None
        self.graph = None

    async def ensure_current(self) -> None:
        current = config_fingerprint()
        if self.graph is not None and current == self.fingerprint:
            return
        async with self._lock:
            current = config_fingerprint()
            if self.graph is not None and current == self.fingerprint:
                return
            old_context = self.context
            old_task = self._init_task
            context = _build_context()
            self.context = context
            self.graph = build_graph(context)
            self.fingerprint = current
            self._init_task = asyncio.create_task(_initialize_memories(context))
            if old_task is not None:
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass
            if old_context is not None:
                await old_context.aclose()

    async def close(self) -> None:
        if self._init_task is not None:
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                pass
            self._init_task = None
        if self.context:
            await self.context.aclose()
            self.context = None
            self.graph = None


RUNTIME = Runtime()


@app.on_event("startup")
async def _startup() -> None:
    await RUNTIME.ensure_current()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await RUNTIME.close()


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _normalize_conversation(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="'messages' must be an array")
    conversation: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _content_to_text(message.get("content")).strip()
        if role in ALLOWED_ROLES and content:
            conversation.append({"role": role, "content": content})
    if not conversation or conversation[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the final non-empty message must be from the user")
    return conversation


def _graph_input(conversation: list[dict], prism: str) -> dict:
    return {
        "user_input": conversation[-1]["content"],
        "conversation": conversation,
        "prism": prism,
    }


async def _run(conversation: list[dict], prism: str) -> str:
    await RUNTIME.ensure_current()
    result = await RUNTIME.graph.ainvoke(_graph_input(conversation, prism))
    answer = result.get("synthesis_output", "")
    if not answer:
        errors = result.get("errors") or ["synthesis returned an empty response"]
        raise RuntimeError("; ".join(str(error) for error in errors))
    return answer


async def _stream(conversation: list[dict], prism: str) -> AsyncIterator[str]:
    """Yield synthesis deltas as soon as the upstream model emits them."""
    await RUNTIME.ensure_current()
    async for event in RUNTIME.graph.astream(
        _graph_input(conversation, prism),
        stream_mode="custom",
    ):
        if isinstance(event, dict) and event.get("type") == "token":
            content = event.get("content")
            if isinstance(content, str) and content:
                yield content


def _chunk_payload(*, request_id: str, created: int, model: str, text: str = "", finish_reason: str | None = None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": LOGICAL_MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}],
    }


@app.post("/v1/reload")
async def reload_config() -> dict:
    RUNTIME.fingerprint = ""
    await RUNTIME.ensure_current()
    return {"ok": True, "fingerprint": RUNTIME.fingerprint}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc

    conversation = _normalize_conversation(body.get("messages"))
    prism = str(body.get("prism") or "neutral")
    if prism not in ALLOWED_PRISMS:
        raise HTTPException(status_code=400, detail=f"unsupported prism: {prism}")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = str(body.get("model") or LOGICAL_MODEL_ID)

    if bool(body.get("stream", False)):
        async def event_gen() -> AsyncIterator[str]:
            try:
                async for token in _stream(conversation, prism):
                    yield _chunk_payload(
                        request_id=request_id,
                        created=created,
                        model=model,
                        text=token,
                    )
                yield _chunk_payload(
                    request_id=request_id,
                    created=created,
                    model=model,
                    finish_reason="stop",
                )
            except Exception as exc:
                error = {
                    "error": {
                        "message": f"engine stream failed: {exc}",
                        "type": "engine_error",
                    }
                }
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    try:
        answer = await _run(conversation, prism)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"engine failed: {exc}") from exc

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/healthz")
async def healthz() -> dict:
    await RUNTIME.ensure_current()
    return {
        "status": "ok",
        "config_fingerprint": RUNTIME.fingerprint,
        "rag_initializing": bool(RUNTIME._init_task and not RUNTIME._init_task.done()),
        "circuits": sorted(RUNTIME.context.memories.keys()) if RUNTIME.context else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PY_ENGINE_PORT", "8823")))
