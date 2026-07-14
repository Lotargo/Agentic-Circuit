"""LangGraph circuit with selective recall and structured memory persistence."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, StreamWriter

from ..config import CircuitConfig
from ..memory import MemoryManager
from ..providers import ClientRegistry
from ..rag import EmbeddingClient, MemoryHit, RerankClient, VectorMemory
from ..tools import WebSearchTool
from .prompts import phase1_messages, phase2_messages, router_messages, synthesis_messages


def _merge(left, right):
    if not left:
        return right or {}
    if not right:
        return left
    return {**left, **right}


def _merge_errors(left, right):
    return list(left or []) + list(right or [])


class CircuitState(TypedDict, total=False):
    user_input: str
    conversation: list[dict]
    prism: str
    memory_scope: str
    workspace_id: str
    project_id: str
    conversation_id: str
    memory_contexts: list[MemoryHit]
    route: str
    router_raw: str
    circuit: str
    circuit_phase1: Annotated[dict[str, str], _merge]
    circuit_phase2: Annotated[dict[str, str], _merge]
    synthesis_output: str
    errors: Annotated[list[str], _merge_errors]


@dataclass
class EngineContext:
    config: CircuitConfig
    clients: ClientRegistry
    embeddings: EmbeddingClient
    rerank: Optional[RerankClient] = None
    web: Optional[WebSearchTool] = None
    memories: dict[str, VectorMemory] = field(default_factory=dict)
    structured_memory: Optional[VectorMemory] = None
    synthesis_memory: Optional[object] = None
    memory_manager: Optional[MemoryManager] = None

    async def aclose(self) -> None:
        close = getattr(self.clients, "aclose", None)
        if close:
            await close()
        if self.embeddings:
            await self.embeddings.aclose()
        if self.rerank:
            await self.rerank.aclose()
        if self.web:
            await self.web.aclose()

        unique: dict[int, VectorMemory] = {
            id(memory): memory for memory in self.memories.values()
        }
        if self.structured_memory is not None:
            unique[id(self.structured_memory)] = self.structured_memory
        for memory in unique.values():
            await memory.aclose()


class CompositeMemory:
    """Retrieve across collections, deduplicate and rank globally."""

    def __init__(
        self,
        memories: list[VectorMemory],
        rerank_client: RerankClient | None = None,
    ):
        self.memories = memories
        self._rerank = rerank_client

    async def ensure_collection(self) -> None:
        await asyncio.gather(
            *(memory.ensure_collection() for memory in self.memories),
            return_exceptions=True,
        )

    async def retrieve(
        self,
        query: str,
        *,
        scope: str | None,
        project_id: str = "",
        conversation_id: str = "",
        top_k: int = 5,
        use_rerank: bool = True,
    ) -> list[MemoryHit]:
        if not self.memories or not scope or top_k <= 0:
            return []

        per_collection = max(top_k * 3, 8)
        batches = await asyncio.gather(
            *(
                memory.retrieve(
                    query,
                    scope=scope,
                    project_id=project_id,
                    conversation_id=conversation_id,
                    top_k=per_collection,
                    use_rerank=False,
                )
                for memory in self.memories
            ),
            return_exceptions=True,
        )

        candidates: list[MemoryHit] = []
        seen: set[tuple[str, str, str]] = set()
        for batch in batches:
            if not isinstance(batch, list):
                continue
            for hit in batch:
                key = (hit.canonical_key or hit.source, hit.memory_type, hit.text)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(hit)

        candidate_limit = max(top_k * 5, 20)
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[:candidate_limit]
        if use_rerank and self._rerank and candidates:
            try:
                ranked = await self._rerank.rerank_indices(
                    query,
                    [hit.rerank_text() for hit in candidates],
                    top_n=candidate_limit,
                )
                candidates = [
                    replace(
                        candidates[index],
                        score=max(0.0, float(score)) * max(candidates[index].score, 1e-6),
                    )
                    for index, score in ranked
                    if 0 <= index < len(candidates)
                ]
                candidates.sort(key=lambda item: item.score, reverse=True)
            except Exception:
                pass
        return candidates[:top_k]


def _parse_route(raw: str) -> str:
    return "slow" if re.findall(r"[a-z]+", (raw or "").lower()) == ["slow"] else "fast"


async def _safe_retrieve(
    memory,
    query: str,
    scope: str | None,
    project_id: str,
    conversation_id: str,
    label: str,
    errors: list[str],
    *,
    top_k: int = 12,
) -> list[MemoryHit]:
    try:
        return await memory.retrieve(
            query,
            scope=scope,
            project_id=project_id,
            conversation_id=conversation_id,
            top_k=top_k,
        )
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return []


async def _safe_upsert(
    memory,
    text: str,
    scope: str | None,
    label: str,
    errors: list[str],
    **metadata,
) -> None:
    try:
        await memory.upsert(text, scope=scope, **metadata)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")


def build_graph(ctx: EngineContext):
    circuits = sorted(ctx.config.circuit_collections)

    async def router_node(state: CircuitState) -> dict:
        agent = ctx.config.router
        result = await ctx.clients.get(agent.model.provider).acomplete(
            router_messages(agent, state["user_input"]),
            agent.model,
        )
        update: dict = {
            "router_raw": result.content,
            "route": _parse_route(result.content),
        }
        if result.error:
            update["errors"] = [f"router: {result.error}"]
        return update

    async def memory_recall_node(state: CircuitState) -> dict:
        errors: list[str] = []
        scope = state.get("memory_scope")
        project_id = state.get("project_id", "")
        conversation_id = state.get("conversation_id", "")
        candidates = (
            await _safe_retrieve(
                ctx.synthesis_memory,
                state["user_input"],
                scope,
                project_id,
                conversation_id,
                "memory:retrieve",
                errors,
                top_k=16,
            )
            if ctx.synthesis_memory
            else []
        )
        selected = candidates[:6]
        if ctx.memory_manager and candidates:
            try:
                selected = await ctx.memory_manager.select(
                    state["user_input"],
                    candidates,
                    project_id=project_id,
                    top_k=6,
                )
            except Exception as exc:
                errors.append(f"memory:select: {type(exc).__name__}: {exc}")
        update: dict = {"memory_contexts": selected}
        if errors:
            update["errors"] = errors
        return update

    async def circuit_node(state: CircuitState) -> dict:
        circuit = state["circuit"]
        phase1 = ctx.config.get(f"{circuit}-1")
        phase2 = ctx.config.get(f"{circuit}-2")
        errors: list[str] = []
        prism = state.get("prism") or "neutral"
        conversation = state["conversation"]
        historical = state.get("memory_contexts", [])

        result1 = await ctx.clients.get(phase1.model.provider).acomplete(
            phase1_messages(
                phase1,
                conversation,
                historical if phase1.tools.rag else [],
                prism,
            ),
            phase1.model,
        )
        if result1.error:
            errors.append(f"{circuit}-1: {result1.error}")

        result2 = await ctx.clients.get(phase2.model.provider).acomplete(
            phase2_messages(
                phase2,
                conversation,
                result1.content,
                historical if phase2.tools.rag else [],
                prism,
            ),
            phase2.model,
        )
        if result2.error:
            errors.append(f"{circuit}-2: {result2.error}")

        # Perspective outputs remain ephemeral. The memory manager later extracts
        # only durable user facts and explicit project decisions from the complete turn.
        update: dict = {
            "circuit_phase1": {circuit: result1.content},
            "circuit_phase2": {circuit: result2.content},
        }
        if errors:
            update["errors"] = errors
        return update

    async def synthesis_node(state: CircuitState, *, writer: StreamWriter) -> dict:
        agent = ctx.config.synthesis
        errors: list[str] = []
        scope = state.get("memory_scope")
        project_id = state.get("project_id", "")
        conversation_id = state.get("conversation_id", "")
        prism = state.get("prism") or "neutral"
        contexts = state.get("memory_contexts", [])

        web_results: list[str] = []
        if agent.tools.web_search and ctx.web:
            try:
                web_results = await ctx.web.search(state["user_input"])
                if ctx.rerank and web_results:
                    web_results = [
                        text
                        for text, _ in await ctx.rerank.rerank(
                            state["user_input"],
                            web_results,
                            top_n=5,
                        )
                    ]
            except Exception as exc:
                errors.append(f"web_search: {type(exc).__name__}: {exc}")

        messages = synthesis_messages(
            agent,
            state["conversation"],
            state.get("circuit_phase1", {}),
            state.get("circuit_phase2", {}),
            contexts,
            web_results,
            prism,
        )
        client = ctx.clients.get(agent.model.provider)
        chunks: list[str] = []
        stream_method = getattr(client, "astream", None)
        if stream_method is not None:
            try:
                async for token in stream_method(messages, agent.model):
                    chunks.append(token)
                    writer({"type": "token", "content": token})
            except Exception as exc:
                errors.append(f"synthesis_stream: {type(exc).__name__}: {exc}")
        if not chunks:
            result = await client.acomplete(messages, agent.model)
            if result.content:
                chunks.append(result.content)
                writer({"type": "token", "content": result.content})
            if result.error:
                errors.append(f"synthesis: {result.error}")

        output = "".join(chunks)
        if (
            output
            and scope
            and ctx.structured_memory is not None
            and ctx.memory_manager is not None
        ):
            try:
                extracted = await ctx.memory_manager.extract(
                    state["conversation"],
                    output,
                    project_id=project_id,
                    existing=contexts,
                )
                for candidate in extracted:
                    await _safe_upsert(
                        ctx.structured_memory,
                        candidate.content,
                        scope,
                        "memory:upsert",
                        errors,
                        kind=candidate.memory_type,
                        memory_type=candidate.memory_type,
                        canonical_key=candidate.canonical_key,
                        source=candidate.source,
                        query=state["user_input"],
                        prism=prism,
                        project_id=project_id,
                        conversation_id=conversation_id,
                        confidence=candidate.confidence,
                        importance=candidate.importance,
                        ttl_days=candidate.ttl_days,
                    )
            except Exception as exc:
                errors.append(f"memory:extract: {type(exc).__name__}: {exc}")

        update: dict = {"synthesis_output": output}
        if errors:
            update["errors"] = errors
        return update

    def route_decision(state: CircuitState):
        if state.get("route") == "slow":
            return [
                Send(
                    "circuit",
                    {
                        "circuit": circuit,
                        "user_input": state["user_input"],
                        "conversation": state["conversation"],
                        "prism": state.get("prism") or "neutral",
                        "memory_scope": state.get("memory_scope", ""),
                        "workspace_id": state.get("workspace_id", ""),
                        "project_id": state.get("project_id", ""),
                        "conversation_id": state.get("conversation_id", ""),
                        "memory_contexts": state.get("memory_contexts", []),
                    },
                )
                for circuit in circuits
            ]
        return "fast"

    graph = StateGraph(CircuitState)
    graph.add_node("router", router_node)
    graph.add_node("memory_recall", memory_recall_node)
    graph.add_node("circuit", circuit_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge(START, "router")
    graph.add_edge("router", "memory_recall")
    graph.add_conditional_edges("memory_recall", route_decision, {"fast": "synthesis"})
    graph.add_edge("circuit", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()
