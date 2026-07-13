"""LangGraph circuit: router -> parallel circuits -> streamed synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, StreamWriter

from ..config import CircuitConfig
from ..providers import ClientRegistry
from ..rag import EmbeddingClient, RerankClient, VectorMemory
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
    synthesis_memory: Optional[object] = None

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
        for memory in self.memories.values():
            await memory.aclose()


class CompositeMemory:
    def __init__(self, memories: list[VectorMemory]):
        self.memories = memories

    async def ensure_collection(self) -> None:
        for memory in self.memories:
            await memory.ensure_collection()

    async def retrieve(self, query: str, top_k: int = 5, use_rerank: bool = True) -> list[str]:
        if not self.memories:
            return []
        per_collection = max(1, top_k // len(self.memories))
        results: list[str] = []
        for memory in self.memories:
            results.extend(await memory.retrieve(query, per_collection, use_rerank))
        return results[:top_k]

    async def upsert(self, text: str, doc_id: Optional[str] = None) -> str:
        return ""


def _parse_route(raw: str) -> str:
    return "slow" if re.findall(r"[a-z]+", (raw or "").lower()) == ["slow"] else "fast"


async def _safe_retrieve(memory, query: str, label: str, errors: list[str]) -> list[str]:
    try:
        return await memory.retrieve(query)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return []


async def _safe_upsert(memory, text: str, label: str, errors: list[str]) -> None:
    try:
        await memory.upsert(text)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")


def build_graph(ctx: EngineContext):
    circuits = sorted({agent.circuit for agent in ctx.config.agents.values() if agent.circuit})

    async def router_node(state: CircuitState) -> dict:
        agent = ctx.config.router
        result = await ctx.clients.get(agent.model.provider).acomplete(
            router_messages(agent, state["user_input"]), agent.model
        )
        update: dict = {"router_raw": result.content, "route": _parse_route(result.content)}
        if result.error:
            update["errors"] = [f"router: {result.error}"]
        return update

    async def circuit_node(state: CircuitState) -> dict:
        circuit = state["circuit"]
        phase1 = ctx.config.get(f"{circuit}-1")
        phase2 = ctx.config.get(f"{circuit}-2")
        memory = ctx.memories[circuit]
        errors: list[str] = []
        prism = state.get("prism") or "neutral"
        conversation = state["conversation"]
        historical = (
            await _safe_retrieve(memory, state["user_input"], f"{circuit}:rag_retrieve", errors)
            if phase1.tools.rag or phase2.tools.rag
            else []
        )
        result1 = await ctx.clients.get(phase1.model.provider).acomplete(
            phase1_messages(phase1, conversation, historical if phase1.tools.rag else [], prism),
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
        if phase1.tools.rag and result1.content:
            await _safe_upsert(memory, result1.content, f"{circuit}-1:rag_upsert", errors)
        if phase2.tools.rag and result2.content:
            await _safe_upsert(memory, result2.content, f"{circuit}-2:rag_upsert", errors)
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
        contexts = (
            await _safe_retrieve(
                ctx.synthesis_memory,
                state["user_input"],
                "synthesis:rag_retrieve",
                errors,
            )
            if ctx.synthesis_memory and agent.tools.rag
            else []
        )
        web_results: list[str] = []
        if agent.tools.web_search and ctx.web:
            try:
                web_results = await ctx.web.search(state["user_input"])
                if ctx.rerank and web_results:
                    web_results = [
                        text
                        for text, _ in await ctx.rerank.rerank(
                            state["user_input"], web_results, top_n=5
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
            state.get("prism") or "neutral",
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

        update: dict = {"synthesis_output": "".join(chunks)}
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
                    },
                )
                for circuit in circuits
            ]
        return "fast"

    graph = StateGraph(CircuitState)
    graph.add_node("router", router_node)
    graph.add_node("circuit", circuit_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision, {"fast": "synthesis"})
    graph.add_edge("circuit", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()
