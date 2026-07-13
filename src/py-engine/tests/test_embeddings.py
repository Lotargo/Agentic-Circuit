import json

import httpx

from agentic_circuit.rag.embeddings import EmbeddingClient


async def test_e5_query_prefix_and_tei_payload():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        assert json.loads(request.content) == {"inputs": ["query: похожий текст"]}
        return httpx.Response(200, json=[[1.0, 0.0]])

    client = EmbeddingClient(
        url="http://embedding.test",
        model="intfloat/multilingual-e5-small",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.embed(["похожий текст"], input_type="query")
    finally:
        await client.aclose()

    assert result == [[1.0, 0.0]]


async def test_e5_passage_prefix():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"inputs": ["passage: сохранённый текст"]}
        return httpx.Response(200, json={"embeddings": [[0.0, 1.0]]})

    client = EmbeddingClient(
        url="http://embedding.test",
        model="intfloat/multilingual-e5-small",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.embed(["сохранённый текст"], input_type="passage")
    finally:
        await client.aclose()

    assert result == [[0.0, 1.0]]
