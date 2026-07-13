import json

import httpx

from agentic_circuit.rag.rerank import RerankClient


async def test_tei_rerank_contract_and_top_n():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        payload = json.loads(request.content)
        assert payload == {
            "query": "запрос",
            "texts": ["первый", "второй"],
            "return_text": True,
            "raw_scores": False,
        }
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.9, "text": "второй"},
                {"index": 0, "score": 0.2, "text": "первый"},
            ],
        )

    client = RerankClient(
        url="http://rerank.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.rerank(
            "запрос",
            ["первый", "второй"],
            top_n=1,
        )
    finally:
        await client.aclose()

    assert result == [("второй", 0.9)]


async def test_tei_rerank_can_restore_text_from_index():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"index": 1, "score": 0.7}])

    client = RerankClient(
        url="http://rerank.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.rerank("q", ["a", "b"])
    finally:
        await client.aclose()

    assert result == [("b", 0.7)]
