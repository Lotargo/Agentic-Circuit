"""Minimal OpenAI-compatible provider used by GitHub Actions smoke tests."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def response_content(messages: list[dict]) -> str:
    system = messages[0].get("content", "") if messages else ""
    user = messages[-1].get("content", "") if messages else ""
    if "MEMORY_SELECT" in system:
        match = re.search(r'"id"\s*:\s*"([^"]+)"', user)
        selected = [match.group(1)] if match else []
        return json.dumps(
            {"selected_ids": selected, "outdated_ids": []},
            ensure_ascii=False,
        )
    if "MEMORY_EXTRACT" in system:
        return json.dumps(
            {
                "memories": [
                    {
                        "should_store": True,
                        "sensitive": False,
                        "memory_type": "user_fact",
                        "canonical_key": "user.identity.name",
                        "content": "Пользователя зовут Олег",
                        "source": "user_explicit",
                        "confidence": 0.99,
                        "importance": 0.8,
                        "ttl_days": None,
                    }
                ]
            },
            ensure_ascii=False,
        )
    if "маршрутизатор" in system or "функцию выбора глубины" in system:
        return "fast"
    return "SMOKE_OK"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        payload = {
            "id": "chatcmpl-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content(messages),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
