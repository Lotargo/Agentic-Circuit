"""MongoDB storage for raw source texts and OpenWebUI settings."""

from __future__ import annotations

import os

from . import rag  # noqa: F401  (keep package importable)

MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27617")
MONGODB_DB = os.environ.get("MONGODB_DB", "chat_openwebui")


class MongoStore:
    def __init__(self, url: str | None = None, db: str | None = None):
        self.url = url or MONGODB_URL
        self.db = db or MONGODB_DB
        self._client = None

    def _ensure(self):
        if self._client is None:
            from pymongo import AsyncMongoClient

            self._client = AsyncMongoClient(self.url)
        return self._client

    async def save_raw_text(self, collection: str, doc: dict) -> str:
        db = self._ensure()[self.db]
        res = await db[collection].insert_one(doc)
        return str(res.inserted_id)

    async def get_settings(self, key: str) -> dict | None:
        db = self._ensure()[self.db]
        return await db["settings"].find_one({"key": key})

    async def set_settings(self, key: str, value: dict) -> None:
        db = self._ensure()[self.db]
        await db["settings"].update_one({"key": key}, {"$set": {"key": key, **value}}, upsert=True)
