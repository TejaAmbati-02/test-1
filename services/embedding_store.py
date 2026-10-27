from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis

from config import Settings, settings as _settings
from utils.logger import logger
from services.embedder import EmbeddingService, embedding_service
from services.redis_client import RedisClientService, redis_client

# Each search stores ONE Redis key — the search's request_id (a UUID) — whose
# value is a JSON object with its embeddings:
#   key   = "<request_id>"
#   value = {"text_emb": [...] | null, "image_emb": [...] | null, "query": "..." | null}
# The frontend sends that request_id back, with only filters / sort / page, to
# refine the same search without re-sending the query or re-uploading the image;
# the embeddings are read back from Redis by that key. The query text is kept
# alongside text_emb because the BM25 half of a text search needs it. Each key
# expires REDIS_TTL_SECONDS after it was last used (SET ... EX / EXPIRE).
REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class RequestIdNotFound(Exception):
    """The request_id is malformed or its search context expired from Redis."""


class EmbeddingStoreUnavailable(Exception):
    """Redis could not be reached while resolving a request_id."""


@dataclass
class SearchContext:
    """The query and/or image of one search, with their embeddings."""

    request_id: str                     # the search-context handle returned to the client
    query: str | None
    query_vector: list[float] | None    # stored as text_emb
    image_vector: list[float] | None    # stored as image_emb
    reused: bool                        # True when the context came from a previous request_id


def _normalise_query(query: str) -> str:
    return " ".join(query.split()).lower()


class EmbeddingStore:
    """Redis-backed store of search contexts, addressable by request_id.

    One key per search (key = request_id, value = JSON):
      {"text_emb": [...] | null, "image_emb": [...] | null, "query": ... | null}

    Only the image's embedding is kept, never the image itself.
    Redis failures never fail a search that sent a query or image: it runs as
    usual, its context just cannot be reused later.

    Uses the application's shared Redis client (services.redis_client), created
    once at startup and reused for every request.
    """

    def __init__(self, settings: Settings, embedder: EmbeddingService, redis_svc: RedisClientService) -> None:
        self._settings = settings
        self._emb = embedder
        self._redis_svc = redis_svc
        # Strong refs to in-flight background writes so they are not garbage-collected.
        self._pending: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self._redis_svc.enabled

    def _redis(self) -> redis.Redis:
        # The shared client; raises RedisNotConnected (handled like any Redis failure) before startup.
        return self._redis_svc.client

    async def _get(self, request_id: str) -> dict[str, Any] | None:
        """Read the value stored under ``request_id``; an expired key reads as missing."""
        raw = await self._redis().get(request_id)
        return json.loads(raw) if raw else None

    async def _set(self, request_id: str, value: dict[str, Any]) -> None:
        """Store ``value`` under the key ``request_id``, expiring after REDIS_TTL_SECONDS."""
        await self._redis().set(request_id, json.dumps(value), ex=self._settings.redis_ttl_seconds)

    def _in_background(self, what: str, key: str, coro_fn) -> None:
        """Run a Redis write without making the search wait on the round trips.

        If the write fails, a later lookup by that request_id is a plain miss.
        """
        async def run() -> None:
            try:
                await coro_fn()
                logger.info(f"EmbeddingStore — {what}", extra={"input_data": {"request_id": key}})
            except Exception as exc:
                logger.warning(f"EmbeddingStore — Redis write failed for {key} ({what}): {exc}")

        task = asyncio.create_task(run())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _load(self, request_id: str) -> SearchContext:
        if not REQUEST_ID_PATTERN.match(request_id or ""):
            raise RequestIdNotFound(f"'{request_id}' is not a valid request_id")
        if not self.enabled:
            raise EmbeddingStoreUnavailable("Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)")
        try:
            value = await self._get(request_id)
        except Exception as exc:
            raise EmbeddingStoreUnavailable(str(exc)) from exc
        if value is None:
            raise RequestIdNotFound(f"request_id '{request_id}' was not found or its search has expired")
        logger.info("EmbeddingStore — resolved request_id", extra={"input_data": {"request_id": request_id}})
        return SearchContext(
            request_id=request_id,
            query=value.get("query"),
            query_vector=value.get("text_emb"),
            image_vector=value.get("image_emb"),
            reused=True,
        )

    async def resolve(
        self,
        request_id: str,
        query: str | None,
        image: tuple[bytes, str] | None,
        previous_request_id: str | None,
    ) -> SearchContext:
        """Resolve the search context (query and/or image embeddings) of one search.

        * ``previous_request_id`` with no new query / image: the previous search
          is refined — its stored context is reused as-is and its request_id is
          returned again (the key's expiry is refreshed).
        * Otherwise this is a new search context under ``request_id``: a raw
          query / image replaces that part of the previous context (if any),
          only what is not already embedded is sent to OCI (concurrently), and
          the context is stored in the background.

        Raises:
            RequestIdNotFound: ``previous_request_id`` is malformed or expired.
            EmbeddingStoreUnavailable: Redis is down while ``previous_request_id`` must be read.
        """
        prev = await self._load(previous_request_id) if previous_request_id else None
        if prev and not query and image is None:
            if self.enabled:
                self._in_background("refreshed search context", prev.request_id,
                                    lambda: self._redis().expire(prev.request_id, self._settings.redis_ttl_seconds))
            return prev

        q = query or (prev.query if prev else None)

        async def query_vector() -> list[float] | None:
            if not q:
                return None
            if prev and prev.query and _normalise_query(prev.query) == _normalise_query(q):
                return prev.query_vector
            return await self._emb.encode_query(q)

        async def image_vector() -> list[float] | None:
            if image is None:
                return prev.image_vector if prev else None
            return await self._emb.encode_image(*image)

        q_vec, i_vec = await asyncio.gather(query_vector(), image_vector())
        if self.enabled:
            value = {"text_emb": q_vec, "image_emb": i_vec, "query": q}
            self._in_background("stored search context", request_id, lambda: self._set(request_id, value))
        return SearchContext(request_id, q, q_vec, i_vec, reused=False)


embedding_store = EmbeddingStore(_settings, embedding_service, redis_client)
