from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError

from config import Settings, settings as _settings
from utils.logger import logger
from services.embedder import EmbeddingService, embedding_service
from services.redis_client import RedisClientService, redis_client

# Search contexts live in one Redis hash — the "collection" (REDIS_COLLECTION).
# HSET creates the collection on first write and adds to it afterwards. Each
# search stores ONE entry — its query and/or image embeddings — under the
# search's request_id (a UUID). The frontend sends that request_id back, with
# only filters / sort / page, to refine the same search without re-sending the
# query or re-uploading the image. Each entry expires REDIS_TTL_SECONDS after it
# was last used (per-field HEXPIRE, Redis >= 7.4).
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
    query_vector: list[float] | None
    image_sha256: str | None
    image_vector: list[float] | None
    reused: bool                        # True when the context came from a previous request_id


def _normalise_query(query: str) -> str:
    return " ".join(query.split()).lower()


class EmbeddingStore:
    """Redis-backed store of search contexts, addressable by request_id.

    Collection entries (field = request_id, value = JSON), one per search:
      {"query": ... | null, "query_embedding": [...] | null,
       "image_sha256": ... | null, "mime": ... | null, "image_embedding": [...] | null,
       "model": ..., "dimension": ..., "created_at": ...}

    Only the image's embedding and hash are kept, never the image itself.
    Redis failures never fail a search that sent a query or image: it runs as
    usual, its context just cannot be reused later.

    Uses the application's shared Redis client (services.redis_client), created
    once at startup and reused for every request.
    """

    def __init__(self, settings: Settings, embedder: EmbeddingService, redis_svc: RedisClientService) -> None:
        self._settings = settings
        self._emb = embedder
        self._redis_svc = redis_svc
        # Flips to False if the server lacks per-field expiry (HEXPIRE, Redis < 7.4).
        self._field_ttl = True
        # Strong refs to in-flight background writes so they are not garbage-collected.
        self._pending: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self._redis_svc.enabled

    def _redis(self) -> redis.Redis:
        # The shared client; raises RedisNotConnected (handled like any Redis failure) before startup.
        return self._redis_svc.client

    @property
    def collection(self) -> str:
        return self._settings.redis_collection

    async def _get(self, key: str) -> dict[str, Any] | None:
        """Read one entry from the collection; expired entries read as missing."""
        raw = await self._redis().hget(self.collection, key)
        return json.loads(raw) if raw else None

    async def _set(self, key: str, record: dict[str, Any]) -> None:
        """Add an entry to the collection (creating the collection if needed) with a TTL."""
        record = {
            **record,
            "model": self._settings.embedding_model_name,
            "dimension": self._settings.embedding_dimension,
            "created_at": int(time.time()),
        }
        client = self._redis()
        existed, _ = await client.pipeline(transaction=False).exists(self.collection).hset(self.collection, key, json.dumps(record)).execute()
        if not existed:
            logger.info(f"EmbeddingStore — created collection '{self.collection}'")
        await self._expire_entry(client, key, self._settings.redis_ttl_seconds)

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

    async def _expire_entry(self, client: redis.Redis, key: str, ttl: int) -> None:
        """Expire just this entry after `ttl` seconds; fall back to the whole collection on Redis < 7.4."""
        if self._field_ttl:
            try:
                await client.hexpire(self.collection, ttl, key)
                return
            except (ResponseError, AttributeError) as exc:
                self._field_ttl = False
                logger.warning(
                    f"EmbeddingStore — per-entry expiry (HEXPIRE) unsupported ({exc}); "
                    f"expiring the whole collection {ttl}s after its latest write instead (needs Redis >= 7.4)"
                )
        await client.expire(self.collection, ttl)

    async def _load(self, request_id: str) -> SearchContext:
        if not REQUEST_ID_PATTERN.match(request_id or ""):
            raise RequestIdNotFound(f"'{request_id}' is not a valid request_id")
        if not self.enabled:
            raise EmbeddingStoreUnavailable("Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)")
        try:
            record = await self._get(request_id)
        except Exception as exc:
            raise EmbeddingStoreUnavailable(str(exc)) from exc
        if record is None:
            raise RequestIdNotFound(f"request_id '{request_id}' was not found or its search has expired")
        logger.info("EmbeddingStore — resolved request_id", extra={"input_data": {"request_id": request_id}})
        return SearchContext(
            request_id=request_id,
            query=record.get("query"),
            query_vector=record.get("query_embedding"),
            image_sha256=record.get("image_sha256"),
            image_vector=record.get("image_embedding"),
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
          returned again (the entry's expiry is refreshed).
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
                                    lambda: self._expire_entry(self._redis(), prev.request_id, self._settings.redis_ttl_seconds))
            return prev

        q = query or (prev.query if prev else None)
        image_sha = hashlib.sha256(image[0]).hexdigest() if image else (prev.image_sha256 if prev else None)

        async def query_vector() -> list[float] | None:
            if not q:
                return None
            if prev and prev.query and _normalise_query(prev.query) == _normalise_query(q):
                return prev.query_vector
            return await self._emb.encode_query(q)

        async def image_vector() -> list[float] | None:
            if image is None:
                return prev.image_vector if prev else None
            if prev and prev.image_sha256 == image_sha:
                return prev.image_vector
            return await self._emb.encode_image(*image)

        q_vec, i_vec = await asyncio.gather(query_vector(), image_vector())
        if self.enabled:
            record = {
                "query": q, "query_embedding": q_vec,
                "image_sha256": image_sha, "mime": image[1] if image else None, "image_embedding": i_vec,
            }
            self._in_background("stored search context", request_id, lambda: self._set(request_id, record))
        return SearchContext(request_id, q, q_vec, image_sha, i_vec, reused=False)


embedding_store = EmbeddingStore(_settings, embedding_service, redis_client)
