"""Search contexts in Redis, so a search can be refined by its request_id.

Each search stores ONE Redis key — its request_id (a UUID) — holding JSON:

    {"text_emb": [...] | null, "image_emb": [...] | null, "query": "..." | null}

The frontend sends that request_id back (with only filters / sort / page) to
refine the same search without re-sending the query or re-uploading the image.
The query text is kept next to text_emb because the BM25 half of a text search
needs it. Only the image's embedding is stored, never the image. Each key
expires REDIS_TTL_SECONDS after it was last used.

Redis failures never fail a search that sent a query or image: the search
runs as usual, its context just can't be reused later.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from config import Settings, settings as _settings
from services.embedder import EmbeddingService, embedding_service
from services.redis_client import RedisClientService, redis_client
from utils.logger import logger

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


class EmbeddingStore:
    def __init__(self, settings: Settings, embedder: EmbeddingService, redis_svc: RedisClientService) -> None:
        self._settings = settings
        self._emb = embedder
        self._redis_svc = redis_svc
        # Strong references to in-flight background writes so they aren't garbage-collected mid-write.
        self._pending: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return self._redis_svc.enabled

    async def resolve(
        self,
        request_id: str,
        query: str | None,
        image: tuple[bytes, str] | None,
        previous_request_id: str | None,
    ) -> SearchContext:
        """Work out the query / image embeddings for this search.

        * ``previous_request_id`` and no new query / image: refine that search —
          its stored context is reused as-is (same request_id, expiry refreshed).
        * Otherwise this is a new context stored under ``request_id``. A new query
          or image replaces that part of the previous context (if any); only what
          isn't already embedded is sent to OCI (query and image concurrently).

        Raises:
            RequestIdNotFound: ``previous_request_id`` is malformed or expired.
            EmbeddingStoreUnavailable: Redis is down/disabled while ``previous_request_id`` must be read.
        """
        start = time.perf_counter()
        prev = await self._load_previous(previous_request_id) if previous_request_id else None

        if prev and not query and image is None:
            if self.enabled:
                await self._save_in_background(prev.request_id, None)  # just refresh its expiry
            logger.info(f"EmbeddingStore — reusing stored context {prev.request_id} as-is")
            return prev

        q = query or (prev.query if prev else None)
        same_query_as_before = bool(
            q and prev and prev.query and " ".join(prev.query.split()).lower() == " ".join(q.split()).lower()
        )

        # Embed the query and the image concurrently (each reuses the previous vector when it can).
        q_vec, i_vec = await asyncio.gather(
            self._query_vector(q, prev, same_query_as_before),
            self._image_vector(image, prev),
        )
        if self.enabled:
            await self._save_in_background(request_id, {"text_emb": q_vec, "image_emb": i_vec, "query": q})
        logger.info(
            f"EmbeddingStore — new context {request_id} (text_emb_dim={len(q_vec or [])}, "
            f"image_emb_dim={len(i_vec or [])}) in {(time.perf_counter() - start) * 1000:.1f}ms",
            extra={"status": "success"},
        )
        return SearchContext(request_id, q, q_vec, i_vec, reused=False)

    async def _query_vector(self, q: str | None, prev: SearchContext | None, same_query_as_before: bool) -> list[float] | None:
        """The query embedding: none without a query, the stored one for an unchanged query, else a new one."""
        if not q:
            return None
        if same_query_as_before and prev is not None:
            return prev.query_vector
        return await self._emb.encode_query(q)

    async def _image_vector(self, image: tuple[bytes, str] | None, prev: SearchContext | None) -> list[float] | None:
        """The image embedding: the stored one when no new image was sent, else a new one."""
        if image is None:
            return prev.image_vector if prev else None
        return await self._emb.encode_image(*image)

    async def _load_previous(self, request_id: str) -> SearchContext:
        """Read a stored search context back from Redis."""
        if not REQUEST_ID_PATTERN.match(request_id or ""):
            raise RequestIdNotFound(f"'{request_id}' is not a valid request_id")
        if not self.enabled:
            raise EmbeddingStoreUnavailable("Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)")
        try:
            raw = await self._redis_svc.client.get(request_id)
            value: dict[str, Any] | None = json.loads(raw) if raw else None
        except Exception as exc:
            raise EmbeddingStoreUnavailable(str(exc)) from exc
        if value is None:
            raise RequestIdNotFound(f"request_id '{request_id}' was not found or its search has expired")
        logger.info(f"EmbeddingStore — loaded context {request_id}", extra={"status": "success"})
        return SearchContext(
            request_id=request_id,
            query=value.get("query"),
            query_vector=value.get("text_emb"),
            image_vector=value.get("image_emb"),
            reused=True,
        )

    async def _save_in_background(self, request_id: str, value: dict[str, Any] | None) -> None:
        """Store ``value`` under ``request_id`` (or, if None, just refresh its expiry) without making the search wait.

        A failed write is only logged: a later lookup by that request_id is then a plain miss.
        """
        task = asyncio.create_task(self._write(request_id, value))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _write(self, request_id: str, value: dict[str, Any] | None) -> None:
        """Do the Redis write for :meth:`_save_in_background`. A failure is logged, never raised."""
        ttl = self._settings.redis_ttl_seconds
        start = time.perf_counter()
        try:
            if value is None:
                await self._redis_svc.client.expire(request_id, ttl)
            else:
                await self._redis_svc.client.set(request_id, json.dumps(value), ex=ttl)
            logger.info(f"EmbeddingStore — saved {request_id} ({(time.perf_counter() - start) * 1000:.1f}ms)")
        except Exception as exc:
            logger.warning(
                f"EmbeddingStore — Redis write failed for {request_id}: {type(exc).__name__}: {exc}",
                extra={"status": "error", "error_details": str(exc)},
            )


embedding_store = EmbeddingStore(_settings, embedding_service, redis_client)
