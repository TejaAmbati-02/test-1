"""Search contexts in Redis, so a search can be refined by its request_id.

Each search stores ONE Redis key — its request_id (a UUID) — holding JSON:

    {"text_emb": [...] | null, "image_emb": [...] | null, "query": "..." | null}

The frontend sends that request_id back (with only filters / sort / page) to refine the same
search without re-sending the query or re-uploading the image. The query text is kept next to
text_emb because the BM25 half of a text search needs it. Only the image's embedding is stored,
never the image. Each key expires REDIS_TTL_SECONDS after it was last used.

Redis failures never fail a search that sent a query or image: the search runs as usual, its
context just can't be reused later.

Use the ready-made object:

    from services.embedding_store import embedding_store
    ctx = await embedding_store.resolve(request_id, query, image, previous_request_id)
"""
import asyncio
import json
import re
import time
from typing import Any

from config import Settings, settings
from models.response import ApiError
from services.embedder import EmbeddingService, embedding_service
from services.redis_client import RedisClientService, redis_client
from utils.logger import logger


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
    ) -> dict[str, Any]:
        """Work out the query / image embeddings for this search.

        Returns ``{"request_id", "query", "query_vector", "image_vector", "reused"}``:
        * ``previous_request_id`` and no new query / image: refine that search — its stored context
          is reused as-is (same request_id, expiry refreshed, reused=True).
        * Otherwise a new context stored under ``request_id``. A new query or image replaces that part
          of the previous context (if any); only what isn't already embedded is sent to OCI (query and
          image concurrently).

        Raises ApiError 400 REQUEST_ID_NOT_FOUND (malformed / expired previous_request_id) or
        503 EMBEDDING_STORE_UNAVAILABLE (Redis down / disabled while previous_request_id must be read).
        """
        start = time.perf_counter()
        ttl = self._settings.redis_ttl_seconds

        # 1. Load the previous search's context from Redis, if a previous_request_id was sent.
        prev: dict[str, Any] | None = None
        if previous_request_id:
            pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            if not re.match(pattern, previous_request_id):
                raise ApiError(400, "REQUEST_ID_NOT_FOUND", f"'{previous_request_id}' is not a valid request_id")
            if not self.enabled:
                problem = "Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)"
                logger.warning(f"EmbeddingStore — {problem}", extra={"status": "error"})
                raise ApiError(503, "EMBEDDING_STORE_UNAVAILABLE",
                               f"request_id cannot be resolved ({problem}); send search_query / image instead.")
            try:
                raw = await self._redis_svc.client.get(previous_request_id)
                stored = json.loads(raw) if raw else None
            except Exception as exc:
                logger.warning(f"EmbeddingStore — Redis unavailable while reading {previous_request_id}: {exc}",
                               extra={"status": "error", "error_details": str(exc)})
                raise ApiError(503, "EMBEDDING_STORE_UNAVAILABLE",
                               f"request_id cannot be resolved ({exc}); send search_query / image instead.")
            if stored is None:
                logger.warning(f"EmbeddingStore — request_id {previous_request_id} not found or expired",
                               extra={"status": "error"})
                raise ApiError(400, "REQUEST_ID_NOT_FOUND",
                               f"request_id '{previous_request_id}' was not found or its search has expired")
            prev = {"request_id": previous_request_id, "query": stored.get("query"),
                    "query_vector": stored.get("text_emb"), "image_vector": stored.get("image_emb"), "reused": True}
            logger.info(f"EmbeddingStore — loaded context {previous_request_id}", extra={"status": "success"})

        # 2. Refining: no new image, and no query or the SAME query as before (ignoring case and extra
        #    spaces) -> it is the same search: reuse the previous context as-is (same request_id) and
        #    refresh its expiry in the background. A failed refresh is only logged.
        same_query_as_before = bool(prev and query and prev["query"]
                                    and " ".join(prev["query"].split()).lower() == " ".join(query.split()).lower())
        if prev and image is None and (not query or same_query_as_before):
            try:
                task = asyncio.create_task(self._redis_svc.client.expire(prev["request_id"], ttl))
                self._pending.add(task)
                task.add_done_callback(lambda t: (
                    self._pending.discard(t),
                    t.cancelled() or t.exception() is None or logger.warning(
                        f"EmbeddingStore — Redis expiry refresh failed for {prev['request_id']}: {t.exception()!r}",
                        extra={"status": "error", "error_details": str(t.exception())}),
                ))
            except Exception as exc:
                logger.warning(f"EmbeddingStore — Redis expiry refresh failed for {prev['request_id']}: {exc!r}",
                               extra={"status": "error", "error_details": str(exc)})
            logger.info(f"EmbeddingStore — reusing stored context {prev['request_id']} as-is")
            return prev

        # 3. New context. The query: the new one, else the previous one. Re-embed it only when it changed
        #    (an unchanged query reuses the stored vector); a new image is always embedded, otherwise the
        #    previous image vector is kept. The query and the image are embedded concurrently.
        q = query or (prev["query"] if prev else None)
        same_query = bool(q and prev and prev["query"]
                          and " ".join(prev["query"].split()).lower() == " ".join(q.split()).lower())
        embed_query = bool(q) and not same_query
        new_q_vec, new_i_vec = await asyncio.gather(
            self._emb.encode_query(q) if embed_query else asyncio.sleep(0, result=None),
            self._emb.encode_image(*image) if image is not None else asyncio.sleep(0, result=None),
        )
        q_vec = new_q_vec if embed_query else (prev["query_vector"] if q and prev else None)
        i_vec = new_i_vec if image is not None else (prev["image_vector"] if prev else None)

        # 4. Store it under this request_id in the background, so the search doesn't wait. A failed
        #    write is only logged: a later lookup by that request_id is then a plain miss.
        if self.enabled:
            value = json.dumps({"text_emb": q_vec, "image_emb": i_vec, "query": q})
            try:
                task = asyncio.create_task(self._redis_svc.client.set(request_id, value, ex=ttl))
                self._pending.add(task)
                task.add_done_callback(lambda t: (
                    self._pending.discard(t),
                    t.cancelled() or (logger.info(f"EmbeddingStore — saved {request_id}") if t.exception() is None
                                      else logger.warning(f"EmbeddingStore — Redis write failed for {request_id}: "
                                                          f"{t.exception()!r}",
                                                          extra={"status": "error", "error_details": str(t.exception())})),
                ))
            except Exception as exc:
                logger.warning(f"EmbeddingStore — Redis write failed for {request_id}: {exc!r}",
                               extra={"status": "error", "error_details": str(exc)})

        logger.info(f"EmbeddingStore — new context {request_id} (text_emb_dim={len(q_vec or [])}, "
                    f"image_emb_dim={len(i_vec or [])}) in {(time.perf_counter() - start) * 1000:.1f}ms",
                    extra={"status": "success"})
        return {"request_id": request_id, "query": q, "query_vector": q_vec, "image_vector": i_vec, "reused": False}


# The one object the app uses — import it where needed:  from services.embedding_store import embedding_store
embedding_store = EmbeddingStore(settings, embedding_service, redis_client)
