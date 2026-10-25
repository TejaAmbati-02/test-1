from __future__ import annotations

import base64
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

# All embeddings live in one Redis hash — the "collection" (REDIS_COLLECTION).
# HSET creates the collection on first write and adds to it afterwards. Each
# entry's field name is its embedding id, content-addressed: the same query text
# (normalised) or the same image bytes always hash to the same id, so a repeat
# request reuses the stored embedding instead of calling OCI again. The model
# id, dimension and normalisation are part of the hash, so changing any of them
# never serves a stale vector. Each entry expires REDIS_TTL_SECONDS after it is
# written (per-field HEXPIRE, Redis >= 7.4).
KEY_PREFIX = "asmo:emb"
KEY_PATTERN = re.compile(rf"^{KEY_PREFIX}:(query|image):[0-9a-f]{{64}}$")


class EmbeddingIdNotFound(Exception):
    """The embedding id is malformed, of the wrong kind, or expired from Redis."""


class EmbeddingStoreUnavailable(Exception):
    """Redis could not be reached while resolving an embedding id."""


@dataclass
class QueryEmbedding:
    key: str | None          # Redis key (embedding id); None when Redis was unavailable
    query: str
    vector: list[float]
    cached: bool


@dataclass
class ImageEmbedding:
    key: str | None
    data: bytes
    mime: str
    vector: list[float]
    cached: bool


def _normalise_query(query: str) -> str:
    return " ".join(query.split()).lower()


class EmbeddingStore:
    """Redis-backed cache of query / image embeddings, addressable by id.

    Collection entries (field = embedding id, value = JSON):
      query → {"type": "query", "query": ..., "embedding": [...], "model": ..., "dimension": ..., "created_at": ...}
      image → {"type": "image", "image_base64": ..., "mime": ..., "embedding": [...], ...}

    Redis failures never fail a search that sent a query or image: the embedding
    is generated as usual and returned without an id.
    """

    def __init__(self, settings: Settings, embedder: EmbeddingService) -> None:
        self._settings = settings
        self._emb = embedder
        self._client: redis.Redis | None = None
        # Flips to False if the server lacks per-field expiry (HEXPIRE, Redis < 7.4).
        self._field_ttl = True

    @property
    def enabled(self) -> bool:
        return bool(self._settings.redis_enabled and self._settings.redis_host)

    def _redis(self) -> redis.Redis:
        if self._client is None:
            cfg = self._settings
            tls = {"ssl": True, "ssl_cert_reqs": cfg.redis_ssl_cert_reqs} if cfg.redis_ssl else {}
            self._client = redis.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                username=cfg.redis_username or None,
                password=cfg.redis_password or None,
                db=cfg.redis_db,
                socket_connect_timeout=cfg.redis_timeout,
                socket_timeout=cfg.redis_timeout,
                **tls,
            )
        return self._client

    async def ping(self) -> bool:
        if not self.enabled:
            return False
        try:
            return bool(await self._redis().ping())
        except Exception as exc:
            logger.warning(f"EmbeddingStore.ping — Redis unreachable: {exc}")
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _key(self, kind: str, content: bytes) -> str:
        cfg = self._settings
        h = hashlib.sha256()
        h.update(f"{cfg.embedding_model_name}|{cfg.embedding_dimension}|{cfg.embedding_normalize}|".encode())
        h.update(content)
        return f"{KEY_PREFIX}:{kind}:{h.hexdigest()}"

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
        ttl = self._settings.redis_ttl_seconds
        client = self._redis()
        existed, _ = await client.pipeline(transaction=False).exists(self.collection).hset(self.collection, key, json.dumps(record)).execute()
        if not existed:
            logger.info(f"EmbeddingStore — created collection '{self.collection}'")
        await self._expire_entry(client, key, ttl)

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

    # --- Embedding by content -------------------------------------------------

    async def query_embedding(self, query: str) -> QueryEmbedding:
        """Return the query's embedding, from Redis when this query was seen before."""
        if not self.enabled:
            return QueryEmbedding(None, query, await self._emb.encode_query(query), False)
        key = self._key("query", _normalise_query(query).encode("utf-8"))
        try:
            record = await self._get(key)
            if record is not None:
                logger.info("EmbeddingStore.query_embedding — Redis hit", extra={"input_data": {"key": key}})
                return QueryEmbedding(key, query, record["embedding"], True)
        except Exception as exc:
            logger.warning(f"EmbeddingStore.query_embedding — Redis read failed, embedding without cache: {exc}")
            return QueryEmbedding(None, query, await self._emb.encode_query(query), False)

        vector = await self._emb.encode_query(query)
        try:
            await self._set(key, {"type": "query", "query": query, "embedding": vector})
            logger.info("EmbeddingStore.query_embedding — stored new embedding", extra={"input_data": {"key": key}})
        except Exception as exc:
            logger.warning(f"EmbeddingStore.query_embedding — Redis write failed: {exc}")
            key = None
        return QueryEmbedding(key, query, vector, False)

    async def image_embedding(self, data: bytes, mime: str) -> ImageEmbedding:
        """Return the image's embedding, from Redis when these exact bytes were seen before."""
        if not self.enabled:
            return ImageEmbedding(None, data, mime, await self._emb.encode_image(data, mime), False)
        key = self._key("image", data)
        try:
            record = await self._get(key)
            if record is not None:
                logger.info("EmbeddingStore.image_embedding — Redis hit", extra={"input_data": {"key": key}})
                return ImageEmbedding(key, data, mime, record["embedding"], True)
        except Exception as exc:
            logger.warning(f"EmbeddingStore.image_embedding — Redis read failed, embedding without cache: {exc}")
            return ImageEmbedding(None, data, mime, await self._emb.encode_image(data, mime), False)

        vector = await self._emb.encode_image(data, mime)
        try:
            await self._set(key, {
                "type": "image",
                "image_base64": base64.b64encode(data).decode("ascii"),
                "mime": mime,
                "embedding": vector,
            })
            logger.info("EmbeddingStore.image_embedding — stored new embedding", extra={"input_data": {"key": key}})
        except Exception as exc:
            logger.warning(f"EmbeddingStore.image_embedding — Redis write failed: {exc}")
            key = None
        return ImageEmbedding(key, data, mime, vector, False)

    # --- Embedding by id --------------------------------------------------------

    async def _load(self, key: str, kind: str) -> dict[str, Any]:
        match = KEY_PATTERN.match(key or "")
        if not match or match.group(1) != kind:
            raise EmbeddingIdNotFound(f"'{key}' is not a valid {kind} embedding id")
        if not self.enabled:
            raise EmbeddingStoreUnavailable("Redis is disabled (REDIS_ENABLED=false or REDIS_HOST unset)")
        try:
            record = await self._get(key)
        except Exception as exc:
            raise EmbeddingStoreUnavailable(str(exc)) from exc
        if record is None:
            raise EmbeddingIdNotFound(f"{kind} embedding id '{key}' was not found or has expired")
        return record

    async def load_query(self, key: str) -> QueryEmbedding:
        record = await self._load(key, "query")
        logger.info("EmbeddingStore.load_query — resolved id", extra={"input_data": {"key": key}})
        return QueryEmbedding(key, record["query"], record["embedding"], True)

    async def load_image(self, key: str) -> ImageEmbedding:
        record = await self._load(key, "image")
        logger.info("EmbeddingStore.load_image — resolved id", extra={"input_data": {"key": key}})
        return ImageEmbedding(key, base64.b64decode(record["image_base64"]), record["mime"], record["embedding"], True)


embedding_store = EmbeddingStore(_settings, embedding_service)
