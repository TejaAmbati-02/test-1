from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.embedding_store import (
    EmbeddingStore,
    EmbeddingStoreUnavailable,
    RequestIdNotFound,
    SearchContext,
)
from services.redis_client import RedisClientService

NEW_ID = "11111111-1111-4111-8111-111111111111"
OLD_ID = "22222222-2222-4222-8222-222222222222"


class FakeRedis:
    """In-memory stand-in for the parts of redis.asyncio.Redis the store uses."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key], self.ttls[key] = value, ex

    async def expire(self, key, ttl):
        self.ttls[key] = ttl


@pytest.fixture
def embedder():
    emb = MagicMock()
    emb.encode_query = AsyncMock(return_value=[0.1, 0.2])
    emb.encode_image = AsyncMock(return_value=[0.3, 0.4])
    return emb


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def store(test_settings, embedder, fake_redis):
    settings = test_settings.model_copy(update={"redis_host": "redis.test", "redis_ttl_seconds": 60})
    redis_svc = RedisClientService(settings)
    redis_svc._client = fake_redis
    return EmbeddingStore(settings, embedder, redis_svc)


@pytest.fixture
def disabled_store(test_settings, embedder):
    return EmbeddingStore(test_settings, embedder, RedisClientService(test_settings))  # REDIS_HOST is empty


async def _drain(store: EmbeddingStore) -> None:
    """Wait for the background Redis writes to finish."""
    if store._pending:
        await asyncio.gather(*list(store._pending))
    await asyncio.sleep(0)  # let the done-callbacks clear _pending


def _stored(fake_redis: FakeRedis, key: str) -> dict:
    return json.loads(fake_redis.data[key])


class TestNewContext:
    async def test_query_is_embedded_and_stored(self, store, embedder, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", None, None)
        assert ctx == SearchContext(NEW_ID, "valve", [0.1, 0.2], None, reused=False)
        embedder.encode_query.assert_awaited_once_with("valve")
        await _drain(store)
        assert _stored(fake_redis, NEW_ID) == {"text_emb": [0.1, 0.2], "image_emb": None, "query": "valve"}
        assert fake_redis.ttls[NEW_ID] == 60
        assert not store._pending

    async def test_query_and_image_are_both_embedded(self, store, embedder, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", (b"img", "image/png"), None)
        assert (ctx.query_vector, ctx.image_vector) == ([0.1, 0.2], [0.3, 0.4])
        embedder.encode_image.assert_awaited_once_with(b"img", "image/png")
        await _drain(store)
        assert _stored(fake_redis, NEW_ID)["image_emb"] == [0.3, 0.4]

    async def test_disabled_store_embeds_without_saving(self, disabled_store, embedder):
        ctx = await disabled_store.resolve(NEW_ID, "valve", None, None)
        assert ctx.query_vector == [0.1, 0.2]
        assert not disabled_store._pending

    async def test_background_write_failure_is_swallowed(self, store, fake_redis):
        fake_redis.set = AsyncMock(side_effect=ConnectionError("write failed"))
        ctx = await store.resolve(NEW_ID, "valve", None, None)
        await _drain(store)  # must not raise
        assert ctx.query_vector == [0.1, 0.2]


class TestRefine:
    @pytest.fixture
    def previous(self, fake_redis):
        fake_redis.data[OLD_ID] = json.dumps({"text_emb": [0.5], "image_emb": [0.6], "query": "Ball Valve"})

    async def test_previous_context_is_reused_as_is(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID)
        assert ctx == SearchContext(OLD_ID, "Ball Valve", [0.5], [0.6], reused=True)
        embedder.encode_query.assert_not_awaited()
        await _drain(store)
        assert fake_redis.ttls[OLD_ID] == 60  # expiry refreshed
        assert NEW_ID not in fake_redis.data

    async def test_new_query_replaces_text_and_keeps_image(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, "pump", None, OLD_ID)
        assert ctx == SearchContext(NEW_ID, "pump", [0.1, 0.2], [0.6], reused=False)
        await _drain(store)
        assert _stored(fake_redis, NEW_ID) == {"text_emb": [0.1, 0.2], "image_emb": [0.6], "query": "pump"}

    async def test_same_query_reuses_its_vector(self, store, embedder, previous):
        ctx = await store.resolve(NEW_ID, "  ball   VALVE ", None, OLD_ID)
        assert ctx.query_vector == [0.5]
        embedder.encode_query.assert_not_awaited()

    async def test_new_image_replaces_image_and_keeps_query(self, store, embedder, previous):
        ctx = await store.resolve(NEW_ID, None, (b"img", "image/jpeg"), OLD_ID)
        assert (ctx.query, ctx.query_vector, ctx.image_vector) == ("Ball Valve", [0.5], [0.3, 0.4])
        embedder.encode_query.assert_not_awaited()

    @pytest.mark.parametrize("bad_id", ["nope", "AAAAAAAA-2222-4222-8222-222222222222"])  # ids are lowercase
    async def test_malformed_request_id(self, store, bad_id):
        with pytest.raises(RequestIdNotFound, match="not a valid request_id"):
            await store.resolve(NEW_ID, None, None, bad_id)

    async def test_expired_request_id(self, store):
        with pytest.raises(RequestIdNotFound, match="not found or its search has expired"):
            await store.resolve(NEW_ID, None, None, OLD_ID)

    async def test_disabled_store(self, disabled_store):
        with pytest.raises(EmbeddingStoreUnavailable, match="disabled"):
            await disabled_store.resolve(NEW_ID, None, None, OLD_ID)

    async def test_redis_error(self, store, fake_redis):
        fake_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(EmbeddingStoreUnavailable, match="down"):
            await store.resolve(NEW_ID, None, None, OLD_ID)
