from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from services import embedding_store as store_module
from services.embedding_store import (
    REQUEST_ID_PATTERN,
    EmbeddingStore,
    EmbeddingStoreUnavailable,
    RequestIdNotFound,
    SearchContext,
    _normalise_query,
)
from services.redis_client import RedisClientService, RedisConfig


class FakeRedis:
    """In-memory stand-in for the parts of redis.asyncio.Redis the store uses."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.set_calls: list = []
        self.expire_calls: list = []

    async def get(self, name):
        value = self.data.get(name)
        return value.encode() if value is not None else None

    async def set(self, name, value, ex=None):
        self.set_calls.append((name, ex))
        self.data[name] = value
        return True

    async def expire(self, name, ttl):
        self.expire_calls.append((name, ttl))
        return name in self.data


def _rid() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def embedder():
    emb = MagicMock()
    emb.encode_query = AsyncMock(return_value=[0.1, 0.2])
    emb.encode_image = AsyncMock(return_value=[0.3, 0.4])
    return emb


@pytest.fixture
def redis_settings(test_settings):
    return test_settings.model_copy(update={"redis_host": "redis.test", "redis_ttl_seconds": 60})


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def redis_svc(fake_redis):
    svc = RedisClientService(RedisConfig(host="redis.test", enabled=True))
    svc._client = fake_redis  # "connected" to the in-memory fake
    return svc


@pytest.fixture
def store(redis_settings, embedder, redis_svc):
    return EmbeddingStore(redis_settings, embedder, redis_svc)


@pytest.fixture
def disabled_store(test_settings, embedder):
    return EmbeddingStore(test_settings, embedder, RedisClientService(RedisConfig(host="", enabled=True)))


async def _drain(store: EmbeddingStore) -> None:
    if store._pending:
        await asyncio.gather(*list(store._pending))
    await asyncio.sleep(0)  # let the done-callbacks clear _pending


def _put(fake_redis: FakeRedis, rid: str, text_emb=None, image_emb=None, query=None) -> None:
    fake_redis.data[rid] = json.dumps({"text_emb": text_emb, "image_emb": image_emb, "query": query})


def test_normalise_query_collapses_whitespace_and_case():
    assert _normalise_query("  Ball   VALVE\t2in ") == "ball valve 2in"


class TestRequestIdPattern:
    def test_accepts_uuid4(self):
        assert REQUEST_ID_PATTERN.match(_rid())

    @pytest.mark.parametrize("value", ["", "garbage", "1234", str(uuid.uuid4()).upper(), _rid() + "x"])
    def test_rejects_other_values(self, value):
        assert not REQUEST_ID_PATTERN.match(value)


class TestEnabled:
    @pytest.mark.parametrize("enabled, host, expected", [
        (True, "redis.test", True),
        (True, "", False),
        (False, "redis.test", False),
    ])
    def test_enabled_follows_redis_client(self, test_settings, embedder, enabled, host, expected):
        svc = RedisClientService(RedisConfig(host=host, enabled=enabled))
        assert EmbeddingStore(test_settings, embedder, svc).enabled is expected

    def test_redis_uses_shared_client(self, store, fake_redis):
        assert store._redis() is fake_redis

    def test_shared_module_instance_uses_shared_redis_client(self):
        from services.embedder import embedding_service
        from services.redis_client import redis_client

        assert store_module.embedding_store._redis_svc is redis_client
        assert store_module.embedding_store._emb is embedding_service


class TestGetSet:
    async def test_set_stores_json_with_ttl(self, store, fake_redis):
        rid = _rid()
        await store._set(rid, {"text_emb": [1.0], "image_emb": None, "query": "q"})
        assert fake_redis.set_calls == [(rid, 60)]
        assert json.loads(fake_redis.data[rid]) == {"text_emb": [1.0], "image_emb": None, "query": "q"}

    async def test_get_round_trip(self, store):
        rid = _rid()
        await store._set(rid, {"text_emb": [1.0], "image_emb": [2.0], "query": "q"})
        assert await store._get(rid) == {"text_emb": [1.0], "image_emb": [2.0], "query": "q"}

    async def test_get_missing_key(self, store):
        assert await store._get(_rid()) is None

    async def test_get_error_propagates(self, store, fake_redis):
        fake_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await store._get(_rid())

    async def test_set_error_propagates(self, store, fake_redis):
        fake_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ConnectionError):
            await store._set(_rid(), {"text_emb": None, "image_emb": None, "query": None})

    async def test_not_connected_raises(self, redis_settings, embedder):
        from services.redis_client import RedisNotConnected

        s = EmbeddingStore(redis_settings, embedder, RedisClientService(RedisConfig(host="redis.test")))
        with pytest.raises(RedisNotConnected):
            await s._get(_rid())


class TestInBackground:
    async def test_runs_and_clears_pending(self, store):
        ran = AsyncMock()
        store._in_background("op", "k", ran)
        assert len(store._pending) == 1
        await _drain(store)
        ran.assert_awaited_once()
        assert not store._pending

    async def test_failure_is_swallowed(self, store):
        store._in_background("op", "k", AsyncMock(side_effect=ConnectionError("write failed")))
        await _drain(store)  # must not raise
        assert not store._pending


class TestLoad:
    async def test_loads_stored_context(self, store, fake_redis):
        rid = _rid()
        _put(fake_redis, rid, text_emb=[1.0], image_emb=[2.0], query="valve")
        ctx = await store._load(rid)
        assert ctx == SearchContext(rid, "valve", [1.0], [2.0], reused=True)

    @pytest.mark.parametrize("rid", ["", None, "garbage", "asmo:emb:query:xyz"])
    async def test_malformed_id(self, store, rid):
        with pytest.raises(RequestIdNotFound, match="not a valid request_id"):
            await store._load(rid)

    async def test_malformed_id_checked_before_enabled(self, disabled_store):
        with pytest.raises(RequestIdNotFound):
            await disabled_store._load("garbage")

    async def test_expired_id(self, store):
        with pytest.raises(RequestIdNotFound, match="not found or its search has expired"):
            await store._load(_rid())

    async def test_disabled_store(self, disabled_store):
        with pytest.raises(EmbeddingStoreUnavailable, match="disabled"):
            await disabled_store._load(_rid())

    async def test_redis_error(self, store, fake_redis):
        fake_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(EmbeddingStoreUnavailable, match="down"):
            await store._load(_rid())

    async def test_not_connected_is_unavailable(self, redis_settings, embedder):
        s = EmbeddingStore(redis_settings, embedder, RedisClientService(RedisConfig(host="redis.test")))
        with pytest.raises(EmbeddingStoreUnavailable):
            await s._load(_rid())


class TestResolveNewContext:
    async def test_query_only(self, store, embedder, fake_redis):
        rid = _rid()
        ctx = await store.resolve(rid, "Ball Valve", None, None)
        assert ctx == SearchContext(rid, "Ball Valve", [0.1, 0.2], None, reused=False)
        embedder.encode_query.assert_awaited_once_with("Ball Valve")
        embedder.encode_image.assert_not_awaited()
        await _drain(store)
        assert json.loads(fake_redis.data[rid]) == {"text_emb": [0.1, 0.2], "image_emb": None, "query": "Ball Valve"}
        assert fake_redis.set_calls == [(rid, 60)]
        assert not store._pending

    async def test_image_only(self, store, embedder, fake_redis):
        rid = _rid()
        ctx = await store.resolve(rid, None, (b"img", "image/png"), None)
        assert ctx.query is None and ctx.query_vector is None
        assert ctx.image_vector == [0.3, 0.4]
        assert ctx.reused is False
        embedder.encode_image.assert_awaited_once_with(b"img", "image/png")
        embedder.encode_query.assert_not_awaited()
        await _drain(store)
        assert json.loads(fake_redis.data[rid]) == {"text_emb": None, "image_emb": [0.3, 0.4], "query": None}

    async def test_query_and_image(self, store, embedder):
        ctx = await store.resolve(_rid(), "valve", (b"img", "image/jpeg"), None)
        assert ctx.query_vector == [0.1, 0.2]
        assert ctx.image_vector == [0.3, 0.4]
        embedder.encode_query.assert_awaited_once()
        embedder.encode_image.assert_awaited_once()

    async def test_neither_query_nor_image(self, store, embedder):
        ctx = await store.resolve(_rid(), None, None, None)
        assert ctx.query is None and ctx.query_vector is None and ctx.image_vector is None
        embedder.encode_query.assert_not_awaited()
        embedder.encode_image.assert_not_awaited()

    async def test_disabled_embeds_without_storing(self, disabled_store, embedder):
        ctx = await disabled_store.resolve(_rid(), "valve", None, None)
        assert ctx.query_vector == [0.1, 0.2]
        assert not disabled_store._pending

    async def test_background_write_failure_is_swallowed(self, store, fake_redis):
        fake_redis.set = AsyncMock(side_effect=ConnectionError("write failed"))
        ctx = await store.resolve(_rid(), "valve", None, None)
        await _drain(store)  # must not raise
        assert ctx.query_vector == [0.1, 0.2]

    async def test_not_connected_write_failure_is_swallowed(self, redis_settings, embedder):
        s = EmbeddingStore(redis_settings, embedder, RedisClientService(RedisConfig(host="redis.test")))
        ctx = await s.resolve(_rid(), "valve", None, None)
        await _drain(s)  # RedisNotConnected in the background write must not raise
        assert ctx.query_vector == [0.1, 0.2]

    async def test_embedding_error_propagates(self, store, embedder):
        embedder.encode_query.side_effect = RuntimeError("oci down")
        with pytest.raises(RuntimeError, match="oci down"):
            await store.resolve(_rid(), "valve", None, None)
        assert not store._pending


class TestResolvePrevious:
    async def test_refine_reuses_context_and_refreshes_expiry(self, store, embedder, fake_redis):
        prev = _rid()
        _put(fake_redis, prev, text_emb=[1.0], image_emb=[2.0], query="valve")
        ctx = await store.resolve(_rid(), None, None, prev)
        assert ctx == SearchContext(prev, "valve", [1.0], [2.0], reused=True)
        embedder.encode_query.assert_not_awaited()
        embedder.encode_image.assert_not_awaited()
        await _drain(store)
        assert fake_redis.expire_calls == [(prev, 60)]
        assert fake_redis.set_calls == []

    async def test_refine_with_empty_query_string_reuses(self, store, fake_redis):
        prev = _rid()
        _put(fake_redis, prev, text_emb=[1.0], query="valve")
        ctx = await store.resolve(_rid(), "", None, prev)
        assert ctx.reused is True and ctx.request_id == prev

    async def test_refresh_failure_is_swallowed(self, store, fake_redis):
        prev = _rid()
        _put(fake_redis, prev, text_emb=[1.0], query="valve")
        fake_redis.expire = AsyncMock(side_effect=ConnectionError("down"))
        ctx = await store.resolve(_rid(), None, None, prev)
        await _drain(store)  # must not raise
        assert ctx.reused is True

    async def test_new_query_keeps_previous_image(self, store, embedder, fake_redis):
        prev, rid = _rid(), _rid()
        _put(fake_redis, prev, text_emb=[1.0], image_emb=[2.0], query="valve")
        ctx = await store.resolve(rid, "pump", None, prev)
        assert ctx == SearchContext(rid, "pump", [0.1, 0.2], [2.0], reused=False)
        embedder.encode_query.assert_awaited_once_with("pump")
        embedder.encode_image.assert_not_awaited()
        await _drain(store)
        assert json.loads(fake_redis.data[rid]) == {"text_emb": [0.1, 0.2], "image_emb": [2.0], "query": "pump"}
        # The previous context is left untouched.
        assert json.loads(fake_redis.data[prev])["query"] == "valve"

    async def test_same_query_after_normalisation_is_not_re_embedded(self, store, embedder, fake_redis):
        prev, rid = _rid(), _rid()
        _put(fake_redis, prev, text_emb=[1.0], query="Ball Valve")
        ctx = await store.resolve(rid, "  ball   VALVE ", (b"img", "image/png"), prev)
        assert ctx.query == "  ball   VALVE "
        assert ctx.query_vector == [1.0]
        assert ctx.image_vector == [0.3, 0.4]
        embedder.encode_query.assert_not_awaited()
        embedder.encode_image.assert_awaited_once_with(b"img", "image/png")

    async def test_new_image_keeps_previous_query(self, store, embedder, fake_redis):
        prev, rid = _rid(), _rid()
        _put(fake_redis, prev, text_emb=[1.0], image_emb=[2.0], query="valve")
        ctx = await store.resolve(rid, None, (b"img", "image/png"), prev)
        assert ctx == SearchContext(rid, "valve", [1.0], [0.3, 0.4], reused=False)
        embedder.encode_query.assert_not_awaited()

    async def test_previous_without_query_embeds_new_query(self, store, embedder, fake_redis):
        prev = _rid()
        _put(fake_redis, prev, image_emb=[2.0])
        ctx = await store.resolve(_rid(), "valve", None, prev)
        assert ctx.query_vector == [0.1, 0.2]
        assert ctx.image_vector == [2.0]
        embedder.encode_query.assert_awaited_once_with("valve")

    async def test_unknown_previous_id(self, store, embedder):
        with pytest.raises(RequestIdNotFound):
            await store.resolve(_rid(), "valve", None, _rid())
        embedder.encode_query.assert_not_awaited()

    async def test_malformed_previous_id(self, store):
        with pytest.raises(RequestIdNotFound):
            await store.resolve(_rid(), None, None, "garbage")

    async def test_previous_id_with_redis_down(self, store, fake_redis):
        fake_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(EmbeddingStoreUnavailable):
            await store.resolve(_rid(), "valve", None, _rid())

    async def test_previous_id_with_redis_disabled(self, disabled_store):
        with pytest.raises(EmbeddingStoreUnavailable):
            await disabled_store.resolve(_rid(), None, None, _rid())
