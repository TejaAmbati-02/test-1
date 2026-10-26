from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import ResponseError

from services import embedding_store as store_module
from services.embedding_store import (
    KEY_PATTERN,
    EmbeddingIdNotFound,
    EmbeddingStore,
    EmbeddingStoreUnavailable,
    _normalise_query,
)


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self._redis = redis
        self._ops: list = []

    def exists(self, name):
        self._ops.append(lambda: int(name in self._redis.hashes))
        return self

    def hset(self, name, key, value):
        def op():
            self._redis.hashes.setdefault(name, {})[key] = value
            return 1
        self._ops.append(op)
        return self

    async def execute(self):
        return [op() for op in self._ops]


class FakeRedis:
    """In-memory stand-in for the parts of redis.asyncio.Redis the store uses."""

    def __init__(self, hexpire_supported: bool = True) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.hexpire_calls: list = []
        self.expire_calls: list = []
        self.hexpire_supported = hexpire_supported
        self.closed = False

    async def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    async def hexpire(self, name, ttl, key):
        if not self.hexpire_supported:
            raise ResponseError("unknown command 'HEXPIRE'")
        self.hexpire_calls.append((name, ttl, key))

    async def expire(self, name, ttl):
        self.expire_calls.append((name, ttl))

    async def ping(self):
        return True

    async def aclose(self):
        self.closed = True


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
def store(redis_settings, embedder, fake_redis):
    s = EmbeddingStore(redis_settings, embedder)
    s._client = fake_redis
    return s


async def _drain(store: EmbeddingStore) -> None:
    if store._pending:
        await asyncio.gather(*list(store._pending))
    await asyncio.sleep(0)  # let the done-callbacks clear _pending


def test_normalise_query_collapses_whitespace_and_case():
    assert _normalise_query("  Ball   VALVE\t2in ") == "ball valve 2in"


class TestEnabled:
    @pytest.mark.parametrize("enabled, host, expected", [
        (True, "redis.test", True),
        (True, "", False),
        (False, "redis.test", False),
    ])
    def test_enabled(self, test_settings, embedder, enabled, host, expected):
        s = EmbeddingStore(test_settings.model_copy(update={"redis_enabled": enabled, "redis_host": host}), embedder)
        assert s.enabled is expected

    def test_collection_name(self, store):
        assert store.collection == "asmo:embeddings"


class TestClientLifecycle:
    def test_redis_client_built_once_with_tls(self, monkeypatch, redis_settings, embedder):
        ctor = MagicMock(return_value="client")
        monkeypatch.setattr(store_module.redis, "Redis", ctor)
        s = EmbeddingStore(redis_settings, embedder)
        assert s._redis() == "client"
        assert s._redis() == "client"
        ctor.assert_called_once()
        kwargs = ctor.call_args.kwargs
        assert kwargs["host"] == "redis.test"
        assert kwargs["ssl"] is True
        assert kwargs["ssl_cert_reqs"] == "none"
        assert kwargs["username"] == "default"
        assert kwargs["password"] is None  # empty password -> None

    def test_redis_client_without_tls(self, monkeypatch, redis_settings, embedder):
        ctor = MagicMock()
        monkeypatch.setattr(store_module.redis, "Redis", ctor)
        EmbeddingStore(redis_settings.model_copy(update={"redis_ssl": False}), embedder)._redis()
        assert "ssl" not in ctor.call_args.kwargs

    async def test_ping_disabled(self, test_settings, embedder):
        assert await EmbeddingStore(test_settings, embedder).ping() is False

    async def test_ping_ok(self, store):
        assert await store.ping() is True

    async def test_ping_failure_returns_false(self, store, fake_redis):
        fake_redis.ping = AsyncMock(side_effect=ConnectionError("down"))
        assert await store.ping() is False

    async def test_close(self, store, fake_redis):
        await store.close()
        assert fake_redis.closed and store._client is None
        await store.close()  # no client — no-op


class TestKey:
    def test_key_is_content_addressed(self, store):
        k1 = store._key("query", b"valve")
        assert k1 == store._key("query", b"valve")
        assert k1 != store._key("query", b"pump")
        assert k1.startswith("asmo:emb:query:")
        assert store._key("image", b"valve").startswith("asmo:emb:image:")
        assert KEY_PATTERN.match(k1)

    def test_key_depends_on_model_settings(self, store, redis_settings, embedder):
        other = EmbeddingStore(redis_settings.model_copy(update={"embedding_dimension": 8}), embedder)
        assert store._key("query", b"x") != other._key("query", b"x")


class TestQueryEmbedding:
    async def test_disabled_embeds_without_id(self, test_settings, embedder):
        s = EmbeddingStore(test_settings, embedder)
        result = await s.query_embedding("valve")
        assert result.key is None and result.cached is False
        assert result.vector == [0.1, 0.2]

    async def test_miss_embeds_and_stores_in_background(self, store, embedder, fake_redis):
        result = await store.query_embedding("Ball Valve")
        assert result.cached is False
        assert result.key == store._key("query", b"ball valve")
        await _drain(store)
        record = json.loads(fake_redis.hashes["asmo:embeddings"][result.key])
        assert record["type"] == "query"
        assert record["query"] == "Ball Valve"
        assert record["embedding"] == [0.1, 0.2]
        assert record["model"] == "cohere.embed-v4.0"
        assert record["dimension"] == 4
        assert "created_at" in record
        assert fake_redis.hexpire_calls == [("asmo:embeddings", 60, result.key)]
        assert not store._pending

    async def test_hit_returns_cached_vector(self, store, embedder):
        first = await store.query_embedding("valve")
        await _drain(store)
        second = await store.query_embedding("  VALVE ")
        assert second.cached is True
        assert second.key == first.key
        assert embedder.encode_query.await_count == 1

    async def test_read_failure_falls_back_to_embedding(self, store, embedder, fake_redis):
        fake_redis.hget = AsyncMock(side_effect=ConnectionError("down"))
        result = await store.query_embedding("valve")
        assert result.key is None and result.cached is False
        assert result.vector == [0.1, 0.2]
        assert not store._pending

    async def test_background_write_failure_is_swallowed(self, store, fake_redis):
        async def boom(*_):
            raise ConnectionError("write failed")
        fake_redis.hexpire = boom
        result = await store.query_embedding("valve")
        await _drain(store)  # must not raise
        assert result.key is not None


class TestImageEmbedding:
    async def test_disabled_embeds_without_id(self, test_settings, embedder):
        result = await EmbeddingStore(test_settings, embedder).image_embedding(b"img", "image/png")
        assert result.key is None
        assert result.vector == [0.3, 0.4]
        assert result.mime == "image/png"

    async def test_miss_then_hit(self, store, embedder, fake_redis):
        first = await store.image_embedding(b"img", "image/png")
        assert first.cached is False
        await _drain(store)
        record = json.loads(fake_redis.hashes["asmo:embeddings"][first.key])
        assert record["type"] == "image"
        assert base64.b64decode(record["image_base64"]) == b"img"
        assert record["mime"] == "image/png"

        second = await store.image_embedding(b"img", "image/png")
        assert second.cached is True and second.vector == [0.3, 0.4]
        assert embedder.encode_image.await_count == 1

    async def test_read_failure_falls_back_to_embedding(self, store, embedder, fake_redis):
        fake_redis.hget = AsyncMock(side_effect=ConnectionError("down"))
        result = await store.image_embedding(b"img", "image/png")
        assert result.key is None
        embedder.encode_image.assert_awaited_once_with(b"img", "image/png")


class TestExpiry:
    async def test_falls_back_to_collection_expiry_without_hexpire(self, store, fake_redis):
        fake_redis.hexpire_supported = False
        await store._set("k1", {"type": "query"})
        assert store._field_ttl is False
        assert fake_redis.expire_calls == [("asmo:embeddings", 60)]
        # Once unsupported, HEXPIRE is not attempted again.
        fake_redis.hexpire = AsyncMock()
        await store._set("k2", {"type": "query"})
        fake_redis.hexpire.assert_not_awaited()
        assert len(fake_redis.expire_calls) == 2


class TestLoadById:
    async def test_load_query(self, store):
        stored = await store.query_embedding("valve")
        await _drain(store)
        loaded = await store.load_query(stored.key)
        assert loaded.query == "valve"
        assert loaded.vector == [0.1, 0.2]
        assert loaded.cached is True

    async def test_load_image(self, store):
        stored = await store.image_embedding(b"img", "image/jpeg")
        await _drain(store)
        loaded = await store.load_image(stored.key)
        assert loaded.data == b"img"
        assert loaded.mime == "image/jpeg"
        assert loaded.vector == [0.3, 0.4]

    @pytest.mark.parametrize("key", ["", None, "garbage", "asmo:emb:query:xyz"])
    async def test_malformed_id(self, store, key):
        with pytest.raises(EmbeddingIdNotFound, match="not a valid query embedding id"):
            await store.load_query(key)

    async def test_wrong_kind(self, store):
        key = store._key("image", b"img")
        with pytest.raises(EmbeddingIdNotFound):
            await store.load_query(key)

    async def test_expired_id(self, store):
        with pytest.raises(EmbeddingIdNotFound, match="not found or has expired"):
            await store.load_image(store._key("image", b"never stored"))

    async def test_disabled_store(self, test_settings, embedder, store):
        key = store._key("query", b"valve")
        with pytest.raises(EmbeddingStoreUnavailable):
            await EmbeddingStore(test_settings, embedder).load_query(key)

    async def test_redis_error(self, store, fake_redis):
        fake_redis.hget = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(EmbeddingStoreUnavailable, match="down"):
            await store.load_query(store._key("query", b"valve"))
