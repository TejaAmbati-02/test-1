from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.response import ApiError
from services.embedding_store import EmbeddingStore
from services.redis_client import RedisClientService

NEW_ID = "11111111-1111-4111-8111-111111111111"
OLD_ID = "22222222-2222-4222-8222-222222222222"
NO_FILTERS = {"raw": {}, "clauses": [], "multi_select": {}}
SAR = {"raw": {"currency": "SAR"}, "clauses": [{"term": {"currency": "SAR"}}], "multi_select": {}}


class FakeRedis:
    """In-memory stand-in for the parts of redis.asyncio.Redis the store uses."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key], self.ttls[key] = value, ex
        return True  # redis-py returns True on a successful SET

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
        await asyncio.gather(*list(store._pending), return_exceptions=True)
    await asyncio.sleep(0)  # let the done-callbacks clear _pending


def _stored(fake_redis: FakeRedis, key: str) -> dict:
    return json.loads(fake_redis.data[key])


class TestNewContext:
    async def test_query_is_embedded_and_stored(self, store, embedder, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", None, None)
        assert ctx == {"request_id": NEW_ID, "query": "valve", "query_vector": [0.1, 0.2], "image_vector": None, "image_blank": False, "filters": NO_FILTERS, "facets": None, "facets_filters": None, "reused": False}
        embedder.encode_query.assert_awaited_once_with("valve")
        await _drain(store)
        assert _stored(fake_redis, NEW_ID) == {"text_emb": [0.1, 0.2], "image_emb": None, "query": "valve", "image_blank": False,
                                               "filters": {}, "filter_clauses": [], "multi_select_filters": {}, "facets": None, "facets_filters": None}
        assert fake_redis.ttls[NEW_ID] == 60
        assert not store._pending

    async def test_query_and_image_are_both_embedded(self, store, embedder, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", (b"img", "image/png"), None)
        assert (ctx["query_vector"], ctx["image_vector"]) == ([0.1, 0.2], [0.3, 0.4])
        embedder.encode_image.assert_awaited_once_with(b"img", "image/png")
        await _drain(store)
        assert _stored(fake_redis, NEW_ID)["image_emb"] == [0.3, 0.4]

    async def test_blank_image_is_embedded_and_stored_with_its_flag(self, store, embedder, fake_redis):
        ctx = await store.resolve(NEW_ID, None, (b"img", "image/png"), None, image_blank=True)
        assert (ctx["request_id"], ctx["image_vector"], ctx["image_blank"]) == (NEW_ID, [0.3, 0.4], True)
        await _drain(store)
        assert _stored(fake_redis, NEW_ID) == {"text_emb": None, "image_emb": [0.3, 0.4], "query": None, "image_blank": True,
                                               "filters": {}, "filter_clauses": [], "multi_select_filters": {}, "facets": None, "facets_filters": None}

    async def test_disabled_store_embeds_without_saving(self, disabled_store, embedder):
        ctx = await disabled_store.resolve(NEW_ID, "valve", None, None)
        assert ctx["query_vector"] == [0.1, 0.2]
        assert ctx["request_id"] is None  # nothing stored -> no request_id
        assert not disabled_store._pending

    async def test_write_failure_returns_no_request_id(self, store, fake_redis):
        fake_redis.set = AsyncMock(side_effect=ConnectionError("write failed"))
        ctx = await store.resolve(NEW_ID, "valve", None, None)  # must not raise
        assert ctx["query_vector"] == [0.1, 0.2]
        assert ctx["request_id"] is None

    async def test_unconfirmed_write_returns_no_request_id(self, store, fake_redis):
        fake_redis.set = AsyncMock(return_value=None)
        ctx = await store.resolve(NEW_ID, "valve", None, None)
        assert ctx["request_id"] is None


class TestRefine:
    @pytest.fixture
    def previous(self, fake_redis):
        fake_redis.data[OLD_ID] = json.dumps({"text_emb": [0.5], "image_emb": [0.6], "query": "Ball Valve"})

    async def test_previous_context_is_reused_as_is(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID)
        assert ctx == {"request_id": OLD_ID, "query": "Ball Valve", "query_vector": [0.5], "image_vector": [0.6], "image_blank": False, "filters": NO_FILTERS, "facets": None, "facets_filters": None, "reused": True}
        embedder.encode_query.assert_not_awaited()
        await _drain(store)
        assert fake_redis.ttls[OLD_ID] == 60  # expiry refreshed
        assert NEW_ID not in fake_redis.data

    async def test_new_query_replaces_text_and_keeps_image(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, "pump", None, OLD_ID)
        assert ctx == {"request_id": NEW_ID, "query": "pump", "query_vector": [0.1, 0.2], "image_vector": [0.6], "image_blank": False, "filters": NO_FILTERS, "facets": None, "facets_filters": None, "reused": False}
        await _drain(store)
        assert _stored(fake_redis, NEW_ID) == {"text_emb": [0.1, 0.2], "image_emb": [0.6], "query": "pump", "image_blank": False,
                                               "filters": {}, "filter_clauses": [], "multi_select_filters": {}, "facets": None, "facets_filters": None}

    async def test_same_query_is_the_same_search(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, "  ball   VALVE ", None, OLD_ID)
        assert ctx == {"request_id": OLD_ID, "query": "Ball Valve", "query_vector": [0.5], "image_vector": [0.6], "image_blank": False, "filters": NO_FILTERS, "facets": None, "facets_filters": None, "reused": True}
        embedder.encode_query.assert_not_awaited()
        await _drain(store)
        assert NEW_ID not in fake_redis.data  # no new context

    async def test_same_query_with_a_new_image_is_a_new_search(self, store, embedder, previous):
        ctx = await store.resolve(NEW_ID, "Ball Valve", (b"img", "image/jpeg"), OLD_ID)
        assert (ctx["request_id"], ctx["query_vector"], ctx["image_vector"]) == (NEW_ID, [0.5], [0.3, 0.4])
        embedder.encode_query.assert_not_awaited()  # the unchanged query keeps its stored vector

    async def test_new_image_replaces_image_and_keeps_query(self, store, embedder, previous):
        ctx = await store.resolve(NEW_ID, None, (b"img", "image/jpeg"), OLD_ID)
        assert (ctx["query"], ctx["query_vector"], ctx["image_vector"]) == ("Ball Valve", [0.5], [0.3, 0.4])
        embedder.encode_query.assert_not_awaited()

    @pytest.mark.parametrize("bad_id", ["nope", "AAAAAAAA-2222-4222-8222-222222222222"])  # ids are lowercase
    async def test_malformed_request_id(self, store, bad_id):
        with pytest.raises(ApiError) as e:
            await store.resolve(NEW_ID, None, None, bad_id)
        assert (e.value.status_code, e.value.detail["code"]) == (400, "REQUEST_ID_NOT_FOUND")
        assert "not a valid request_id" in e.value.detail["message"]

    async def test_expired_request_id(self, store):
        with pytest.raises(ApiError) as e:
            await store.resolve(NEW_ID, None, None, OLD_ID)
        assert (e.value.status_code, e.value.detail["code"]) == (400, "REQUEST_ID_NOT_FOUND")
        assert "not found or its search has expired" in e.value.detail["message"]

    async def test_disabled_store(self, disabled_store):
        with pytest.raises(ApiError) as e:
            await disabled_store.resolve(NEW_ID, None, None, OLD_ID)
        assert (e.value.status_code, e.value.detail["code"]) == (503, "EMBEDDING_STORE_UNAVAILABLE")
        assert "disabled" in e.value.detail["message"]

    async def test_redis_error(self, store, fake_redis):
        fake_redis.get = AsyncMock(side_effect=ConnectionError("down"))
        with pytest.raises(ApiError) as e:
            await store.resolve(NEW_ID, None, None, OLD_ID)
        assert (e.value.status_code, e.value.detail["code"]) == (503, "EMBEDDING_STORE_UNAVAILABLE")
        assert "down" in e.value.detail["message"]


class TestFilters:
    @pytest.fixture
    def previous(self, fake_redis):
        fake_redis.data[OLD_ID] = json.dumps({"text_emb": [0.5], "image_emb": None, "query": "valve",
                                              "filters": SAR["raw"], "filter_clauses": SAR["clauses"],
                                              "multi_select_filters": {}})

    async def test_filters_are_stored_with_a_new_search(self, store, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", None, None, filters=SAR)
        assert (ctx["request_id"], ctx["filters"]) == (NEW_ID, SAR)
        stored = _stored(fake_redis, NEW_ID)
        assert (stored["filters"], stored["filter_clauses"], stored["multi_select_filters"]) == (
            {"currency": "SAR"}, [{"term": {"currency": "SAR"}}], {})

    async def test_omitted_filters_reuse_the_stored_ones(self, store, embedder, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID)
        assert (ctx["request_id"], ctx["filters"], ctx["reused"]) == (OLD_ID, SAR, True)
        embedder.encode_query.assert_not_awaited()

    async def test_new_filters_replace_the_stored_ones_under_the_same_id(self, store, embedder, fake_redis, previous):
        usd = {"raw": {"currency": "USD"}, "clauses": [{"term": {"currency": "USD"}}], "multi_select": {}}
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID, filters=usd)
        assert (ctx["request_id"], ctx["query"], ctx["query_vector"], ctx["filters"], ctx["reused"]) == (
            OLD_ID, "valve", [0.5], usd, True)
        embedder.encode_query.assert_not_awaited()
        assert _stored(fake_redis, OLD_ID)["filters"] == {"currency": "USD"}
        assert NEW_ID not in fake_redis.data

    async def test_empty_filters_clear_the_stored_ones(self, store, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID, filters=NO_FILTERS)
        assert (ctx["request_id"], ctx["filters"]) == (OLD_ID, NO_FILTERS)
        assert _stored(fake_redis, OLD_ID)["filter_clauses"] == []

    async def test_new_query_keeps_the_stored_filters(self, store, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, "pump", None, OLD_ID)
        assert (ctx["request_id"], ctx["filters"]) == (NEW_ID, SAR)
        assert _stored(fake_redis, NEW_ID)["filters"] == {"currency": "SAR"}

    async def test_failed_filter_update_returns_no_request_id(self, store, fake_redis, previous):
        fake_redis.set = AsyncMock(side_effect=ConnectionError("write failed"))
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID, filters=NO_FILTERS)
        assert ctx["request_id"] is None


class TestFacets:
    FACETS = {"category": [{"id": "c1", "value": "Chemicals", "count": 7}], "sub_category": [], "product_type": [], "seller": []}

    @pytest.fixture
    def previous(self, fake_redis):
        fake_redis.data[OLD_ID] = json.dumps({"text_emb": [0.5], "image_emb": None, "query": "valve",
                                              "filters": {}, "filter_clauses": [], "multi_select_filters": {},
                                              "facets": self.FACETS, "facets_filters": {"currency": "SAR"}})

    async def test_save_facets_rewrites_the_context(self, store, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", None, None, filters=SAR)
        await store.save_facets(ctx, self.FACETS)
        stored = _stored(fake_redis, NEW_ID)
        assert (stored["facets"], stored["filters"], stored["text_emb"]) == (self.FACETS, {"currency": "SAR"}, [0.1, 0.2])
        assert stored["facets_filters"] == {"currency": "SAR"}  # the filters these facets came from
        assert fake_redis.ttls[NEW_ID] == 60

    async def test_save_facets_without_request_id_does_nothing(self, store, fake_redis):
        fake_redis.set = AsyncMock()
        await store.save_facets({"request_id": None}, self.FACETS)
        fake_redis.set.assert_not_awaited()

    async def test_save_facets_failure_is_only_logged(self, store, fake_redis):
        ctx = await store.resolve(NEW_ID, "valve", None, None)
        fake_redis.set = AsyncMock(side_effect=ConnectionError("down"))
        await store.save_facets(ctx, self.FACETS)  # must not raise

    async def test_refine_returns_the_saved_facets(self, store, previous):
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID)
        assert (ctx["facets"], ctx["facets_filters"]) == (self.FACETS, {"currency": "SAR"})

    async def test_new_filters_keep_the_saved_facets(self, store, fake_redis, previous):
        usd = {"raw": {"currency": "USD"}, "clauses": [{"term": {"currency": "USD"}}], "multi_select": {}}
        ctx = await store.resolve(NEW_ID, None, None, OLD_ID, filters=usd)
        assert (ctx["facets"], ctx["facets_filters"]) == (self.FACETS, {"currency": "SAR"})  # not the new filters
        stored = _stored(fake_redis, OLD_ID)
        assert (stored["filters"], stored["facets_filters"]) == ({"currency": "USD"}, {"currency": "SAR"})

    async def test_new_query_drops_the_saved_facets(self, store, fake_redis, previous):
        ctx = await store.resolve(NEW_ID, "pump", None, OLD_ID)
        assert (ctx["facets"], ctx["facets_filters"]) == (None, None)
        assert _stored(fake_redis, NEW_ID)["facets"] is None
