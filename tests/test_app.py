"""Tests for app.py: health, latency probes, lifespan, heartbeats, middleware, handlers."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import app as app_module
from app import _latency_response, _measure_latency
from services.embedder import embedding_service
from services.embedding_store import SearchContext, embedding_store
from services.opensearch import os_client
from services.redis_client import redis_client
from services.search import search_service
from utils.keycloak_auth import keycloak_auth


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


class TestHealth:
    @pytest.fixture(autouse=True)
    def _deps(self, monkeypatch):
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(return_value="asmo-cluster"))
        monkeypatch.setattr(embedding_service, "_client", object())

    def test_ok_with_redis_disabled(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "ok",
            "opensearch_connected": True,
            "opensearch_cluster": "asmo-cluster",
            "embedding_model_loaded": True,
            "embedding_model_name": "cohere.embed-v4.0",
            "redis_connected": None,
        }

    def test_redis_not_pinged_when_disabled(self, client, monkeypatch):
        ping = AsyncMock(return_value=True)
        monkeypatch.setattr(redis_client, "ping", ping)
        client.get("/health")
        ping.assert_not_awaited()

    def test_ok_with_redis_connected(self, client, enable_redis, monkeypatch):
        monkeypatch.setattr(redis_client, "ping", AsyncMock(return_value=True))
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["redis_connected"] is True

    def test_degraded_when_redis_down(self, client, enable_redis, monkeypatch):
        monkeypatch.setattr(redis_client, "ping", AsyncMock(return_value=False))
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["redis_connected"] is False

    def test_degraded_when_opensearch_down(self, client, monkeypatch):
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(return_value=None))
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["opensearch_connected"] is False
        assert body["opensearch_cluster"] is None

    def test_degraded_when_model_not_loaded(self, client, monkeypatch):
        monkeypatch.setattr(embedding_service, "_client", None)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["embedding_model_loaded"] is False

    def test_unexpected_error_is_500(self, client, monkeypatch):
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(side_effect=RuntimeError("boom")))
        resp = client.get("/health")
        assert resp.status_code == 500
        assert resp.json() == {"code": "INTERNAL_ERROR", "message": "boom"}


# --------------------------------------------------------------------------
# Latency helpers and endpoints
# --------------------------------------------------------------------------


class TestLatencyHelpers:
    async def test_measure_latency_all_succeed(self):
        call = AsyncMock()
        times, error = await _measure_latency(call, 3)
        assert len(times) == 3 and error is None
        assert all(t >= 0 for t in times)
        assert call.await_count == 3

    async def test_measure_latency_stops_on_first_error(self):
        call = AsyncMock(side_effect=[None, RuntimeError("down"), None])
        times, error = await _measure_latency(call, 3)
        assert len(times) == 1
        assert error == "sample 2 failed: down"
        assert call.await_count == 2

    def test_latency_response_stats(self):
        r = _latency_response("oci", "ep", 3, [1.0, 2.0, 6.0], None)
        assert (r.min_ms, r.avg_ms, r.max_ms) == (1.0, 3.0, 6.0)
        assert r.ok is True and r.error is None

    def test_latency_response_without_samples(self):
        r = _latency_response("redis", "ep", 2, [], "failed")
        assert (r.min_ms, r.avg_ms, r.max_ms) == (0.0, 0.0, 0.0)
        assert r.ok is False and r.error == "failed"

    def test_latency_response_partial_is_not_ok(self):
        assert _latency_response("x", "ep", 3, [1.0], None).ok is False

    def test_latency_response_avg_is_rounded(self):
        assert _latency_response("x", "ep", 3, [1.0, 1.0, 2.0], None).avg_ms == 1.3


class TestLatencyEndpoints:
    @pytest.fixture
    def os_raw(self, monkeypatch):
        raw = AsyncMock()
        monkeypatch.setattr(os_client, "get_client", AsyncMock(return_value=raw))
        return raw

    def test_opensearch(self, client, os_raw):
        body = client.get("/health/latency/opensearch", params={"samples": 3}).json()
        assert body["target"] == "opensearch"
        assert body["endpoint"] == "opensearch.test:9200"
        assert body["samples"] == 3 and len(body["samples_ms"]) == 3
        assert body["ok"] is True
        assert os_raw.info.await_count == 3

    @pytest.mark.parametrize("requested, expected", [(0, 1), (-5, 1), (100, 20)])
    def test_samples_are_clamped(self, client, os_raw, requested, expected):
        body = client.get("/health/latency/opensearch", params={"samples": requested}).json()
        assert body["samples"] == expected
        assert os_raw.info.await_count == expected

    def test_default_sample_count(self, client, os_raw):
        assert client.get("/health/latency/opensearch").json()["samples"] == 5

    def test_opensearch_failure(self, client, os_raw):
        os_raw.info.side_effect = ConnectionError("refused")
        body = client.get("/health/latency/opensearch", params={"samples": 2}).json()
        assert body["ok"] is False
        assert body["error"] == "sample 1 failed: refused"
        assert body["samples_ms"] == []

    def test_redis_disabled(self, client):
        body = client.get("/health/latency/redis").json()
        assert body["target"] == "redis"
        assert body["ok"] is False
        assert "disabled" in body["error"]

    def test_redis_disabled_samples_are_clamped(self, client):
        body = client.get("/health/latency/redis", params={"samples": 50}).json()
        assert body["samples"] == 20
        assert body["samples_ms"] == []
        assert (body["min_ms"], body["avg_ms"], body["max_ms"]) == (0.0, 0.0, 0.0)

    def test_redis_enabled(self, client, enable_redis, monkeypatch):
        fake = MagicMock()
        fake.ping = AsyncMock(return_value=True)
        monkeypatch.setattr(redis_client, "_client", fake)
        body = client.get("/health/latency/redis", params={"samples": 4}).json()
        assert body["ok"] is True
        assert body["endpoint"] == "redis.test:443"
        assert len(body["samples_ms"]) == 4
        assert fake.ping.await_count == 4

    def test_redis_enabled_but_not_connected(self, client, enable_redis, monkeypatch):
        # Enabled but connect() never ran: redis_client.client raises RedisNotConnected.
        monkeypatch.setattr(redis_client, "_client", None)
        body = client.get("/health/latency/redis", params={"samples": 3}).json()
        assert body["ok"] is False
        assert body["samples_ms"] == []
        assert body["error"].startswith("sample 1 failed: Redis client is not connected")

    def test_redis_ping_failure_midway(self, client, enable_redis, monkeypatch):
        fake = MagicMock()
        fake.ping = AsyncMock(side_effect=[True, ConnectionError("reset")])
        monkeypatch.setattr(redis_client, "_client", fake)
        body = client.get("/health/latency/redis", params={"samples": 3}).json()
        assert body["ok"] is False
        assert len(body["samples_ms"]) == 1
        assert body["error"] == "sample 2 failed: reset"

    def test_oci(self, client, monkeypatch):
        encode = AsyncMock(return_value=[0.1])
        monkeypatch.setattr(embedding_service, "encode_query", encode)
        body = client.get("/health/latency/oci", params={"samples": 2}).json()
        assert body["target"] == "oci"
        assert body["endpoint"] == "https://genai.test"
        assert body["ok"] is True
        encode.assert_awaited_with("latency check")
        assert encode.await_count == 2

    def test_oci_failure(self, client, monkeypatch):
        monkeypatch.setattr(embedding_service, "encode_query", AsyncMock(side_effect=RuntimeError("throttled")))
        body = client.get("/health/latency/oci").json()
        assert body["ok"] is False
        assert "throttled" in body["error"]

    def test_samples_must_be_an_integer(self, client):
        resp = client.get("/health/latency/oci", params={"samples": "many"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "VALIDATION_ERROR"
        assert resp.json()["message"].startswith("query.samples:")


# --------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# --------------------------------------------------------------------------


@pytest.fixture
def startup_deps(monkeypatch):
    deps = {
        "load": AsyncMock(),
        "encode_query": AsyncMock(return_value=[0.1]),
        "os_ping": AsyncMock(return_value=True),
        "os_close": AsyncMock(),
        "redis_connect": AsyncMock(),
        "redis_ping": AsyncMock(return_value=True),
        "redis_close": AsyncMock(),
        "keycloak_close": AsyncMock(),
        "setup_pipeline": AsyncMock(),
    }
    monkeypatch.setattr(embedding_service, "load", deps["load"])
    monkeypatch.setattr(embedding_service, "encode_query", deps["encode_query"])
    monkeypatch.setattr(os_client, "ping", deps["os_ping"])
    monkeypatch.setattr(os_client, "close", deps["os_close"])
    monkeypatch.setattr(redis_client, "connect", deps["redis_connect"])
    monkeypatch.setattr(redis_client, "ping", deps["redis_ping"])
    monkeypatch.setattr(redis_client, "close", deps["redis_close"])
    monkeypatch.setattr(keycloak_auth, "close", deps["keycloak_close"])
    monkeypatch.setattr(search_service, "setup_pipeline", deps["setup_pipeline"])
    return deps


def _heartbeat_tasks() -> list[asyncio.Task]:
    names = ("_opensearch_heartbeat", "_oci_embed_heartbeat")
    return [t for t in asyncio.all_tasks() if t.get_coro().__name__ in names]


class TestLifespan:
    async def test_startup_and_shutdown(self, startup_deps):
        async with app_module.lifespan(app_module.app):
            startup_deps["load"].assert_awaited_once()
            startup_deps["encode_query"].assert_awaited_once_with("warmup")
            startup_deps["os_ping"].assert_awaited_once()
            startup_deps["setup_pipeline"].assert_awaited_once()
            startup_deps["redis_connect"].assert_not_awaited()  # Redis disabled
            startup_deps["redis_ping"].assert_not_awaited()
            tasks = _heartbeat_tasks()
            assert len(tasks) == 2
            assert not any(t.done() for t in tasks)
        assert all(t.cancelled() for t in tasks)
        startup_deps["os_close"].assert_awaited_once()
        startup_deps["redis_close"].assert_awaited_once()
        startup_deps["keycloak_close"].assert_awaited_once()

    async def test_redis_is_connected_then_pinged_when_enabled(self, startup_deps, enable_redis):
        async with app_module.lifespan(app_module.app):
            startup_deps["redis_connect"].assert_awaited_once()
            startup_deps["redis_ping"].assert_awaited_once()

    async def test_redis_unreachable_does_not_block_startup(self, startup_deps, enable_redis):
        startup_deps["redis_ping"].return_value = False
        async with app_module.lifespan(app_module.app):
            startup_deps["setup_pipeline"].assert_awaited_once()

    async def test_warmup_failure_does_not_block_startup(self, startup_deps):
        startup_deps["encode_query"].side_effect = RuntimeError("cold")
        async with app_module.lifespan(app_module.app):
            startup_deps["setup_pipeline"].assert_awaited_once()

    async def test_opensearch_ping_false_does_not_block_startup(self, startup_deps):
        startup_deps["os_ping"].return_value = False
        async with app_module.lifespan(app_module.app):
            startup_deps["setup_pipeline"].assert_awaited_once()

    @pytest.mark.parametrize("failing", ["load", "setup_pipeline", "os_ping"])
    async def test_startup_failure_propagates(self, startup_deps, failing):
        startup_deps[failing].side_effect = RuntimeError("fatal")
        with pytest.raises(RuntimeError, match="fatal"):
            async with app_module.lifespan(app_module.app):
                pass
        assert _heartbeat_tasks() == []

    async def test_redis_connect_failure_propagates(self, startup_deps, enable_redis):
        startup_deps["redis_connect"].side_effect = RuntimeError("bad tls")
        with pytest.raises(RuntimeError, match="bad tls"):
            async with app_module.lifespan(app_module.app):
                pass
        startup_deps["setup_pipeline"].assert_not_awaited()

    async def test_shutdown_failure_propagates_after_stopping_heartbeats(self, startup_deps):
        startup_deps["os_close"].side_effect = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            async with app_module.lifespan(app_module.app):
                tasks = _heartbeat_tasks()
        assert tasks and all(t.cancelled() for t in tasks)
        startup_deps["redis_close"].assert_not_awaited()


# --------------------------------------------------------------------------
# Heartbeats
# --------------------------------------------------------------------------


class TestHeartbeats:
    @pytest.fixture
    def fake_sleep(self, monkeypatch):
        """asyncio.sleep stand-in: returns twice, then cancels the loop."""
        sleep = AsyncMock(side_effect=[None, None, asyncio.CancelledError()])
        monkeypatch.setattr(app_module.asyncio, "sleep", sleep)
        return sleep

    async def test_opensearch_heartbeat_survives_errors(self, monkeypatch, fake_sleep):
        ping = AsyncMock(side_effect=[ConnectionError("blip"), True])
        monkeypatch.setattr(os_client, "ping", ping)
        with pytest.raises(asyncio.CancelledError):
            await app_module._opensearch_heartbeat(7)
        assert ping.await_count == 2
        fake_sleep.assert_awaited_with(7)

    async def test_oci_heartbeat_survives_errors(self, monkeypatch, fake_sleep):
        encode = AsyncMock(side_effect=[RuntimeError("cold"), [0.1]])
        monkeypatch.setattr(embedding_service, "encode_query", encode)
        with pytest.raises(asyncio.CancelledError):
            await app_module._oci_embed_heartbeat(60)
        assert encode.await_count == 2
        encode.assert_awaited_with("warmup")
        fake_sleep.assert_awaited_with(60)

    async def test_cancel_during_ping_propagates(self, monkeypatch):
        monkeypatch.setattr(app_module.asyncio, "sleep", AsyncMock(return_value=None))
        ping = AsyncMock(side_effect=asyncio.CancelledError())
        monkeypatch.setattr(os_client, "ping", ping)
        with pytest.raises(asyncio.CancelledError):
            await app_module._opensearch_heartbeat(1)
        ping.assert_awaited_once()


# --------------------------------------------------------------------------
# Middleware and exception handlers
# --------------------------------------------------------------------------


class TestMiddleware:
    def test_request_context_is_set_and_cleared(self, client, monkeypatch):
        # TestClient runs the app on another thread, so spy on the context calls
        # instead of reading the thread-local directly.
        set_ctx, clear_ctx = MagicMock(), MagicMock()
        monkeypatch.setattr(app_module, "set_log_context", set_ctx)
        monkeypatch.setattr(app_module, "clear_log_context", clear_ctx)
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(return_value="c"))
        client.get("/health")
        set_ctx.assert_called_once()
        assert len(set_ctx.call_args.args[0]) == 36  # a UUID4 string
        clear_ctx.assert_called_once()

    def test_context_is_cleared_even_when_the_request_fails(self, client, monkeypatch):
        clear_ctx = MagicMock()
        monkeypatch.setattr(app_module, "clear_log_context", clear_ctx)
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(side_effect=RuntimeError("boom")))
        assert client.get("/health").status_code == 500
        clear_ctx.assert_called_once()

    def test_each_request_gets_a_unique_id(self, client, monkeypatch):
        from tests.conftest import make_search_response

        monkeypatch.setattr(
            embedding_store, "resolve",
            AsyncMock(side_effect=lambda rid, query, image, prev: SearchContext(rid, None, None, None, False)),
        )
        monkeypatch.setattr(
            search_service, "filter_browse",
            AsyncMock(side_effect=lambda **kw: make_search_response(request_id=kw["request_id"])),
        )
        data = {"channel": "web", "language": "en", "filters": '{"currency": "SAR"}'}
        # /v1/search only accepts multipart; an empty image part forces it and counts as "no image".
        files = {"image": ("a.png", b"", "image/png")}
        ids = set()
        for _ in range(3):
            resp = client.post("/v1/search", data=data, files=files)
            assert resp.status_code == 200, resp.text
            ids.add(resp.json()["request_id"])
        assert len(ids) == 3
        assert all(len(i) == 36 for i in ids)  # the middleware's UUID4 request id

    def test_cors_headers(self, client, monkeypatch):
        monkeypatch.setattr(os_client, "cluster_name", AsyncMock(return_value="c"))
        resp = client.get("/health", headers={"Origin": "https://shop.example"})
        assert resp.headers["access-control-allow-origin"] == "*"


class TestExceptionHandlers:
    def test_unknown_route_uses_error_contract(self, client):
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404
        assert resp.json() == {"code": "HTTP_ERROR", "message": "Not Found"}

    def test_method_not_allowed(self, client):
        resp = client.get("/v1/search")
        assert resp.status_code == 405
        assert resp.json()["code"] == "HTTP_ERROR"

    async def test_http_handler_passes_structured_detail_through(self):
        exc = HTTPException(status_code=409, detail={"code": "CONFLICT", "message": "taken"})
        resp = await app_module.http_exception_handler(MagicMock(), exc)
        assert resp.status_code == 409
        assert json.loads(resp.body) == {"code": "CONFLICT", "message": "taken"}

    async def test_http_handler_wraps_plain_detail(self):
        resp = await app_module.http_exception_handler(MagicMock(), StarletteHTTPException(403, "nope"))
        assert resp.status_code == 403
        assert json.loads(resp.body) == {"code": "HTTP_ERROR", "message": "nope"}

    async def test_http_handler_wraps_partial_dict_detail(self):
        resp = await app_module.http_exception_handler(MagicMock(), HTTPException(400, detail={"code": "X"}))
        assert json.loads(resp.body)["code"] == "HTTP_ERROR"

    async def test_http_handler_5xx(self):
        resp = await app_module.http_exception_handler(MagicMock(), StarletteHTTPException(503, "down"))
        assert resp.status_code == 503
        assert json.loads(resp.body) == {"code": "HTTP_ERROR", "message": "down"}

    async def test_validation_handler_formats_location(self):
        exc = RequestValidationError([{"loc": ("body", "page"), "msg": "Input should be greater than 0", "type": "x"}])
        resp = await app_module.validation_exception_handler(MagicMock(), exc)
        assert resp.status_code == 400
        assert json.loads(resp.body) == {"code": "VALIDATION_ERROR", "message": "page: Input should be greater than 0"}

    async def test_validation_handler_body_only_location_uses_bare_message(self):
        exc = RequestValidationError([{"loc": ("body",), "msg": "Field required", "type": "missing"}])
        resp = await app_module.validation_exception_handler(MagicMock(), exc)
        assert json.loads(resp.body) == {"code": "VALIDATION_ERROR", "message": "Field required"}

    async def test_validation_handler_reports_only_first_error(self):
        exc = RequestValidationError([
            {"loc": ("query", "q"), "msg": "too short", "type": "x"},
            {"loc": ("query", "channel"), "msg": "missing", "type": "y"},
        ])
        resp = await app_module.validation_exception_handler(MagicMock(), exc)
        assert json.loads(resp.body)["message"] == "query.q: too short"

    async def test_validation_handler_without_errors(self):
        resp = await app_module.validation_exception_handler(MagicMock(), RequestValidationError([]))
        assert json.loads(resp.body) == {"code": "VALIDATION_ERROR", "message": "Request validation failed"}

    async def test_global_handler(self):
        resp = await app_module.global_exception_handler(MagicMock(), ValueError("bad"))
        assert resp.status_code == 500
        assert json.loads(resp.body) == {"code": "INTERNAL_ERROR", "message": "bad"}


def test_app_metadata_and_routes():
    assert app_module.app.title == "ASMO AI Search API"
    assert app_module.app.version == "2.1.0"
    # Read paths from the OpenAPI schema: newer FastAPI keeps included routers as
    # nested _IncludedRouter entries in app.routes, which have no .path.
    paths = set(app_module.app.openapi()["paths"])
    assert {
        "/health", "/health/latency/opensearch", "/health/latency/redis", "/health/latency/oci",
        "/indices", "/v1/search", "/v1/search/suggestions",
    } <= paths
    assert app_module.app.docs_url == "/docs"


def test_heartbeat_intervals():
    assert app_module.OPENSEARCH_HEARTBEAT_INTERVAL == 5
    assert app_module.OCI_EMBED_HEARTBEAT_INTERVAL == 60  # OCI_EMBED_HEARTBEAT_INTERVAL in conftest


def test_openapi_documents_multipart_search_body(client):
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/v1/search"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body["required"] == ["channel", "language"]
    assert body["properties"]["image"]["format"] == "binary"
    assert "request_id" in body["properties"]
