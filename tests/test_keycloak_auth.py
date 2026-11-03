"""Tests for utils.keycloak_auth — Keycloak is replaced by an httpx MockTransport."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import logging
import ssl
import time
from typing import Annotated, Any, Callable
from urllib.parse import parse_qsl

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.testclient import TestClient

from utils import keycloak_auth as keycloak_module
from utils.keycloak_auth import KeycloakAuth

INTROSPECT_URL = "https://keycloak.test/realms/asmo/protocol/openid-connect/token/introspect"


def _make_auth(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> tuple[KeycloakAuth, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    auth = KeycloakAuth("https://keycloak.test/", "asmo", "ai-app", "test-secret", **kwargs)
    auth._client = httpx.AsyncClient(transport=httpx.MockTransport(_record))
    return auth, calls


def _active(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"active": True, "sub": "user-1", "preferred_username": "ali", "exp": time.time() + 300})


def _app(auth: KeycloakAuth) -> TestClient:
    app = FastAPI()

    @app.get("/with-request")
    @auth.require_auth
    async def with_request(request: Request, q: Annotated[str, Query()] = "x") -> dict:
        return {"q": q, "user": request.state.user["preferred_username"]}

    @app.get("/without-request")
    @auth.require_auth
    async def without_request(q: str = "x") -> dict:
        return {"q": q}

    return TestClient(app)


BEARER = {"Authorization": "Bearer good-token"}


def test_valid_token_runs_endpoint_and_exposes_claims() -> None:
    auth, calls = _make_auth(_active)
    resp = _app(auth).get("/with-request", params={"q": "pump"}, headers=BEARER)
    assert resp.status_code == 200
    assert resp.json() == {"q": "pump", "user": "ali"}
    assert str(calls[0].url) == INTROSPECT_URL
    assert b"token=good-token" in calls[0].content
    assert calls[0].headers["authorization"].startswith("Basic ")


def test_endpoint_without_request_param_is_supported() -> None:
    auth, _ = _make_auth(_active)
    client = _app(auth)
    assert client.get("/without-request", params={"q": "a"}, headers=BEARER).json() == {"q": "a"}
    assert client.get("/without-request").status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
def test_missing_or_malformed_header_is_401(headers: dict) -> None:
    auth, calls = _make_auth(_active)
    resp = _app(auth).get("/with-request", headers=headers)
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "UNAUTHORIZED"
    assert resp.headers["www-authenticate"].startswith("Bearer")
    assert calls == []


def test_inactive_token_is_401() -> None:
    auth, _ = _make_auth(lambda _r: httpx.Response(200, json={"active": False}))
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == 401


@pytest.mark.parametrize(
    ("handler", "status", "code"),
    [
        (lambda _r: httpx.Response(401, json={"error": "unauthorized_client"}), 502, "AUTH_SERVER_ERROR"),
        (lambda _r: httpx.Response(200, text="not json"), 502, "AUTH_SERVER_ERROR"),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)), 503, "AUTH_UNAVAILABLE"),
        (lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)), 504, "AUTH_TIMEOUT"),
    ],
)
def test_keycloak_failures_map_to_status(handler: Callable, status: int, code: str) -> None:
    auth, _ = _make_auth(handler)
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == status
    assert resp.json()["detail"]["code"] == code


def test_valid_token_is_cached() -> None:
    auth, calls = _make_auth(_active)
    client = _app(auth)
    assert client.get("/with-request", headers=BEARER).status_code == 200
    assert client.get("/with-request", headers=BEARER).status_code == 200
    assert len(calls) == 1


def test_cache_disabled_introspects_every_time() -> None:
    auth, calls = _make_auth(_active, cache_ttl=0)
    client = _app(auth)
    client.get("/with-request", headers=BEARER)
    client.get("/with-request", headers=BEARER)
    assert len(calls) == 2


def test_disabled_auth_skips_check() -> None:
    auth, calls = _make_auth(_active, enabled=False)
    assert _app(auth).get("/without-request", params={"q": "a"}).json() == {"q": "a"}
    assert calls == []


def test_sync_endpoint_is_rejected() -> None:
    auth, _ = _make_auth(_active)
    with pytest.raises(TypeError):
        auth.require_auth(lambda: None)  # type: ignore[arg-type]


def test_from_env_reads_keycloak_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "true")
    monkeypatch.setenv("KEYCLOAK_SERVER_URL", "https://kc.example/")
    monkeypatch.setenv("KEYCLOAK_CACHE_TTL", "0")
    auth = KeycloakAuth.from_env()
    assert auth.introspect_url == "https://kc.example/realms/asmo/protocol/openid-connect/token/introspect"
    assert (auth.client_id, auth.client_secret, auth.cache_ttl, auth.enabled) == ("ai-app", "test-secret", 0, True)


@pytest.mark.parametrize("missing", ["KEYCLOAK_SERVER_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET"])
def test_from_env_requires_settings_when_enabled(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "true")
    monkeypatch.setenv(missing, "")
    with pytest.raises(RuntimeError, match=missing):
        KeycloakAuth.from_env()


def test_from_env_allows_missing_settings_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "false")
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "")
    assert KeycloakAuth.from_env().enabled is False


# Token extraction


def _request(headers: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "server": ("test", 80),
        "path": "/x",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    return Request(scope)


@pytest.mark.parametrize("value", ["Bearer tok", "bearer tok", "BEARER tok", "Bearer   tok  "])
def test_extract_token_accepts_bearer_scheme_case_insensitively(value: str) -> None:
    assert KeycloakAuth.extract_token(_request({"Authorization": value})) == "tok"


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({}, "Missing Authorization header"),
        ({"Authorization": "Token abc"}, "Authorization header must be 'Bearer <access_token>'"),
        ({"Authorization": "Bearer"}, "Authorization header must be 'Bearer <access_token>'"),
        ({"Authorization": "Bearer    "}, "Authorization header must be 'Bearer <access_token>'"),
    ],
)
def test_extract_token_rejects_missing_or_malformed(headers: dict, message: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        KeycloakAuth.extract_token(_request(headers))
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == {"code": "UNAUTHORIZED", "message": message}
    assert exc_info.value.headers == {"WWW-Authenticate": 'Bearer realm="api"'}


# Introspection


def test_introspection_request_shape() -> None:
    auth, calls = _make_auth(_active)
    _app(auth).get("/with-request", headers=BEARER)
    (call,) = calls
    assert call.method == "POST"
    assert call.headers["accept"] == "application/json"
    assert dict(parse_qsl(call.content.decode())) == {"token": "good-token", "token_type_hint": "access_token"}
    assert base64.b64decode(call.headers["authorization"].split(" ", 1)[1]) == b"ai-app:test-secret"


@pytest.mark.parametrize("body", [{"active": False}, {"active": "true"}, {"active": 1}, {}, ["active"], "active"])
def test_non_active_introspection_body_is_401(body: Any) -> None:
    auth, _ = _make_auth(lambda _r: httpx.Response(200, json=body))
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == 401
    assert resp.json()["detail"] == {"code": "UNAUTHORIZED", "message": "Invalid or expired access token"}
    assert resp.headers["www-authenticate"] == 'Bearer realm="api"'


def test_rejected_token_is_not_cached() -> None:
    auth, calls = _make_auth(lambda _r: httpx.Response(200, json={"active": False}))
    client = _app(auth)
    client.get("/with-request", headers=BEARER)
    client.get("/with-request", headers=BEARER)
    assert len(calls) == 2
    assert auth._cache == {}


def test_keycloak_server_error_has_no_www_authenticate_and_reports_status() -> None:
    auth, _ = _make_auth(lambda _r: httpx.Response(403, text="forbidden"))
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == 502
    assert resp.json()["detail"] == {"code": "AUTH_SERVER_ERROR", "message": "Authentication server returned HTTP 403"}
    assert "www-authenticate" not in resp.headers


def test_other_transport_errors_are_503() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("connection reset", request=request)

    auth, _ = _make_auth(_handler)
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "AUTH_UNAVAILABLE"


def test_connect_timeout_is_504() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    auth, _ = _make_auth(_handler)
    resp = _app(auth).get("/with-request", headers=BEARER)
    assert resp.status_code == 504
    assert resp.json()["detail"] == {"code": "AUTH_TIMEOUT", "message": "Authentication server timed out"}


async def test_introspect_returns_claims_directly() -> None:
    auth, _ = _make_auth(_active)
    claims = await auth.introspect("good-token")
    assert claims["active"] is True
    assert claims["preferred_username"] == "ali"


# Cache


def test_cache_is_keyed_by_token_hash_not_raw_token() -> None:
    auth, _ = _make_auth(_active)
    _app(auth).get("/with-request", headers=BEARER)
    assert list(auth._cache) == [hashlib.sha256(b"good-token").hexdigest()]


def test_expired_cache_entry_triggers_new_introspection() -> None:
    auth, calls = _make_auth(_active)
    client = _app(auth)
    client.get("/with-request", headers=BEARER)
    for key, (claims, _expiry) in list(auth._cache.items()):
        auth._cache[key] = (claims, time.monotonic() - 1)
    assert client.get("/with-request", headers=BEARER).status_code == 200
    assert len(calls) == 2


def test_different_tokens_are_introspected_separately() -> None:
    auth, calls = _make_auth(_active)
    client = _app(auth)
    client.get("/with-request", headers=BEARER)
    client.get("/with-request", headers={"Authorization": "Bearer other-token"})
    assert len(calls) == 2


def test_cache_ttl_is_capped_by_token_expiry() -> None:
    auth, _ = _make_auth(_active, cache_ttl=60)
    auth._cache_put("k", {"active": True, "exp": time.time() + 5})
    _claims, expires_at = auth._cache["k"]
    assert expires_at - time.monotonic() <= 5


def test_cache_uses_ttl_when_token_has_no_exp() -> None:
    auth, _ = _make_auth(_active, cache_ttl=60)
    auth._cache_put("k", {"active": True})
    _claims, expires_at = auth._cache["k"]
    assert 55 < expires_at - time.monotonic() <= 60


def test_token_already_expired_is_not_cached() -> None:
    auth, calls = _make_auth(lambda _r: httpx.Response(200, json={"active": True, "sub": "u", "exp": time.time() - 1}))
    client = _app(auth)
    assert client.get("/without-request", headers=BEARER).status_code == 200
    assert client.get("/without-request", headers=BEARER).status_code == 200
    assert len(calls) == 2
    assert auth._cache == {}


def test_cache_get_evicts_expired_entry() -> None:
    auth, _ = _make_auth(_active)
    auth._cache["k"] = ({"active": True}, time.monotonic() - 1)
    assert auth._cache_get("k") is None
    assert "k" not in auth._cache


def test_full_cache_prunes_expired_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keycloak_module, "_MAX_CACHE_ENTRIES", 2)
    auth, _ = _make_auth(_active)
    live = ({"active": True}, time.monotonic() + 60)
    auth._cache = {"old": ({"active": True}, time.monotonic() - 1), "live": live}
    auth._cache_put("new", {"active": True})
    assert set(auth._cache) == {"live", "new"}


def test_full_cache_of_live_entries_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keycloak_module, "_MAX_CACHE_ENTRIES", 2)
    auth, _ = _make_auth(_active)
    live = ({"active": True}, time.monotonic() + 60)
    auth._cache = {"a": live, "b": live}
    auth._cache_put("new", {"active": True})
    assert set(auth._cache) == {"new"}


# Logging never leaks credentials


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_logs_never_contain_raw_token_or_client_secret() -> None:
    capture = _ListHandler()
    module_logger = keycloak_module.logger
    handlers, level = list(module_logger.handlers), module_logger.level
    module_logger.handlers = [capture]
    module_logger.setLevel(logging.DEBUG)
    try:
        ok, _ = _make_auth(_active)
        _app(ok).get("/with-request", headers=BEARER)
        bad, _ = _make_auth(lambda _r: httpx.Response(200, json={"active": False}))
        _app(bad).get("/with-request", headers=BEARER)
        _app(bad).get("/with-request", headers={"Authorization": "Basic good-token"})
    finally:
        module_logger.handlers = handlers
        module_logger.setLevel(level)
    assert capture.records
    text = " ".join(f"{r.getMessage()} {getattr(r, 'error_details', '')}" for r in capture.records)
    assert "good-token" not in text
    assert "test-secret" not in text


# Decorator


def test_wrapper_signature_injects_keyword_only_request() -> None:
    auth, _ = _make_auth(_active)

    async def endpoint(q: str = "x", **extra: Any) -> dict:
        return {}

    wrapped = auth.require_auth(endpoint)
    sig = inspect.signature(wrapped)
    assert list(sig.parameters) == ["q", "_keycloak_request", "extra"]
    injected = sig.parameters["_keycloak_request"]
    assert injected.kind is inspect.Parameter.KEYWORD_ONLY
    assert injected.annotation is Request
    assert sig.parameters["q"].annotation is str
    assert sig.return_annotation is dict
    assert wrapped.__name__ == "endpoint"


def test_wrapper_signature_unchanged_when_request_declared() -> None:
    auth, _ = _make_auth(_active)

    async def endpoint(req: Request, q: str = "x") -> dict:
        return {}

    params = inspect.signature(auth.require_auth(endpoint)).parameters
    assert list(params) == ["req", "q"]
    assert params["req"].annotation is Request


async def test_wrapper_finds_request_passed_positionally() -> None:
    auth, calls = _make_auth(_active)

    @auth.require_auth
    async def endpoint(request: Request) -> str:
        return request.state.user["sub"]

    assert await endpoint(_request(BEARER)) == "user-1"
    assert len(calls) == 1


async def test_wrapper_without_request_object_is_500() -> None:
    auth, calls = _make_auth(_active)

    @auth.require_auth
    async def endpoint(request: Request) -> str:
        return "ran"

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(request=None)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail["code"] == "INTERNAL_ERROR"
    assert calls == []


async def test_wrapper_does_not_run_endpoint_when_auth_fails() -> None:
    auth, _ = _make_auth(lambda _r: httpx.Response(200, json={"active": False}))
    ran: list[bool] = []

    @auth.require_auth
    async def endpoint(request: Request) -> None:
        ran.append(True)

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(request=_request(BEARER))
    assert exc_info.value.status_code == 401
    assert ran == []


async def test_disabled_auth_does_not_set_user() -> None:
    auth, calls = _make_auth(_active, enabled=False)

    @auth.require_auth
    async def endpoint(request: Request) -> bool:
        return hasattr(request.state, "user")

    assert await endpoint(request=_request()) is False
    assert calls == []


# HTTP client lifecycle


async def test_get_client_creates_one_shared_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict] = []

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(keycloak_module.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(keycloak_module.truststore, "SSLContext", lambda protocol: ("ssl-context", protocol))
    auth = KeycloakAuth("https://keycloak.test", "asmo", "ai-app", "test-secret", timeout=2.5)

    first, second = await asyncio.gather(auth._get_client(), auth._get_client())
    assert first is second
    assert len(created) == 1
    assert created[0]["timeout"] == 2.5
    assert created[0]["verify"] == ("ssl-context", ssl.PROTOCOL_TLS_CLIENT)


async def test_close_closes_client_and_is_idempotent() -> None:
    auth, _ = _make_auth(_active)
    client = auth._client
    await auth.close()
    assert auth._client is None
    assert client.is_closed
    await auth.close()  # no client: no error


# from_env


def test_from_env_reads_timeout_and_strips_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "true")
    monkeypatch.setenv("KEYCLOAK_SERVER_URL", "  https://kc.example/  ")
    monkeypatch.setenv("KEYCLOAK_REALM", " asmo ")
    monkeypatch.setenv("KEYCLOAK_TIMEOUT", "2.5")
    monkeypatch.setenv("KEYCLOAK_CACHE_TTL", "15")
    auth = KeycloakAuth.from_env()
    assert auth.server_url == "https://kc.example"
    assert auth.realm == "asmo"
    assert auth.timeout == 2.5
    assert auth.cache_ttl == 15


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("KEYCLOAK_AUTH_ENABLED", "KEYCLOAK_TIMEOUT", "KEYCLOAK_CACHE_TTL"):
        monkeypatch.delenv(name, raising=False)
    auth = KeycloakAuth.from_env()
    assert (auth.enabled, auth.timeout, auth.cache_ttl) == (True, 5.0, 60)


def test_from_env_lists_every_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYCLOAK_AUTH_ENABLED", "true")
    monkeypatch.delenv("KEYCLOAK_REALM", raising=False)
    monkeypatch.setenv("KEYCLOAK_CLIENT_SECRET", "   ")
    with pytest.raises(RuntimeError, match="KEYCLOAK_REALM, KEYCLOAK_CLIENT_SECRET"):
        KeycloakAuth.from_env()


@pytest.mark.parametrize("name", ["KEYCLOAK_TIMEOUT", "KEYCLOAK_CACHE_TTL"])
def test_from_env_invalid_number_raises(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "abc")
    with pytest.raises(ValueError):
        KeycloakAuth.from_env()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" TRUE ", True), ("yes", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("off", False), ("", False), ("maybe", False)],
)
def test_env_bool(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("KC_TEST_FLAG", value)
    assert keycloak_module._env_bool("KC_TEST_FLAG", True) is expected


@pytest.mark.parametrize("default", [True, False])
def test_env_bool_default_when_unset(monkeypatch: pytest.MonkeyPatch, default: bool) -> None:
    monkeypatch.delenv("KC_TEST_FLAG", raising=False)
    assert keycloak_module._env_bool("KC_TEST_FLAG", default) is default


def test_module_singleton_follows_env() -> None:
    # conftest sets KEYCLOAK_AUTH_ENABLED=false before the module is imported.
    assert keycloak_module.keycloak_auth.enabled is False
    assert keycloak_module.require_auth == keycloak_module.keycloak_auth.require_auth
