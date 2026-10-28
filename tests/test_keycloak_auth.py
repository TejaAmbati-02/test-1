"""Tests for utils.keycloak_auth — Keycloak is replaced by an httpx MockTransport."""
from __future__ import annotations

import time
from typing import Annotated, Any, Callable

import httpx
import pytest
from fastapi import FastAPI, Query, Request
from fastapi.testclient import TestClient

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
