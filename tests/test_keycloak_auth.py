"""Tests for utils.keycloak_auth.require_auth — Keycloak's answer is mocked (no network)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from keycloak.exceptions import KeycloakAuthenticationError, KeycloakConnectionError

from models.response import ApiError
from utils import keycloak_auth


@pytest.fixture
def introspect(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Auth switched on, and Keycloak's token check replaced by a mock that says 'valid'."""
    monkeypatch.setattr(keycloak_auth.keycloak_auth, "enabled", True)
    mock = AsyncMock(return_value={"active": True, "preferred_username": "ali"})
    monkeypatch.setattr(keycloak_auth.keycloak_auth.client, "a_introspect", mock)
    return mock


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_exception_handler(ApiError, lambda _r, e: JSONResponse(e.detail, e.status_code, e.headers))

    @app.get("/items", dependencies=[Depends(keycloak_auth.keycloak_auth.require_auth)])
    async def items(request: Request) -> dict:
        return {"user": getattr(request.state, "user", {}).get("preferred_username")}

    return TestClient(app)


def test_client_uses_env_settings() -> None:
    c = keycloak_auth.keycloak_auth.client
    assert (c.realm_name, c.client_id, c.client_secret_key) == ("asmo", "ai-app", "test-secret")  # from conftest


def test_valid_token_runs_endpoint(client: TestClient, introspect: AsyncMock) -> None:
    resp = client.get("/items", headers={"Authorization": "Bearer AT"})
    assert (resp.status_code, resp.json()) == (200, {"user": "ali"})
    introspect.assert_awaited_once_with("AT")


@pytest.mark.parametrize("value", ["bearer AT", "BEARER  AT "])
def test_bearer_word_is_case_insensitive(client: TestClient, introspect: AsyncMock, value: str) -> None:
    assert client.get("/items", headers={"Authorization": value}).status_code == 200
    introspect.assert_awaited_once_with("AT")


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
def test_missing_or_bad_header_is_401(client: TestClient, introspect: AsyncMock, headers: dict) -> None:
    resp = client.get("/items", headers=headers)
    assert (resp.status_code, resp.json()["code"]) == (401, "UNAUTHORIZED")
    assert resp.headers["www-authenticate"] == 'Bearer realm="api"'
    introspect.assert_not_awaited()


@pytest.mark.parametrize("answer", [{"active": False}, {}, b"not json"])
def test_invalid_token_is_401(client: TestClient, introspect: AsyncMock, answer) -> None:
    introspect.return_value = answer
    resp = client.get("/items", headers={"Authorization": "Bearer old"})
    assert (resp.status_code, resp.json()["code"]) == (401, "UNAUTHORIZED")


@pytest.mark.parametrize("error, status, code", [
    (KeycloakConnectionError("Can't connect to server"), 503, "AUTH_UNAVAILABLE"),
    (KeycloakAuthenticationError("unauthorized_client", 401), 502, "AUTH_SERVER_ERROR"),
])
def test_keycloak_problems(client: TestClient, introspect: AsyncMock, error: Exception, status: int, code: str) -> None:
    introspect.side_effect = error
    resp = client.get("/items", headers={"Authorization": "Bearer AT"})
    assert (resp.status_code, resp.json()["code"]) == (status, code)


def test_auth_off_lets_everything_through(client: TestClient, introspect: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keycloak_auth.keycloak_auth, "enabled", False)
    assert client.get("/items").status_code == 200
    introspect.assert_not_awaited()


async def test_close(monkeypatch: pytest.MonkeyPatch) -> None:
    aclose = AsyncMock()
    monkeypatch.setattr(keycloak_auth.keycloak_auth.client.connection, "aclose", aclose)
    await keycloak_auth.keycloak_auth.close()
    aclose.assert_awaited_once()


def test_logs_token_hash_never_token(client: TestClient, introspect: AsyncMock, caplog: pytest.LogCaptureFixture) -> None:
    import hashlib
    import logging
    with caplog.at_level(logging.INFO, logger="mcit_logger"):
        client.get("/items", headers={"Authorization": "Bearer super-secret-token"})
    assert hashlib.sha256(b"super-secret-token").hexdigest()[:12] in caplog.text
    assert "super-secret-token" not in caplog.text
