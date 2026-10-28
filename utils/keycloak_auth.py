"""Keycloak bearer-token authentication for the API endpoints.

Callers send ``Authorization: Bearer <access_token>`` issued by Keycloak. Each token
is checked against Keycloak's token-introspection endpoint using this service's
confidential client (``client_id`` / ``client_secret``); an active token lets the
endpoint run, anything else is rejected before the endpoint body executes.

Usage — put ``@require_auth`` *below* the route decorator::

    from utils.keycloak_auth import require_auth

    @router.get("/items")
    @require_auth
    async def list_items(request: Request) -> ...:
        claims = request.state.user   # introspected token claims

Status codes returned by the decorator:
    401 UNAUTHORIZED      — missing/malformed Authorization header, or token inactive/expired
    502 AUTH_SERVER_ERROR — Keycloak answered with a non-200 (e.g. wrong client credentials)
    503 AUTH_UNAVAILABLE  — Keycloak could not be reached
    504 AUTH_TIMEOUT      — Keycloak did not answer in time
Otherwise the endpoint runs and returns its own status (200 on success).

Configuration (read from ``.env``; real environment variables take precedence):
    KEYCLOAK_SERVER_URL, KEYCLOAK_REALM, KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET — required
        while auth is enabled; startup fails with a clear error if any is missing.
    KEYCLOAK_TIMEOUT (seconds, default 5), KEYCLOAK_CACHE_TTL (seconds, default 60, 0 = off),
    KEYCLOAK_AUTH_ENABLED (default true; set false only for local dev/tests).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import os
import time
import typing
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException, Request

from utils.logger import logger

# The repo's .env, regardless of the working directory the app is started from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

# Name of the Request parameter injected into endpoints that don't declare one.
_INJECTED_REQUEST_PARAM = "_keycloak_request"
# Upper bound on cached introspection results so the cache can't grow without limit.
_MAX_CACHE_ENTRIES = 10_000


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class KeycloakAuth:
    """Validates Keycloak access tokens via the token-introspection endpoint."""

    def __init__(
        self,
        server_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        timeout: float = 5.0,
        cache_ttl: int = 60,
        enabled: bool = True,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.realm = realm
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.enabled = enabled
        self.introspect_url = f"{self.server_url}/realms/{realm}/protocol/openid-connect/token/introspect"
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        # sha256(token) -> (claims, monotonic expiry). Raw tokens are never kept in memory.
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    @classmethod
    def from_env(cls) -> KeycloakAuth:
        """Build from KEYCLOAK_* in the environment / .env; the connection settings are required."""
        enabled = _env_bool("KEYCLOAK_AUTH_ENABLED", True)
        required = ("KEYCLOAK_SERVER_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if enabled and missing:
            raise RuntimeError(f"Keycloak auth is enabled but {', '.join(missing)} is not set in .env")
        return cls(
            server_url=os.environ.get("KEYCLOAK_SERVER_URL", "").strip(),
            realm=os.environ.get("KEYCLOAK_REALM", "").strip(),
            client_id=os.environ.get("KEYCLOAK_CLIENT_ID", "").strip(),
            client_secret=os.environ.get("KEYCLOAK_CLIENT_SECRET", "").strip(),
            timeout=float(os.environ.get("KEYCLOAK_TIMEOUT", "5")),
            cache_ttl=int(os.environ.get("KEYCLOAK_CACHE_TTL", "60")),
            enabled=enabled,
        )

    # HTTP client

    async def _get_client(self) -> httpx.AsyncClient:
        # One pooled client for the process so each check reuses the warm TLS connection.
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # Errors

    @staticmethod
    def _error(status: int, code: str, message: str) -> HTTPException:
        headers = {"WWW-Authenticate": 'Bearer realm="api"'} if status == 401 else None
        return HTTPException(status_code=status, detail={"code": code, "message": message}, headers=headers)

    # Token handling

    @classmethod
    def extract_token(cls, request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if not header:
            raise cls._error(401, "UNAUTHORIZED", "Missing Authorization header")
        if scheme.lower() != "bearer" or not token.strip():
            raise cls._error(401, "UNAUTHORIZED", "Authorization header must be 'Bearer <access_token>'")
        return token.strip()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        claims, expires_at = entry
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return None
        return claims

    def _cache_put(self, key: str, claims: dict[str, Any]) -> None:
        if self.cache_ttl <= 0:
            return
        ttl = float(self.cache_ttl)
        # Never cache past the token's own expiry.
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            ttl = min(ttl, exp - time.time())
        if ttl <= 0:
            return
        if len(self._cache) >= _MAX_CACHE_ENTRIES:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                self._cache.clear()
        self._cache[key] = (claims, time.monotonic() + ttl)

    async def introspect(self, token: str) -> dict[str, Any]:
        """Return the token's claims if Keycloak reports it active; raise HTTPException otherwise."""
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        client = await self._get_client()
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                self.introspect_url,
                data={"token": token, "token_type_hint": "access_token"},
                auth=(self.client_id, self.client_secret),
                headers={"Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            logger.error(f"Keycloak introspection timed out after {self.timeout}s: {exc}")
            raise self._error(504, "AUTH_TIMEOUT", "Authentication server timed out")
        except httpx.HTTPError as exc:
            logger.error(f"Keycloak introspection request failed: {exc}")
            raise self._error(503, "AUTH_UNAVAILABLE", "Authentication server is unavailable")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            # 401/403 here means Keycloak rejected *our* client credentials, not the caller's token.
            logger.error(f"Keycloak introspection returned HTTP {resp.status_code} in {elapsed_ms:.1f}ms: {resp.text[:200]}")
            raise self._error(502, "AUTH_SERVER_ERROR", f"Authentication server returned HTTP {resp.status_code}")

        try:
            claims = resp.json()
        except ValueError:
            logger.error("Keycloak introspection returned a non-JSON body")
            raise self._error(502, "AUTH_SERVER_ERROR", "Authentication server returned an invalid response")

        if not isinstance(claims, dict) or claims.get("active") is not True:
            logger.warning(f"Rejected inactive/invalid token ({elapsed_ms:.1f}ms)")
            raise self._error(401, "UNAUTHORIZED", "Invalid or expired access token")

        logger.info(f"Token validated for '{claims.get('preferred_username', claims.get('sub', '?'))}' in {elapsed_ms:.1f}ms")
        self._cache_put(key, claims)
        return claims

    async def authenticate(self, request: Request) -> dict[str, Any]:
        """Validate the request's bearer token and store the claims on ``request.state.user``."""
        claims = await self.introspect(self.extract_token(request))
        request.state.user = claims
        return claims

    # Decorator

    def require_auth(self, func: F) -> F:
        """Decorator that rejects the request unless it carries a valid Keycloak token."""
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@require_auth only supports async endpoints, got {func.__qualname__}")

        sig = inspect.signature(func)
        # Resolve string annotations (``from __future__ import annotations``) in the
        # endpoint's own module so FastAPI sees real types on the wrapper's signature.
        try:
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception:
            hints = {}
        params = [p.replace(annotation=hints.get(p.name, p.annotation)) for p in sig.parameters.values()]
        request_param = next((p.name for p in params if p.annotation is Request), None)

        inject = request_param is None
        if inject:
            # Endpoint doesn't take a Request — add one so FastAPI passes it to the wrapper.
            injected = inspect.Parameter(_INJECTED_REQUEST_PARAM, inspect.Parameter.KEYWORD_ONLY, annotation=Request)
            insert_at = next((i for i, p in enumerate(params) if p.kind == inspect.Parameter.VAR_KEYWORD), len(params))
            params.insert(insert_at, injected)
            request_param = _INJECTED_REQUEST_PARAM

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = kwargs.pop(_INJECTED_REQUEST_PARAM) if inject else kwargs.get(request_param)
            if request is None:
                request = next((a for a in args if isinstance(a, Request)), None)
            if self.enabled:
                if request is None:
                    raise self._error(500, "INTERNAL_ERROR", "Authentication could not access the request")
                await self.authenticate(request)
            return await func(*args, **kwargs)

        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=hints.get("return", sig.return_annotation),
        )
        return typing.cast(F, wrapper)


keycloak_auth = KeycloakAuth.from_env()
require_auth = keycloak_auth.require_auth
