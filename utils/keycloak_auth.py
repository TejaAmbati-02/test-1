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

import functools
import hashlib
import inspect
import os
import ssl
import time
import typing
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import truststore
from dotenv import load_dotenv
from fastapi import Request

from models.response import ApiError
from utils.logger import logger

# The repo's .env, regardless of the working directory the app is started from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

# Name of the Request parameter added to endpoints that don't declare one.
_INJECTED_REQUEST_PARAM = "_keycloak_request"
# Upper bound on cached introspection results so the cache can't grow without limit.
_MAX_CACHE_ENTRIES = 10_000
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": 'Bearer realm="api"'}


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
        # sha256(token) -> (claims, monotonic expiry). Raw tokens are never kept in memory.
        self._cache: dict[str, tuple[dict[str, Any], float]] = {}

    @classmethod
    def from_env(cls) -> KeycloakAuth:
        """Build from KEYCLOAK_* in the environment / .env. Sync: it runs once, at import."""
        env = {name: value.strip() for name, value in os.environ.items() if name.startswith("KEYCLOAK_")}
        enabled = env.get("KEYCLOAK_AUTH_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        required = ("KEYCLOAK_SERVER_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET")
        missing = [name for name in required if not env.get(name)]
        if enabled and missing:
            logger.error(f"[KeycloakAuth] auth enabled but missing settings: {', '.join(missing)}",
                         extra={"status": "error", "error_details": f"missing: {', '.join(missing)}"})
            raise RuntimeError(f"Keycloak auth is enabled but {', '.join(missing)} is not set in .env")

        auth = cls(
            server_url=env.get("KEYCLOAK_SERVER_URL", ""),
            realm=env.get("KEYCLOAK_REALM", ""),
            client_id=env.get("KEYCLOAK_CLIENT_ID", ""),
            client_secret=env.get("KEYCLOAK_CLIENT_SECRET", ""),
            timeout=float(env.get("KEYCLOAK_TIMEOUT", "5")),
            cache_ttl=int(env.get("KEYCLOAK_CACHE_TTL", "60")),
            enabled=enabled,
        )
        if not enabled:
            logger.warning("[KeycloakAuth] Keycloak auth is DISABLED (KEYCLOAK_AUTH_ENABLED=false)")
        logger.info(f"[KeycloakAuth] configured — enabled={auth.enabled} server={auth.server_url or '-'} "
                    f"realm={auth.realm or '-'} timeout={auth.timeout}s cache_ttl={auth.cache_ttl}s")
        return auth

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("[KeycloakAuth] HTTP client closed")

    async def authenticate(self, request: Request) -> dict[str, Any]:
        """Validate the request's bearer token and store its claims on ``request.state.user``."""
        where = f"{request.method} {request.url.path}"
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if not header:
            logger.warning(f"[KeycloakAuth] missing Authorization header on {where}")
            raise ApiError(401, "UNAUTHORIZED", "Missing Authorization header", _UNAUTHORIZED_HEADERS)
        if scheme.lower() != "bearer" or not token.strip():
            # Log only the scheme — never the credential itself.
            logger.warning(f"[KeycloakAuth] malformed Authorization header on {where} (scheme='{scheme[:20]}')")
            raise ApiError(401, "UNAUTHORIZED", "Authorization header must be 'Bearer <access_token>'",
                           _UNAUTHORIZED_HEADERS)

        claims = await self.introspect(token.strip())
        request.state.user = claims
        return claims

    async def introspect(self, token: str) -> dict[str, Any]:
        """Return the token's claims if Keycloak reports it active; raise ApiError otherwise.

        Active tokens are cached for KEYCLOAK_CACHE_TTL seconds (never past the token's own expiry).
        """
        key = hashlib.sha256(token.encode()).hexdigest()  # only this hash is kept / logged, never the token

        cached = self._cache.get(key)
        if cached is not None:
            claims, expires_at = cached
            if time.monotonic() < expires_at:
                return claims
            del self._cache[key]

        if self._client is None:
            # One pooled client for the process so each check reuses the warm TLS connection.
            # Verify TLS against the OS trust store so Keycloak validates behind the corporate proxy CA.
            self._client = httpx.AsyncClient(timeout=self.timeout, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))

        start = time.perf_counter()
        try:
            resp = await self._client.post(
                self.introspect_url,
                data={"token": token, "token_type_hint": "access_token"},
                auth=(self.client_id, self.client_secret),
                headers={"Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            logger.error(f"[KeycloakAuth] introspection timed out after {self.timeout}s: {exc!r}",
                         extra={"status": "error", "error_details": f"timeout: {exc!r}"})
            raise ApiError(504, "AUTH_TIMEOUT", "Authentication server timed out")
        except httpx.HTTPError as exc:
            logger.error(f"[KeycloakAuth] introspection request failed ({type(exc).__name__}): {exc!r}",
                         extra={"status": "error", "error_details": f"{type(exc).__name__}: {exc!r}"})
            raise ApiError(503, "AUTH_UNAVAILABLE", "Authentication server is unavailable")
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            # 401/403 here means Keycloak rejected *our* client credentials, not the caller's token.
            logger.error(f"[KeycloakAuth] introspection returned HTTP {resp.status_code}: {resp.text[:200]}",
                         extra={"status": "error", "error_details": f"HTTP {resp.status_code}: {resp.text[:200]}"})
            raise ApiError(502, "AUTH_SERVER_ERROR", f"Authentication server returned HTTP {resp.status_code}")
        try:
            claims = resp.json()
        except ValueError as exc:
            logger.error(f"[KeycloakAuth] introspection returned a non-JSON body: {resp.text[:200]!r}",
                         extra={"status": "error", "error_details": str(exc)})
            raise ApiError(502, "AUTH_SERVER_ERROR", "Authentication server returned an invalid response")
        if not isinstance(claims, dict) or claims.get("active") is not True:
            logger.warning(f"[KeycloakAuth] rejected inactive/invalid token (sha256={key[:12]}…, {elapsed_ms:.1f}ms)",
                           extra={"status": "unauthorized"})
            raise ApiError(401, "UNAUTHORIZED", "Invalid or expired access token", _UNAUTHORIZED_HEADERS)

        logger.info(f"[KeycloakAuth] token validated for '{claims.get('preferred_username', claims.get('sub', '?'))}' "
                    f"in {elapsed_ms:.1f}ms", extra={"status": "success", "latency": f"{elapsed_ms:.1f}ms"})

        ttl = float(self.cache_ttl)
        if isinstance(claims.get("exp"), (int, float)):
            ttl = min(ttl, claims["exp"] - time.time())
        if ttl > 0:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                now = time.monotonic()
                self._cache = {k: v for k, v in self._cache.items() if v[1] > now}  # drop expired entries
                if len(self._cache) >= _MAX_CACHE_ENTRIES:
                    self._cache.clear()
            self._cache[key] = (claims, time.monotonic() + ttl)
        return claims

    def require_auth(self, func: F) -> F:
        """Decorator: reject the request unless it carries a valid Keycloak token.

        Sync on purpose — Python applies decorators at import time. The wrapper it
        returns (which runs on every request) is async.
        """
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"@require_auth only supports async endpoints, got {func.__qualname__}")

        # FastAPI reads the wrapper's signature to decide what to inject. Resolve string
        # annotations (``from __future__ import annotations``) to real types, and make sure
        # a Request parameter exists so the wrapper can read the Authorization header.
        sig = inspect.signature(func)
        try:
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception as exc:
            logger.warning(f"[require_auth] could not resolve type hints for {func.__qualname__}: {exc}")
            hints = {}
        params = [p.replace(annotation=hints.get(p.name, p.annotation)) for p in sig.parameters.values()]
        request_param = next((p.name for p in params if p.annotation is Request), None)
        inject = request_param is None
        if inject:
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
                    logger.error(f"[require_auth] {func.__qualname__}: no Request object — rejecting")
                    raise ApiError(500, "INTERNAL_ERROR", "Authentication could not access the request")
                await self.authenticate(request)
            return await func(*args, **kwargs)

        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=hints.get("return", sig.return_annotation),
        )
        return typing.cast(F, wrapper)


keycloak_auth = KeycloakAuth.from_env()
require_auth = keycloak_auth.require_auth
