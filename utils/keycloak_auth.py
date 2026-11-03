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
import ssl
import typing
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx
import truststore
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
    try:
        value = os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")
        logger.debug(f"[_env_bool] {name}={value} (default={default})")
        return value
    except Exception as exc:
        logger.exception(
            f"[_env_bool] failed to read {name}: {exc}",
            extra={"function_name": "_env_bool", "status": "error", "error_details": str(exc)},
        )
        raise


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
        logger.debug("[KeycloakAuth.from_env] start — reading KEYCLOAK_* settings")
        try:
            enabled = _env_bool("KEYCLOAK_AUTH_ENABLED", True)
            required = ("KEYCLOAK_SERVER_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET")
            missing = [name for name in required if not os.environ.get(name, "").strip()]
            if enabled and missing:
                logger.error(
                    f"[KeycloakAuth.from_env] auth enabled but missing settings: {', '.join(missing)}",
                    extra={"function_name": "KeycloakAuth.from_env", "status": "error",
                           "error_details": f"missing: {', '.join(missing)}"},
                )
                raise RuntimeError(f"Keycloak auth is enabled but {', '.join(missing)} is not set in .env")
            auth = cls(
                server_url=os.environ.get("KEYCLOAK_SERVER_URL", "").strip(),
                realm=os.environ.get("KEYCLOAK_REALM", "").strip(),
                client_id=os.environ.get("KEYCLOAK_CLIENT_ID", "").strip(),
                client_secret=os.environ.get("KEYCLOAK_CLIENT_SECRET", "").strip(),
                timeout=float(os.environ.get("KEYCLOAK_TIMEOUT", "5")),
                cache_ttl=int(os.environ.get("KEYCLOAK_CACHE_TTL", "60")),
                enabled=enabled,
            )
            if not enabled:
                logger.warning("[KeycloakAuth.from_env] Keycloak auth is DISABLED (KEYCLOAK_AUTH_ENABLED=false)")
            logger.info(
                f"[KeycloakAuth.from_env] configured — enabled={auth.enabled} server={auth.server_url or '-'} "
                f"realm={auth.realm or '-'} client_id={auth.client_id or '-'} "
                f"timeout={auth.timeout}s cache_ttl={auth.cache_ttl}s"
            )
            return auth
        except RuntimeError:
            raise
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth.from_env] failed to build Keycloak auth from env: {exc}",
                extra={"function_name": "KeycloakAuth.from_env", "status": "error", "error_details": str(exc)},
            )
            raise

    # HTTP client

    async def _get_client(self) -> httpx.AsyncClient:
        # One pooled client for the process so each check reuses the warm TLS connection.
        try:
            if self._client is None:
                async with self._client_lock:
                    if self._client is None:
                        logger.debug(f"[KeycloakAuth._get_client] creating HTTP client (timeout={self.timeout}s)")
                        # Verify TLS against the OS trust store, not certifi's bundle, so
                        # Keycloak's certificate validates behind the corporate proxy CA.
                        self._client = httpx.AsyncClient(
                            timeout=self.timeout, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                        )
                        logger.info("[KeycloakAuth._get_client] HTTP client created")
            return self._client
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth._get_client] failed to create HTTP client: {exc}",
                extra={"function_name": "KeycloakAuth._get_client", "status": "error", "error_details": str(exc)},
            )
            raise

    async def close(self) -> None:
        try:
            if self._client is not None:
                logger.debug("[KeycloakAuth.close] closing HTTP client")
                await self._client.aclose()
                self._client = None
                logger.info("[KeycloakAuth.close] HTTP client closed")
            else:
                logger.debug("[KeycloakAuth.close] no HTTP client to close")
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth.close] failed to close HTTP client: {exc}",
                extra={"function_name": "KeycloakAuth.close", "status": "error", "error_details": str(exc)},
            )
            raise

    # Errors

    @staticmethod
    def _error(status: int, code: str, message: str) -> HTTPException:
        headers = {"WWW-Authenticate": 'Bearer realm="api"'} if status == 401 else None
        return HTTPException(status_code=status, detail={"code": code, "message": message}, headers=headers)

    # Token handling

    @classmethod
    def extract_token(cls, request: Request) -> str:
        try:
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            if not header:
                logger.warning(f"[KeycloakAuth.extract_token] missing Authorization header on {request.method} {request.url.path}")
                raise cls._error(401, "UNAUTHORIZED", "Missing Authorization header")
            if scheme.lower() != "bearer" or not token.strip():
                # Log only the scheme — never the credential itself.
                logger.warning(
                    f"[KeycloakAuth.extract_token] malformed Authorization header on {request.method} {request.url.path} "
                    f"(scheme='{scheme[:20]}', token_present={bool(token.strip())})"
                )
                raise cls._error(401, "UNAUTHORIZED", "Authorization header must be 'Bearer <access_token>'")
            token = token.strip()
            logger.debug(f"[KeycloakAuth.extract_token] bearer token extracted (length={len(token)})")
            return token
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth.extract_token] failed to read Authorization header: {exc}",
                extra={"function_name": "KeycloakAuth.extract_token", "status": "error", "error_details": str(exc)},
            )
            raise

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        try:
            entry = self._cache.get(key)
            if entry is None:
                logger.debug(f"[KeycloakAuth._cache_get] miss (key={key[:12]}…)")
                return None
            claims, expires_at = entry
            if time.monotonic() >= expires_at:
                self._cache.pop(key, None)
                logger.debug(f"[KeycloakAuth._cache_get] expired entry evicted (key={key[:12]}…)")
                return None
            logger.debug(f"[KeycloakAuth._cache_get] hit (key={key[:12]}…, ttl_left={expires_at - time.monotonic():.1f}s)")
            return claims
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth._cache_get] cache lookup failed: {exc}",
                extra={"function_name": "KeycloakAuth._cache_get", "status": "error", "error_details": str(exc)},
            )
            raise

    def _cache_put(self, key: str, claims: dict[str, Any]) -> None:
        try:
            if self.cache_ttl <= 0:
                logger.debug("[KeycloakAuth._cache_put] caching disabled (KEYCLOAK_CACHE_TTL<=0)")
                return
            ttl = float(self.cache_ttl)
            # Never cache past the token's own expiry.
            exp = claims.get("exp")
            if isinstance(exp, (int, float)):
                ttl = min(ttl, exp - time.time())
            if ttl <= 0:
                logger.debug(f"[KeycloakAuth._cache_put] token expires now/past — not cached (key={key[:12]}…)")
                return
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                before = len(self._cache)
                now = time.monotonic()
                self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
                if len(self._cache) >= _MAX_CACHE_ENTRIES:
                    self._cache.clear()
                logger.warning(
                    f"[KeycloakAuth._cache_put] cache full ({before} entries) — pruned to {len(self._cache)}"
                )
            self._cache[key] = (claims, time.monotonic() + ttl)
            logger.debug(f"[KeycloakAuth._cache_put] cached (key={key[:12]}…, ttl={ttl:.1f}s, size={len(self._cache)})")
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth._cache_put] cache store failed: {exc}",
                extra={"function_name": "KeycloakAuth._cache_put", "status": "error", "error_details": str(exc)},
            )
            raise

    async def introspect(self, token: str) -> dict[str, Any]:
        """Return the token's claims if Keycloak reports it active; raise HTTPException otherwise."""
        try:
            # Only the hash prefix is ever logged — never the raw token.
            key = hashlib.sha256(token.encode()).hexdigest()
            logger.debug(f"[KeycloakAuth.introspect] start (token_sha256={key[:12]}…, length={len(token)})")
            cached = self._cache_get(key)
            if cached is not None:
                logger.debug(
                    f"[KeycloakAuth.introspect] served from cache for "
                    f"'{cached.get('preferred_username', cached.get('sub', '?'))}'"
                )
                return cached

            client = await self._get_client()
            t0 = time.perf_counter()
            try:
                logger.debug(f"[KeycloakAuth.introspect] POST {self.introspect_url} (client_id={self.client_id})")
                resp = await client.post(
                    self.introspect_url,
                    data={"token": token, "token_type_hint": "access_token"},
                    auth=(self.client_id, self.client_secret),
                    headers={"Accept": "application/json"},
                )
            except httpx.TimeoutException as exc:
                logger.error(
                    f"[KeycloakAuth.introspect] Keycloak introspection timed out after {self.timeout}s "
                    f"({self.introspect_url}): {exc!r}",
                    extra={"function_name": "KeycloakAuth.introspect", "status": "error",
                           "error_details": f"timeout: {exc!r}", "latency": f"{self.timeout}s"},
                )
                raise self._error(504, "AUTH_TIMEOUT", "Authentication server timed out")
            except httpx.HTTPError as exc:
                logger.error(
                    f"[KeycloakAuth.introspect] Keycloak introspection request failed "
                    f"({type(exc).__name__}, {self.introspect_url}): {exc!r}",
                    extra={"function_name": "KeycloakAuth.introspect", "status": "error",
                           "error_details": f"{type(exc).__name__}: {exc!r}"},
                )
                raise self._error(503, "AUTH_UNAVAILABLE", "Authentication server is unavailable")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug(f"[KeycloakAuth.introspect] Keycloak answered HTTP {resp.status_code} in {elapsed_ms:.1f}ms")

            if resp.status_code != 200:
                # 401/403 here means Keycloak rejected *our* client credentials, not the caller's token.
                logger.error(
                    f"[KeycloakAuth.introspect] Keycloak introspection returned HTTP {resp.status_code} "
                    f"in {elapsed_ms:.1f}ms: {resp.text[:200]}",
                    extra={"function_name": "KeycloakAuth.introspect", "status": "error",
                           "latency": f"{elapsed_ms:.1f}ms",
                           "error_details": f"HTTP {resp.status_code}: {resp.text[:200]}"},
                )
                raise self._error(502, "AUTH_SERVER_ERROR", f"Authentication server returned HTTP {resp.status_code}")

            try:
                claims = resp.json()
            except ValueError as exc:
                logger.error(
                    f"[KeycloakAuth.introspect] Keycloak introspection returned a non-JSON body "
                    f"(content-type={resp.headers.get('content-type', '?')}, body={resp.text[:200]!r}): {exc}",
                    extra={"function_name": "KeycloakAuth.introspect", "status": "error", "error_details": str(exc)},
                )
                raise self._error(502, "AUTH_SERVER_ERROR", "Authentication server returned an invalid response")

            if not isinstance(claims, dict) or claims.get("active") is not True:
                logger.warning(
                    f"[KeycloakAuth.introspect] Rejected inactive/invalid token ({elapsed_ms:.1f}ms, "
                    f"token_sha256={key[:12]}…, active={claims.get('active') if isinstance(claims, dict) else 'n/a'})",
                    extra={"function_name": "KeycloakAuth.introspect", "status": "unauthorized",
                           "latency": f"{elapsed_ms:.1f}ms"},
                )
                raise self._error(401, "UNAUTHORIZED", "Invalid or expired access token")

            logger.info(
                f"[KeycloakAuth.introspect] Token validated for "
                f"'{claims.get('preferred_username', claims.get('sub', '?'))}' in {elapsed_ms:.1f}ms "
                f"(client={claims.get('azp', claims.get('client_id', '?'))}, exp={claims.get('exp', '?')})",
                extra={"function_name": "KeycloakAuth.introspect", "status": "success", "latency": f"{elapsed_ms:.1f}ms"},
            )
            self._cache_put(key, claims)
            return claims
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth.introspect] unexpected error during token introspection: {exc}",
                extra={"function_name": "KeycloakAuth.introspect", "status": "error", "error_details": str(exc)},
            )
            raise

    async def authenticate(self, request: Request) -> dict[str, Any]:
        """Validate the request's bearer token and store the claims on ``request.state.user``."""
        logger.debug(f"[KeycloakAuth.authenticate] start {request.method} {request.url.path}")
        try:
            claims = await self.introspect(self.extract_token(request))
            request.state.user = claims
            logger.debug(
                f"[KeycloakAuth.authenticate] authenticated "
                f"'{claims.get('preferred_username', claims.get('sub', '?'))}' for {request.method} {request.url.path}"
            )
            return claims
        except HTTPException as exc:
            logger.warning(
                f"[KeycloakAuth.authenticate] rejected {request.method} {request.url.path} — "
                f"HTTP {exc.status_code}: {exc.detail}",
                extra={"function_name": "KeycloakAuth.authenticate", "status": "error",
                       "error_details": str(exc.detail)},
            )
            raise
        except Exception as exc:
            logger.exception(
                f"[KeycloakAuth.authenticate] unexpected error for {request.method} {request.url.path}: {exc}",
                extra={"function_name": "KeycloakAuth.authenticate", "status": "error", "error_details": str(exc)},
            )
            raise

    # Decorator

    def require_auth(self, func: F) -> F:
        """Decorator that rejects the request unless it carries a valid Keycloak token."""
        if not inspect.iscoroutinefunction(func):
            logger.error(f"[require_auth] cannot decorate non-async endpoint {func.__qualname__}")
            raise TypeError(f"@require_auth only supports async endpoints, got {func.__qualname__}")

        sig = inspect.signature(func)
        # Resolve string annotations (``from __future__ import annotations``) in the
        # endpoint's own module so FastAPI sees real types on the wrapper's signature.
        try:
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception as exc:
            logger.warning(
                f"[require_auth] could not resolve type hints for {func.__qualname__} — using raw annotations: {exc}"
            )
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
        logger.debug(
            f"[require_auth] protecting {func.__qualname__} "
            f"(request_param={request_param}, injected={inject}, enabled={self.enabled})"
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = func.__qualname__
            request = kwargs.pop(_INJECTED_REQUEST_PARAM) if inject else kwargs.get(request_param)
            if request is None:
                request = next((a for a in args if isinstance(a, Request)), None)
            where = f"{request.method} {request.url.path}" if request is not None else "<no request>"
            try:
                if self.enabled:
                    if request is None:
                        logger.error(
                            f"[require_auth:{name}] could not access the Request object — rejecting",
                            extra={"function_name": name, "status": "error",
                                   "error_details": "Request not found in endpoint arguments"},
                        )
                        raise self._error(500, "INTERNAL_ERROR", "Authentication could not access the request")
                    logger.debug(f"[require_auth:{name}] authenticating {where}")
                    await self.authenticate(request)
                    logger.debug(f"[require_auth:{name}] auth OK for {where} — running endpoint")
                else:
                    logger.debug(f"[require_auth:{name}] auth disabled — skipping check for {where}")
            except HTTPException as exc:
                logger.warning(
                    f"[require_auth:{name}] auth failed for {where} — HTTP {exc.status_code}",
                    extra={"function_name": name, "status": "error", "error_details": str(exc.detail)},
                )
                raise
            except Exception as exc:
                logger.exception(
                    f"[require_auth:{name}] unexpected error during authentication for {where}: {exc}",
                    extra={"function_name": name, "status": "error", "error_details": str(exc)},
                )
                raise
            return await func(*args, **kwargs)

        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=params,
            return_annotation=hints.get("return", sig.return_annotation),
        )
        return typing.cast(F, wrapper)


keycloak_auth = KeycloakAuth.from_env()
require_auth = keycloak_auth.require_auth
