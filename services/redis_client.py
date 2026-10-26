"""Reusable async Redis client: one shared connection for the whole application.

Depends on ``redis`` (redis-py >= 5), the project logger (``utils.logger``) and
the project settings (``config.settings``). To reuse it elsewhere, copy the file
and point those two imports at the other project's logger and settings (or build
the config with ``RedisConfig.from_env()``).

The shared object ``redis_client`` is created at the bottom of this file from the
application settings. Import it wherever Redis is needed::

    from services.redis_client import redis_client

    @asynccontextmanager
    async def lifespan(app):
        await redis_client.connect()          # create the connection once, at startup
        yield
        await redis_client.close()            # release it on shutdown

    # anywhere else — every caller reuses the same client and connection pool:
    await redis_client.client.set("key", "value")
    ok = await redis_client.ping()
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis

from config import settings
from utils.logger import logger


class RedisNotConnected(Exception):
    """A Redis call was made before connect() (or after close())."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def _ms(start: float) -> str:
    return f"{(time.perf_counter() - start) * 1000:.1f}ms"


@dataclass
class RedisConfig:
    """Connection settings. ``enabled=False`` (or an empty host) turns Redis off."""

    host: str = ""
    port: int = 6379
    username: str | None = None
    password: str | None = None
    db: int = 0
    ssl: bool = False
    ssl_cert_reqs: str = "required"   # "none" | "optional" | "required"
    timeout: float = 10.0             # connect + socket timeout, seconds
    enabled: bool = True
    # Any other redis.asyncio.Redis keyword arguments (e.g. decode_responses=True).
    extra: dict[str, Any] = field(default_factory=dict)
    # Set by from_env(): variables that were absent, so their defaults apply.
    missing_env: list[str] = field(default_factory=list, repr=False)

    def safe_summary(self) -> dict[str, Any]:
        """The settings for logging — never includes the password itself."""
        return {
            "host": self.host, "port": self.port, "db": self.db, "ssl": self.ssl,
            "ssl_cert_reqs": self.ssl_cert_reqs, "timeout_s": self.timeout, "enabled": self.enabled,
            "username": self.username, "password_set": bool(self.password), "extra_options": sorted(self.extra),
        }

    @classmethod
    def from_env(cls, prefix: str = "REDIS_") -> "RedisConfig":
        """Read REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD, REDIS_DB,
        REDIS_SSL, REDIS_SSL_CERT_REQS, REDIS_TIMEOUT and REDIS_ENABLED."""
        env = lambda key, default="": os.environ.get(prefix + key, default)
        missing = [prefix + k for k in ("HOST", "PORT", "SSL") if (prefix + k) not in os.environ]
        try:
            config = cls(
                host=env("HOST"),
                port=int(env("PORT", "6379")),
                username=env("USERNAME") or None,
                password=env("PASSWORD") or None,
                db=int(env("DB", "0")),
                ssl=_env_bool(prefix + "SSL", False),
                ssl_cert_reqs=env("SSL_CERT_REQS", "required"),
                timeout=float(env("TIMEOUT", "10")),
                enabled=_env_bool(prefix + "ENABLED", True),
                missing_env=missing,
            )
        except ValueError as exc:
            logger.error(
                f"RedisConfig.from_env — invalid {prefix}* value: {exc}",
                extra={"function_name": "RedisConfig.from_env", "status": "error", "error_details": str(exc)},
            )
            raise
        logger.debug(
            "RedisConfig.from_env — settings loaded",
            extra={"function_name": "RedisConfig.from_env", "status": "success", "input_data": config.safe_summary()},
        )
        return config


class RedisClientService:
    """Owns one shared ``redis.asyncio.Redis`` client.

    ``connect()`` creates it once (call it at application startup); every caller
    then reuses it — and its connection pool — through :attr:`client`.
    ``close()`` releases it on shutdown.
    """

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._client: redis.Redis | None = None
        logger.debug(
            "RedisClientService — initialised (not connected yet)",
            extra={"function_name": "RedisClientService.__init__", "input_data": config.safe_summary()},
        )

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._config.host)

    @property
    def endpoint(self) -> str:
        return f"{self._config.host}:{self._config.port}"

    @property
    def client(self) -> redis.Redis:
        """The shared client. Raises RedisNotConnected before connect() / after close()."""
        if self._client is None:
            logger.error(
                "RedisClientService.client — used before connect() (or after close())",
                extra={"function_name": "RedisClientService.client", "status": "error",
                       "input_data": {"enabled": self.enabled, "endpoint": self.endpoint}},
            )
            raise RedisNotConnected("Redis client is not connected — call RedisClientService.connect() at startup")
        return self._client

    async def connect(self) -> None:
        """Create the shared client. Idempotent; does nothing when Redis is disabled."""
        fn = "RedisClientService.connect"
        if not self.enabled:
            reason = "REDIS_ENABLED=false" if not self._config.enabled else "REDIS_HOST is empty"
            logger.info(
                f"RedisClientService.connect — Redis disabled ({reason}); not connecting",
                extra={"function_name": fn, "status": "skipped", "input_data": {"reason": reason}},
            )
            return
        if self._client is not None:
            logger.debug(
                "RedisClientService.connect — already connected; reusing the existing client",
                extra={"function_name": fn, "input_data": {"endpoint": self.endpoint}},
            )
            return
        cfg = self._config
        logger.info(
            f"RedisClientService.connect — creating client for {self.endpoint} "
            f"(db={cfg.db}, ssl={cfg.ssl}, ssl_cert_reqs={cfg.ssl_cert_reqs}, timeout={cfg.timeout}s, "
            f"username={cfg.username or '-'}, password={'set' if cfg.password else 'not set'})",
            extra={"function_name": fn, "input_data": cfg.safe_summary()},
        )
        if cfg.missing_env:
            logger.warning(
                f"RedisClientService.connect — {', '.join(cfg.missing_env)} not set; using defaults for them",
                extra={"function_name": fn, "status": "warning", "input_data": {"missing": cfg.missing_env}},
            )
        start = time.perf_counter()
        tls = {"ssl": True, "ssl_cert_reqs": cfg.ssl_cert_reqs} if cfg.ssl else {}
        try:
            self._client = redis.Redis(
                host=cfg.host,
                port=cfg.port,
                username=cfg.username,
                password=cfg.password,
                db=cfg.db,
                socket_connect_timeout=cfg.timeout,
                socket_timeout=cfg.timeout,
                **tls,
                **cfg.extra,
            )
        except Exception as exc:
            logger.error(
                f"RedisClientService.connect — failed to create client for {self.endpoint}: {exc}",
                extra={"function_name": fn, "status": "error", "error_details": str(exc), "latency": _ms(start)},
            )
            raise
        logger.info(
            f"RedisClientService.connect — client created for {self.endpoint} (connections open on first use)",
            extra={"function_name": fn, "status": "success", "latency": _ms(start)},
        )

    async def ping(self) -> bool:
        """True when connected and Redis answers; never raises."""
        fn = "RedisClientService.ping"
        if self._client is None:
            logger.warning(
                "RedisClientService.ping — not connected; returning False",
                extra={"function_name": fn, "status": "error", "input_data": {"enabled": self.enabled}},
            )
            return False
        start = time.perf_counter()
        try:
            ok = bool(await self._client.ping())
        except Exception as exc:
            logger.warning(
                f"RedisClientService.ping — Redis unreachable at {self.endpoint}: {exc}",
                extra={"function_name": fn, "status": "error", "error_details": f"{type(exc).__name__}: {exc}",
                       "latency": _ms(start), "input_data": {"endpoint": self.endpoint}},
            )
            return False
        logger.info(
            f"RedisClientService.ping — {'OK' if ok else 'FAILED'} ({self.endpoint})",
            extra={"function_name": fn, "status": "success" if ok else "error", "latency": _ms(start),
                   "response_payload": {"reachable": ok}},
        )
        return ok

    async def close(self) -> None:
        """Close the shared client. Safe to call more than once."""
        fn = "RedisClientService.close"
        if self._client is None:
            logger.debug("RedisClientService.close — already closed", extra={"function_name": fn})
            return
        logger.info(f"RedisClientService.close — closing connection to {self.endpoint}", extra={"function_name": fn})
        start = time.perf_counter()
        try:
            await self._client.aclose()
            logger.info(
                "RedisClientService.close — connection closed",
                extra={"function_name": fn, "status": "success", "latency": _ms(start)},
            )
        except Exception as exc:
            logger.error(
                f"RedisClientService.close — error while closing: {exc}",
                extra={"function_name": fn, "status": "error", "error_details": str(exc), "latency": _ms(start)},
            )
        finally:
            self._client = None


# The shared instance 
# Created once from the application settings; import it wherever Redis is needed.
redis_client = RedisClientService(RedisConfig(
    host=settings.redis_host,
    port=settings.redis_port,
    username=settings.redis_username or None,
    password=settings.redis_password or None,
    db=settings.redis_db,
    ssl=settings.redis_ssl,
    ssl_cert_reqs=settings.redis_ssl_cert_reqs,
    timeout=settings.redis_timeout,
    enabled=settings.redis_enabled,
))
