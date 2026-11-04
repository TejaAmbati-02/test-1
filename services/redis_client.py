"""One shared async Redis client for the whole application.

    await redis_client.connect()                  # once, at startup (app.py lifespan)
    await redis_client.client.set("key", "value") # anywhere — reuses the same connection pool
    await redis_client.close()                    # on shutdown

Redis is off when REDIS_ENABLED=false or REDIS_HOST is empty.
"""
from __future__ import annotations

import time

import redis.asyncio as redis

from config import Settings, settings as _settings
from utils.logger import logger


class RedisNotConnected(Exception):
    """A Redis call was made before connect() (or after close())."""


class RedisClientService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: redis.Redis | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._settings.redis_enabled and self._settings.redis_host)

    @property
    def endpoint(self) -> str:
        return f"{self._settings.redis_host}:{self._settings.redis_port}"

    @property
    def client(self) -> redis.Redis:
        """The shared client. Raises RedisNotConnected before connect() / after close()."""
        if self._client is None:
            raise RedisNotConnected("Redis client is not connected — call RedisClientService.connect() at startup")
        return self._client

    async def connect(self) -> None:
        """Create the shared client (connections open on first use). Does nothing if disabled or already connected."""
        if not self.enabled or self._client is not None:
            return
        cfg = self._settings
        logger.info(
            f"Redis — creating client for {self.endpoint} (db={cfg.redis_db}, ssl={cfg.redis_ssl}, "
            f"timeout={cfg.redis_timeout}s, password={'set' if cfg.redis_password else 'not set'})"
        )
        tls = {"ssl": True, "ssl_cert_reqs": cfg.redis_ssl_cert_reqs} if cfg.redis_ssl else {}
        self._client = redis.Redis(
            host=cfg.redis_host,
            port=cfg.redis_port,
            username=cfg.redis_username or None,
            password=cfg.redis_password or None,
            db=cfg.redis_db,
            socket_connect_timeout=cfg.redis_timeout,
            socket_timeout=cfg.redis_timeout,
            **tls,
        )

    async def ping(self) -> bool:
        """True when connected and Redis answers; never raises."""
        if self._client is None:
            return False
        start = time.perf_counter()
        try:
            ok = bool(await self._client.ping())
        except Exception as exc:
            logger.warning(f"Redis — unreachable at {self.endpoint}: {type(exc).__name__}: {exc}",
                           extra={"status": "error", "error_details": str(exc)})
            return False
        logger.info(f"Redis ping — {'OK' if ok else 'FAILED'} in {(time.perf_counter() - start) * 1000:.1f}ms")
        return ok

    async def close(self) -> None:
        """Close the shared client. Safe to call more than once."""
        if self._client is None:
            return
        try:
            await self._client.aclose()
            logger.info("Redis — connection closed")
        except Exception as exc:
            logger.exception(f"Redis — error while closing: {exc}", extra={"status": "error", "error_details": str(exc)})
        finally:
            self._client = None


redis_client = RedisClientService(_settings)
