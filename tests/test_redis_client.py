"""Tests for services/redis_client.py (config loading and the shared client lifecycle)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from services import redis_client as rc
from services.redis_client import RedisClientService, RedisConfig, RedisNotConnected, _env_bool

_REDIS_VARS = ("HOST", "PORT", "USERNAME", "PASSWORD", "DB", "SSL", "SSL_CERT_REQS", "TIMEOUT", "ENABLED")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _REDIS_VARS:
        monkeypatch.delenv(f"REDIS_{key}", raising=False)
    return monkeypatch


@pytest.fixture
def fake_redis(monkeypatch):
    """Replace redis.asyncio.Redis with a mock class; returns (cls_mock, instance)."""
    instance = MagicMock()
    instance.ping = AsyncMock(return_value=True)
    instance.aclose = AsyncMock()
    cls = MagicMock(return_value=instance)
    monkeypatch.setattr(rc.redis, "Redis", cls)
    return cls, instance


# --- _env_bool -------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_env_bool_truthy(monkeypatch, value):
    monkeypatch.setenv("X_FLAG", value)
    assert _env_bool("X_FLAG", False) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_env_bool_falsy(monkeypatch, value):
    monkeypatch.setenv("X_FLAG", value)
    assert _env_bool("X_FLAG", True) is False


def test_env_bool_default_when_unset(monkeypatch):
    monkeypatch.delenv("X_FLAG", raising=False)
    assert _env_bool("X_FLAG", True) is True
    assert _env_bool("X_FLAG", False) is False


# --- RedisConfig -----------------------------------------------------------

def test_safe_summary_hides_password():
    summary = RedisConfig(host="h", password="s3cret", extra={"b": 1, "a": 2}).safe_summary()
    assert "s3cret" not in str(summary)
    assert summary["password_set"] is True
    assert summary["extra_options"] == ["a", "b"]


def test_safe_summary_password_not_set():
    assert RedisConfig().safe_summary()["password_set"] is False


def test_from_env_defaults(clean_env):
    cfg = RedisConfig.from_env()
    assert cfg.host == ""
    assert cfg.port == 6379
    assert cfg.username is None and cfg.password is None
    assert cfg.db == 0
    assert cfg.ssl is False
    assert cfg.ssl_cert_reqs == "required"
    assert cfg.timeout == 10.0
    assert cfg.enabled is True
    assert cfg.missing_env == ["REDIS_HOST", "REDIS_PORT", "REDIS_SSL"]


def test_from_env_reads_all_values(clean_env):
    values = {
        "HOST": "redis.example", "PORT": "6380", "USERNAME": "u", "PASSWORD": "p", "DB": "2",
        "SSL": "true", "SSL_CERT_REQS": "none", "TIMEOUT": "2.5", "ENABLED": "false",
    }
    for key, value in values.items():
        clean_env.setenv(f"REDIS_{key}", value)
    cfg = RedisConfig.from_env()
    assert (cfg.host, cfg.port, cfg.username, cfg.password, cfg.db) == ("redis.example", 6380, "u", "p", 2)
    assert (cfg.ssl, cfg.ssl_cert_reqs, cfg.timeout, cfg.enabled) == (True, "none", 2.5, False)
    assert cfg.missing_env == []


def test_from_env_custom_prefix(clean_env):
    clean_env.setenv("CACHE_HOST", "cache.example")
    clean_env.setenv("CACHE_PORT", "1234")
    cfg = RedisConfig.from_env(prefix="CACHE_")
    assert (cfg.host, cfg.port) == ("cache.example", 1234)
    assert cfg.missing_env == ["CACHE_SSL"]


@pytest.mark.parametrize("key,value", [("PORT", "abc"), ("DB", "x"), ("TIMEOUT", "soon")])
def test_from_env_invalid_number_raises(clean_env, key, value):
    clean_env.setenv(f"REDIS_{key}", value)
    with pytest.raises(ValueError):
        RedisConfig.from_env()


# --- RedisClientService ----------------------------------------------------

@pytest.mark.parametrize("host,enabled,expected", [("h", True, True), ("", True, False), ("h", False, False)])
def test_enabled(host, enabled, expected):
    assert RedisClientService(RedisConfig(host=host, enabled=enabled)).enabled is expected


def test_endpoint():
    assert RedisClientService(RedisConfig(host="h", port=1)).endpoint == "h:1"


def test_client_before_connect_raises():
    with pytest.raises(RedisNotConnected):
        _ = RedisClientService(RedisConfig(host="h")).client


@pytest.mark.parametrize("cfg", [RedisConfig(host="h", enabled=False), RedisConfig(host="")])
async def test_connect_disabled_does_nothing(fake_redis, cfg):
    cls, _ = fake_redis
    service = RedisClientService(cfg)
    await service.connect()
    cls.assert_not_called()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_connect_creates_client_without_tls(fake_redis):
    cls, instance = fake_redis
    service = RedisClientService(RedisConfig(host="h", port=1, username="u", password="p", db=3, timeout=4,
                                             extra={"decode_responses": True}))
    await service.connect()
    cls.assert_called_once_with(
        host="h", port=1, username="u", password="p", db=3,
        socket_connect_timeout=4, socket_timeout=4, decode_responses=True,
    )
    assert service.client is instance


async def test_connect_with_tls(fake_redis):
    cls, _ = fake_redis
    await RedisClientService(RedisConfig(host="h", ssl=True, ssl_cert_reqs="none")).connect()
    kwargs = cls.call_args.kwargs
    assert kwargs["ssl"] is True
    assert kwargs["ssl_cert_reqs"] == "none"


async def test_connect_is_idempotent(fake_redis):
    cls, _ = fake_redis
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    await service.connect()
    cls.assert_called_once()


async def test_connect_warns_about_missing_env(fake_redis, monkeypatch):
    warn = MagicMock()
    monkeypatch.setattr(rc.logger, "warning", warn)
    await RedisClientService(RedisConfig(host="h", missing_env=["REDIS_PORT"])).connect()
    warn.assert_called_once()
    assert "REDIS_PORT" in warn.call_args.args[0]


async def test_connect_failure_propagates(monkeypatch):
    monkeypatch.setattr(rc.redis, "Redis", MagicMock(side_effect=RuntimeError("bad config")))
    service = RedisClientService(RedisConfig(host="h"))
    with pytest.raises(RuntimeError, match="bad config"):
        await service.connect()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_ping_not_connected_returns_false():
    assert await RedisClientService(RedisConfig(host="h")).ping() is False


@pytest.mark.parametrize("reply,expected", [(True, True), (False, False)])
async def test_ping_returns_reply(fake_redis, reply, expected):
    _, instance = fake_redis
    instance.ping.return_value = reply
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    assert await service.ping() is expected


async def test_ping_swallows_errors(fake_redis):
    _, instance = fake_redis
    instance.ping.side_effect = ConnectionError("down")
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    assert await service.ping() is False


async def test_close_releases_client(fake_redis):
    _, instance = fake_redis
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    await service.close()
    instance.aclose.assert_awaited_once()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_close_twice_is_safe(fake_redis):
    _, instance = fake_redis
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    await service.close()
    await service.close()
    instance.aclose.assert_awaited_once()


async def test_close_without_connect_is_safe():
    await RedisClientService(RedisConfig(host="h")).close()


async def test_close_error_is_swallowed_and_client_cleared(fake_redis):
    _, instance = fake_redis
    instance.aclose.side_effect = RuntimeError("boom")
    service = RedisClientService(RedisConfig(host="h"))
    await service.connect()
    await service.close()
    with pytest.raises(RedisNotConnected):
        _ = service.client


def test_shared_instance_built_from_settings():
    from config import settings

    assert isinstance(rc.redis_client, RedisClientService)
    assert rc.redis_client.endpoint == f"{settings.redis_host}:{settings.redis_port}"
