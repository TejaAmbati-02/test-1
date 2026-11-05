"""Tests for services/redis_client.py (the shared client lifecycle)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from services import redis_client as rc
from services.redis_client import RedisClientService, RedisNotConnected


def _service(**overrides) -> RedisClientService:
    values = {"redis_host": "redis.test", "redis_enabled": True, "redis_ssl": False}
    values.update(overrides)
    return RedisClientService(settings.model_copy(update=values))


@pytest.fixture
def fake_redis(monkeypatch):
    """Replace redis.asyncio.Redis with a mock class; returns (cls_mock, instance)."""
    instance = MagicMock()
    instance.ping = AsyncMock(return_value=True)
    instance.aclose = AsyncMock()
    cls = MagicMock(return_value=instance)
    monkeypatch.setattr(rc.redis, "Redis", cls)
    return cls, instance


@pytest.mark.parametrize("host,enabled,expected", [("h", True, True), ("", True, False), ("h", False, False)])
def test_enabled(host, enabled, expected):
    assert _service(redis_host=host, redis_enabled=enabled).enabled is expected


def test_endpoint():
    assert _service(redis_host="h", redis_port=1).endpoint == "h:1"


def test_client_before_connect_raises():
    with pytest.raises(RedisNotConnected):
        _ = _service().client


@pytest.mark.parametrize("overrides", [{"redis_enabled": False}, {"redis_host": ""}])
async def test_connect_disabled_does_nothing(fake_redis, overrides):
    cls, _ = fake_redis
    service = _service(**overrides)
    await service.connect()
    cls.assert_not_called()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_connect_creates_client_without_tls(fake_redis):
    cls, instance = fake_redis
    service = _service(redis_port=1, redis_username="u", redis_password="p", redis_db=3, redis_timeout=4)
    await service.connect()
    cls.assert_called_once_with(
        host="redis.test", port=1, username="u", password="p", db=3,
        socket_connect_timeout=4, socket_timeout=4,
    )
    assert service.client is instance


async def test_connect_blank_credentials_become_none(fake_redis):
    cls, _ = fake_redis
    await _service(redis_username="", redis_password="").connect()
    kwargs = cls.call_args.kwargs
    assert kwargs["username"] is None and kwargs["password"] is None


async def test_connect_with_tls(fake_redis):
    cls, _ = fake_redis
    await _service(redis_ssl=True, redis_ssl_cert_reqs="none").connect()
    kwargs = cls.call_args.kwargs
    assert kwargs["ssl"] is True
    assert kwargs["ssl_cert_reqs"] == "none"


async def test_connect_is_idempotent(fake_redis):
    cls, _ = fake_redis
    service = _service()
    await service.connect()
    await service.connect()
    cls.assert_called_once()


async def test_connect_failure_propagates(monkeypatch):
    monkeypatch.setattr(rc.redis, "Redis", MagicMock(side_effect=RuntimeError("bad config")))
    service = _service()
    with pytest.raises(RuntimeError, match="bad config"):
        await service.connect()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_ping_not_connected_returns_false():
    assert await _service().ping() is False


@pytest.mark.parametrize("reply,expected", [(True, True), (False, False)])
async def test_ping_returns_reply(fake_redis, reply, expected):
    _, instance = fake_redis
    instance.ping.return_value = reply
    service = _service()
    await service.connect()
    assert await service.ping() is expected


async def test_ping_swallows_errors(fake_redis):
    _, instance = fake_redis
    instance.ping.side_effect = ConnectionError("down")
    service = _service()
    await service.connect()
    assert await service.ping() is False


async def test_close_releases_client(fake_redis):
    _, instance = fake_redis
    service = _service()
    await service.connect()
    await service.close()
    instance.aclose.assert_awaited_once()
    with pytest.raises(RedisNotConnected):
        _ = service.client


async def test_close_twice_is_safe(fake_redis):
    _, instance = fake_redis
    service = _service()
    await service.connect()
    await service.close()
    await service.close()
    instance.aclose.assert_awaited_once()


async def test_close_without_connect_is_safe():
    await _service().close()


async def test_close_error_is_swallowed_and_client_cleared(fake_redis):
    _, instance = fake_redis
    instance.aclose.side_effect = RuntimeError("boom")
    service = _service()
    await service.connect()
    await service.close()
    with pytest.raises(RedisNotConnected):
        _ = service.client


def test_shared_instance_built_from_settings():
    assert isinstance(rc.redis_client, RedisClientService)
    assert rc.redis_client.endpoint == f"{settings.redis_host}:{settings.redis_port}"
