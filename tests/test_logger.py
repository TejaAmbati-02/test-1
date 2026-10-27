from __future__ import annotations

import json
import logging

import pytest

from utils import logger as logger_module
from utils.logger import (
    JsonFormatter,
    LoggingService,
    clear_log_context,
    log_function,
    set_log_context,
)


def _record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord("app", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_includes_contract_fields():
    out = json.loads(JsonFormatter().format(_record(
        search_query="valve", language="en", channel="web", status="error", latency="5ms",
        input_data={"a": 1}, response_payload={"b": 2}, error_details="boom",
    )))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["search_query"] == "valve"
    assert out["channel"] == "web"
    assert out["status"] == "error"
    assert out["input_data"] == {"a": 1}
    assert out["response_payload"] == {"b": 2}
    assert out["error_details"] == "boom"
    assert out["logger_name"] == "[ASMO Search]"


def test_json_formatter_defaults():
    out = json.loads(JsonFormatter().format(_record()))
    assert out["status"] == "success"
    assert out["response_payload"] == {}
    assert out["search_query"] == ""


def test_json_formatter_stringifies_unserialisable_values():
    out = json.loads(JsonFormatter().format(_record(input_data={1, 2})))
    assert isinstance(out["input_data"], str)


def test_request_id_from_thread_context():
    set_log_context("req-123")
    try:
        assert json.loads(JsonFormatter().format(_record()))["request_id"] == "req-123"
        # An explicit request_id on the record wins over the context.
        assert json.loads(JsonFormatter().format(_record(request_id="explicit")))["request_id"] == "explicit"
    finally:
        clear_log_context()
    assert json.loads(JsonFormatter().format(_record()))["request_id"] == ""


@pytest.fixture
def isolated_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield root
    for h in root.handlers:
        if h not in handlers:
            h.close()
    root.handlers = handlers
    root.setLevel(level)


def test_setup_console_only(isolated_root_logger):
    svc = LoggingService()
    before = len(isolated_root_logger.handlers)
    svc.setup(level="DEBUG")
    assert isolated_root_logger.level == logging.DEBUG
    assert len(isolated_root_logger.handlers) == before + 1
    assert logging.getLogger("oci").level == logging.WARNING


def test_setup_is_idempotent(isolated_root_logger):
    svc = LoggingService()
    svc.setup()
    count = len(isolated_root_logger.handlers)
    svc.setup()
    assert len(isolated_root_logger.handlers) == count


def test_setup_unknown_level_falls_back_to_info(isolated_root_logger):
    LoggingService().setup(level="NOPE")
    assert isolated_root_logger.level == logging.INFO


def test_setup_writes_json_file(isolated_root_logger, tmp_path):
    svc = LoggingService()
    svc.log_dir = tmp_path / "logs"
    svc.setup(level="INFO", to_file=True)
    logging.getLogger("app").info("to file", extra={"status": "success"})
    for h in isolated_root_logger.handlers:
        h.flush()
    files = list(svc.log_dir.glob("asmo-*.log"))
    assert len(files) == 1
    lines = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert any(line["message"] == "to file" for line in lines)


def test_get_logger_returns_named_logger():
    assert LoggingService().get_logger("x.y").name == "x.y"


def test_log_function_sync_success_and_failure(caplog):
    @log_function
    def add(a, b):
        return a + b

    @log_function
    def fail():
        raise ValueError("bad")

    with caplog.at_level(logging.INFO, logger="app"):
        assert add(1, 2) == 3
        with pytest.raises(ValueError):
            fail()
    assert add.__name__ == "add"
    messages = [r.getMessage() for r in caplog.records]
    assert any("add completed" in m for m in messages)
    assert any("fail failed" in m for m in messages)


async def test_log_function_async_success_and_failure(caplog):
    @log_function
    async def double(x):
        return x * 2

    @log_function
    async def fail():
        raise RuntimeError("bad")

    with caplog.at_level(logging.INFO, logger="app"):
        assert await double(4) == 8
        with pytest.raises(RuntimeError):
            await fail()
    messages = [r.getMessage() for r in caplog.records]
    assert any("double completed" in m for m in messages)
    assert any("fail failed" in m for m in messages)


def test_module_singleton_exports():
    assert logger_module.logger.name == "app"
    assert logger_module.setup_logging == logger_module.logging_service.setup
