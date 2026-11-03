"""Tests for utils.logger — Azure Monitor / OpenTelemetry are replaced by fakes, nothing is exported."""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import pytest

from utils import logger as logger_module
from utils.logger import JsonFormatter, clear_log_context, set_log_context

_THIRD_PARTY = ("urllib3", "httpx", "httpcore", "elastic_transport", "oci", "circuitbreaker")


def _record(msg: str = "hello", args=None, **extra) -> logging.LogRecord:
    record = logging.LogRecord("app", logging.INFO, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def _fmt(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(autouse=True)
def restore_loggers():
    """Put the shared loggers back exactly as they were; close any handler a test added."""
    saved = {}
    for name in ("mcit_logger", "app"):
        lg = logging.getLogger(name)
        saved[name] = (list(lg.handlers), lg.level, lg.propagate)
    third = {name: logging.getLogger(name).level for name in _THIRD_PARTY}
    yield
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        for h in lg.handlers:
            if h not in handlers:
                h.close()
        lg.handlers = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    for name, level in third.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Point the module's ``__file__`` into tmp_path so ``_file_handler`` writes to tmp_path/logs."""
    (tmp_path / "utils").mkdir()
    monkeypatch.setattr(logger_module, "__file__", str(tmp_path / "utils" / "logger.py"))
    return tmp_path


# JsonFormatter


def test_json_formatter_includes_contract_fields():
    out = _fmt(_record(
        event="search", user_id="u1", conversation_id="c1", platform="web", language="en",
        request_method="GET", request_path="/search", status_code=200, duration_ms=12.5,
        request_payload={"q": "valve"}, response_payload={"b": 2}, function_name="search",
        input_data={"a": 1}, error_details="boom", status="error", latency="5ms",
        latency_seconds=0.005, metric_type="latency", environment="test",
    ))
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["module"] == "test_logger"
    assert out["event"] == "search"
    assert out["user_id"] == "u1"
    assert out["conversation_id"] == "c1"
    assert out["platform"] == "web"
    assert out["language"] == "en"
    assert out["request_method"] == "GET"
    assert out["request_path"] == "/search"
    assert out["status_code"] == 200
    assert out["duration_ms"] == 12.5
    assert out["request_payload"] == {"q": "valve"}
    assert out["response_payload"] == {"b": 2}
    assert out["function_name"] == "search"
    assert out["input_data"] == {"a": 1}
    assert out["error_details"] == "boom"
    assert out["status"] == "error"
    assert out["latency"] == "5ms"
    assert out["latency_seconds"] == 0.005
    assert out["metric_type"] == "latency"
    assert out["environment"] == "test"
    assert out["logger_name"] == "[ASMOS BUYER ASSISTANT CHATBOT]"


def test_json_formatter_defaults(monkeypatch):
    monkeypatch.delenv("ENV_NAME", raising=False)
    out = _fmt(_record())
    assert out["status"] == "Success"
    assert out["request_payload"] == {}
    assert out["response_payload"] == {}
    assert out["input_data"] == ""
    assert out["error_details"] == ""
    assert out["event"] == ""
    assert out["function_name"] == ""
    assert out["environment"] is None


def test_json_formatter_environment_defaults_to_env_name(monkeypatch):
    monkeypatch.setenv("ENV_NAME", "staging")
    assert _fmt(_record())["environment"] == "staging"
    assert _fmt(_record(environment="prod"))["environment"] == "prod"


def test_json_formatter_timestamp_is_utc():
    record = _record()
    record.created = 0.0
    assert _fmt(record)["timestamp"] == "1970-01-01T00:00:00Z"


def test_json_formatter_applies_message_args():
    assert _fmt(_record("found %d items for %s", (3, "valve")))["message"] == "found 3 items for valve"


def test_json_formatter_keeps_unicode():
    raw = JsonFormatter().format(_record("válvula"))
    assert "válvula" in raw


@pytest.mark.parametrize("field", ["input_data", "request_payload", "response_payload", "error_details"])
def test_json_formatter_stringifies_unserialisable_values(field):
    out = _fmt(_record(**{field: {1, 2}}))
    assert isinstance(out[field], str)


def test_request_id_from_context():
    set_log_context("req-123")
    try:
        assert _fmt(_record())["request_id"] == "req-123"
        # An explicit request_id on the record wins over the context.
        assert _fmt(_record(request_id="explicit"))["request_id"] == "explicit"
    finally:
        clear_log_context()
    assert _fmt(_record())["request_id"] == ""


# Handler / logger construction


def test_configure_logger_replaces_handlers_and_reads_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    first, second = _ListHandler(), _ListHandler()
    lg = logger_module._configure_logger([first])
    lg = logger_module._configure_logger([second])
    assert lg.name == "mcit_logger"
    assert lg.handlers == [second]
    assert lg.level == logging.DEBUG
    assert lg.propagate is False


def test_configure_logger_unknown_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOPE")
    assert logger_module._configure_logger([]).level == logging.INFO


def test_configure_logger_default_level_is_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert logger_module._configure_logger([]).level == logging.INFO


def test_console_handler_writes_json_to_stdout():
    handler = logger_module._console_handler()
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stdout
    assert isinstance(handler.formatter, JsonFormatter)


def test_file_handler_writes_json_to_repo_logs(tmp_repo):
    handler = logger_module._file_handler()
    try:
        assert isinstance(handler, TimedRotatingFileHandler)
        assert isinstance(handler.formatter, JsonFormatter)
        assert handler.when == "MIDNIGHT"
        assert handler.backupCount == 7
        assert handler.encoding == "utf-8"
        files = list((tmp_repo / "logs").glob("buyer-assistant-logs-*.log"))
        assert len(files) == 1

        handler.emit(_record("to file", status="success"))
        handler.flush()
        lines = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
        assert any(line["message"] == "to file" for line in lines)
    finally:
        handler.close()


def test_setup_local_logger_windows_uses_file_only(monkeypatch):
    file_handler = _ListHandler()
    monkeypatch.setattr(logger_module, "IS_WINDOWS", True)
    monkeypatch.setattr(logger_module, "_file_handler", lambda: file_handler)
    assert logger_module.setup_local_logger().handlers == [file_handler]


def test_setup_local_logger_windows_falls_back_to_console_on_oserror(monkeypatch):
    def _boom():
        raise OSError("read-only")

    monkeypatch.setattr(logger_module, "IS_WINDOWS", True)
    monkeypatch.setattr(logger_module, "_file_handler", _boom)
    handlers = logger_module.setup_local_logger().handlers
    assert len(handlers) == 1
    assert type(handlers[0]) is logging.StreamHandler


def test_setup_local_logger_other_os_uses_console_only(monkeypatch):
    monkeypatch.setattr(logger_module, "IS_WINDOWS", False)
    monkeypatch.setattr(logger_module, "_file_handler", lambda: pytest.fail("file handler must not be built"))
    handlers = logger_module.setup_local_logger().handlers
    assert len(handlers) == 1
    assert type(handlers[0]) is logging.StreamHandler


# Azure Monitor (App Insights)


class _FakeProvider:
    def __init__(self) -> None:
        self.processors: list = []

    def add_log_record_processor(self, processor) -> None:
        self.processors.append(processor)


class _FakeOtelHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - never emits in tests
        pass


@pytest.fixture
def fake_otel(monkeypatch):
    """Replace every OpenTelemetry / Azure Monitor object the logger touches."""
    state: dict = {"exporters": [], "providers": [], "set_provider": []}

    def _exporter(**kwargs):
        state["exporters"].append(kwargs)
        return ("exporter", kwargs)

    def _provider():
        provider = _FakeProvider()
        state["providers"].append(provider)
        return provider

    monkeypatch.setattr(logger_module, "AzureMonitorLogExporter", _exporter)
    monkeypatch.setattr(logger_module, "LoggerProvider", _provider)
    monkeypatch.setattr(logger_module, "set_logger_provider", state["set_provider"].append)
    monkeypatch.setattr(logger_module, "BatchLogRecordProcessor", lambda exporter: ("processor", exporter))
    monkeypatch.setattr(logger_module, "LoggingHandler", _FakeOtelHandler)
    return state


@pytest.mark.parametrize("value", [None, "", "   "])
def test_app_insights_without_connection_string_uses_local_logger(monkeypatch, fake_otel, value):
    if value is None:
        monkeypatch.delenv("APP_INSIGHTS_CONNECTION_STRING", raising=False)
    else:
        monkeypatch.setenv("APP_INSIGHTS_CONNECTION_STRING", value)
    local = logging.getLogger("local-sentinel")
    monkeypatch.setattr(logger_module, "setup_local_logger", lambda: local)
    assert logger_module.setup_app_insights_logger() is local
    assert fake_otel["exporters"] == []
    assert fake_otel["set_provider"] == []


def test_app_insights_with_connection_string_adds_azure_handler(monkeypatch, fake_otel):
    monkeypatch.setenv("APP_INSIGHTS_CONNECTION_STRING", "  InstrumentationKey=abc  ")
    lg = logger_module.setup_app_insights_logger()

    assert fake_otel["exporters"] == [{"connection_string": "InstrumentationKey=abc"}]
    provider = fake_otel["providers"][0]
    assert fake_otel["set_provider"] == [provider]
    assert provider.processors == [("processor", ("exporter", {"connection_string": "InstrumentationKey=abc"}))]

    assert lg.name == "mcit_logger"
    console, azure = lg.handlers
    assert type(console) is logging.StreamHandler
    assert isinstance(azure, _FakeOtelHandler)
    assert isinstance(azure.formatter, JsonFormatter)


def test_app_insights_setup_failure_falls_back_to_local_logger(monkeypatch, fake_otel):
    def _broken(**_kwargs):
        raise ValueError("bad connection string")

    monkeypatch.setenv("APP_INSIGHTS_CONNECTION_STRING", "garbage")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setattr(logger_module, "AzureMonitorLogExporter", _broken)
    capture = _ListHandler()
    monkeypatch.setattr(logger_module, "setup_local_logger", lambda: logger_module._configure_logger([capture]))

    lg = logger_module.setup_app_insights_logger()
    assert lg.handlers == [capture]
    (record,) = capture.records
    assert record.getMessage() == "app_insights_setup_failed"
    assert record.event == "app_insights_setup_failed"
    assert record.error_details == "bad connection string"
    assert record.status == "failed"


# setup_logging


def test_setup_logging_sets_level_and_mirrors_app_logger(monkeypatch):
    monkeypatch.setattr(logger_module, "IS_WINDOWS", False)
    handler = _ListHandler()
    logger_module._configure_logger([handler])

    logger_module.setup_logging(level="debug")

    assert logger_module.logger.level == logging.DEBUG
    app_logger = logging.getLogger("app")
    assert app_logger.handlers == [handler]
    assert app_logger.handlers is not logger_module.logger.handlers
    assert app_logger.level == logging.DEBUG
    assert app_logger.propagate is False
    for name in _THIRD_PARTY:
        assert logging.getLogger(name).level == logging.WARNING


def test_setup_logging_unknown_level_falls_back_to_info():
    logger_module.setup_logging(level="NOPE")
    assert logger_module.logger.level == logging.INFO


def test_setup_logging_to_file_adds_file_handler_once(monkeypatch, tmp_repo):
    monkeypatch.setattr(logger_module, "IS_WINDOWS", False)
    logger_module._configure_logger([_ListHandler()])

    logger_module.setup_logging(level="INFO", to_file=True)
    logger_module.setup_logging(level="INFO", to_file=True)

    file_handlers = [h for h in logger_module.logger.handlers if isinstance(h, TimedRotatingFileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0] in logging.getLogger("app").handlers

    logging.getLogger("app").info("to file", extra={"status": "success"})
    file_handlers[0].flush()
    (log_file,) = (tmp_repo / "logs").glob("buyer-assistant-logs-*.log")
    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert any(line["message"] == "to file" for line in lines)


def test_setup_logging_to_file_ignored_on_windows(monkeypatch):
    monkeypatch.setattr(logger_module, "IS_WINDOWS", True)
    monkeypatch.setattr(logger_module, "_file_handler", lambda: pytest.fail("file handler must not be built"))
    logger_module._configure_logger([_ListHandler()])
    logger_module.setup_logging(to_file=True)
    assert not any(isinstance(h, TimedRotatingFileHandler) for h in logger_module.logger.handlers)


def test_setup_logging_file_error_is_logged_not_raised(monkeypatch):
    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(logger_module, "IS_WINDOWS", False)
    monkeypatch.setattr(logger_module, "_file_handler", _boom)
    capture = _ListHandler()
    logger_module._configure_logger([capture])

    logger_module.setup_logging(level="INFO", to_file=True)

    assert logger_module.logger.handlers == [capture]
    (record,) = [r for r in capture.records if r.getMessage() == "log_file_setup_failed"]
    assert record.error_details == "disk full"
    assert record.status == "failed"


# Module import


def test_module_exports_and_otel_limits():
    assert logger_module.logger is logging.getLogger("mcit_logger")
    assert logger_module.logger.propagate is False
    assert os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] == "4096"
    assert os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] == "50"
