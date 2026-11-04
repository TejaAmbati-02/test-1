from __future__ import annotations

import json
import logging
from logging.handlers import TimedRotatingFileHandler

import pytest

from utils import logger as logger_module
from utils.logger import JsonFormatter, logger, request_id_var, setup_logging


def _record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord("app", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def _format(**extra) -> dict:
    return json.loads(JsonFormatter().format(_record(**extra)))


def test_json_formatter_includes_contract_fields():
    out = _format(language="en", status="error", latency="5ms", input_data={"a": 1},
                  response_payload={"b": 2}, error_details="boom", function_name="f")
    assert out["message"] == "hello"
    assert out["level"] == "INFO"
    assert out["language"] == "en"
    assert out["status"] == "error"
    assert out["input_data"] == {"a": 1}
    assert out["response_payload"] == {"b": 2}
    assert out["error_details"] == "boom"
    assert out["function_name"] == "f"
    assert out["timestamp"].endswith("Z")


def test_json_formatter_defaults():
    out = _format()
    assert out["status"] == "Success"
    assert out["response_payload"] == {}
    assert out["input_data"] == ""


def test_json_formatter_stringifies_unserialisable_values():
    assert isinstance(_format(input_data={1, 2})["input_data"], str)


def test_json_formatter_survives_non_string_keys():
    out = _format(input_data={(1, 2): "tuple key"})
    assert "tuple key" in out["input_data"]


def test_request_id_from_context():
    token = request_id_var.set("req-123")
    try:
        assert _format()["request_id"] == "req-123"
        # An explicit request_id on the record wins over the context.
        assert _format(request_id="explicit")["request_id"] == "explicit"
    finally:
        request_id_var.reset(token)
    assert _format()["request_id"] == ""


@pytest.fixture
def restore_logger():
    """Undo setup_logging's changes to the shared loggers after the test."""
    app_logger = logging.getLogger("app")
    saved = [(lg, list(lg.handlers), lg.level) for lg in (logger, app_logger)]
    yield
    for h in logger.handlers:
        if h not in saved[0][1]:
            h.close()
    for lg, handlers, level in saved:
        lg.handlers = handlers
        lg.setLevel(level)


def test_setup_logging_sets_level_and_shares_handlers(restore_logger):
    setup_logging(level="DEBUG")
    assert logger.level == logging.DEBUG
    app_logger = logging.getLogger("app")
    assert app_logger.handlers == logger.handlers
    assert app_logger.level == logging.DEBUG
    assert logging.getLogger("oci").level == logging.WARNING


def test_setup_logging_unknown_level_falls_back_to_info(restore_logger):
    setup_logging(level="NOPE")
    assert logger.level == logging.INFO


def test_setup_logging_adds_one_file_handler(restore_logger, monkeypatch, tmp_path):
    monkeypatch.setattr(logger_module, "IS_WINDOWS", False)
    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path / "logs")
    setup_logging(to_file=True)
    setup_logging(to_file=True)  # idempotent
    file_handlers = [h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)]
    assert len(file_handlers) == 1

    logger.warning("to file")
    file_handlers[0].flush()
    lines = [json.loads(line) for line in next((tmp_path / "logs").glob("*.log")).read_text().splitlines()]
    assert any(line["message"] == "to file" for line in lines)
