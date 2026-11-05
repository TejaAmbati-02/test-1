from __future__ import annotations

import json
import logging

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
    others = [logging.getLogger(n) for n in ("app", "", "uvicorn", "uvicorn.error", "uvicorn.access")]
    saved = [(lg, list(lg.handlers), lg.level) for lg in (logger, *others)]
    yield
    for h in logger.handlers:
        if h not in saved[0][1]:
            h.close()
    for lg, handlers, level in saved:
        lg.handlers = handlers
        lg.setLevel(level)
    logging.captureWarnings(False)


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
    # On Windows the module attaches its file handler at import; start without it
    # (restore_logger puts it back) so setup_logging adds one under tmp_path.
    logger.handlers = [h for h in logger.handlers if h.get_name() != logger_module.FILE_HANDLER_NAME]
    setup_logging(to_file=True)
    setup_logging(to_file=True)  # idempotent
    file_handlers = [h for h in logger.handlers if h.get_name() == logger_module.FILE_HANDLER_NAME]
    assert len(file_handlers) == 1

    logger.warning("to file")
    file_handlers[0].flush()
    lines = [json.loads(line) for line in next((tmp_path / "logs").glob("*.log")).read_text().splitlines()]
    assert any(line["message"] == "to file" for line in lines)


def test_setup_logging_routes_library_logs_but_not_uvicorn(restore_logger):
    uvicorn_before = {n: list(logging.getLogger(n).handlers) for n in ("uvicorn", "uvicorn.error", "uvicorn.access")}
    setup_logging()
    assert logging.getLogger().handlers == logger.handlers  # library warnings -> the app's console/file
    for name, handlers in uvicorn_before.items():
        assert logging.getLogger(name).handlers == handlers  # server messages stay in cmd only


def test_json_formatter_keeps_any_extra_and_traceback():
    try:
        raise ValueError("bad")
    except ValueError:
        import sys
        record = _record(search_query="pump", result={"rank": 1, "score": 0.5})
        record.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(record))
    assert out["search_query"] == "pump"
    assert out["result"] == {"rank": 1, "score": 0.5}
    assert "ValueError: bad" in out["traceback"]
    assert out["source"].startswith("app:")


def test_new_log_file_per_start_named_with_seconds(monkeypatch, tmp_path):
    import re

    monkeypatch.setattr(logger_module, "LOGS_DIR", tmp_path)
    handler = logger_module._file_handler()
    try:
        name = handler.baseFilename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        assert re.fullmatch(r"asmo-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.log", name)
        assert handler.get_name() == logger_module.FILE_HANDLER_NAME
    finally:
        handler.close()
