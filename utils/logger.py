"""The application logger: one JSON object per line.

Where logs go:
  * Windows                              -> log file only (console if the file can't be opened)
  * APP_INSIGHTS_CONNECTION_STRING set   -> console + Azure Application Insights
  * otherwise                            -> console (LOG_TO_FILE=true also writes the log file)

Note: the logging module calls ``JsonFormatter.format`` synchronously and the
handlers are built when this module is imported, so nothing here is async.
"""
import contextvars
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

# OpenTelemetry limits
os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "4096"
os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "50"

IS_WINDOWS = platform.system() == "Windows"
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"  # <repo>/logs, whatever the working directory

# Attributes every LogRecord has. Anything else on a record came from ``extra={...}`` and is logged too.
_STANDARD_RECORD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}

# The current HTTP request's id. The request middleware sets it, so it appears on every log line of that request.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    converter = time.gmtime  # the timestamp ends in "Z" (UTC), so it must be UTC

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "message": record.getMessage(),
            "module": record.module,
            "event": getattr(record, "event", ""),
            "user_id": getattr(record, "user_id", ""),
            "conversation_id": getattr(record, "conversation_id", ""),
            "platform": getattr(record, "platform", ""),
            "language": getattr(record, "language", ""),
            "request_method": getattr(record, "request_method", ""),
            "request_path": getattr(record, "request_path", ""),
            "status_code": getattr(record, "status_code", ""),
            "duration_ms": getattr(record, "duration_ms", ""),
            "request_payload": getattr(record, "request_payload", {}),
            "response_payload": getattr(record, "response_payload", {}),
            "function_name": getattr(record, "function_name", ""),
            "input_data": getattr(record, "input_data", ""),
            "error_details": getattr(record, "error_details", ""),
            "status": getattr(record, "status", "Success"),
            "latency": getattr(record, "latency", ""),
            "latency_seconds": getattr(record, "latency_seconds", ""),
            "metric_type": getattr(record, "metric_type", ""),
            "environment": getattr(record, "environment", os.environ.get("ENV_NAME")),
            "logger_name": "[ASMOS BUYER ASSISTANT CHATBOT]",
            "source": f"{record.name}:{record.funcName}:{record.lineno}",
        }
        # Every other extra={...} field (e.g. search_query, channel, results) — previously dropped.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in log_entry:
                log_entry[key] = value
        # logger.exception(...) — keep the full stack trace.
        if record.exc_info:
            log_entry["traceback"] = self.formatException(record.exc_info)
        try:
            # Values that aren't JSON-serialisable are logged as their str().
            return json.dumps(log_entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # e.g. non-string dict keys or circular data
            simple = (str, int, float, bool, type(None))
            return json.dumps({k: v if isinstance(v, simple) else str(v) for k, v in log_entry.items()}, ensure_ascii=False)


def _file_handler() -> logging.Handler:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        LOGS_DIR / datetime.now().strftime("asmo-%Y-%m-%d-%H-%M-%S.log"),
        when="midnight", interval=1, backupCount=7, encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    return handler


# ---- Build the logger once, at import ----
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(JsonFormatter())

handlers: list[logging.Handler] = [console_handler]
setup_warning: str | None = None
app_insights = os.getenv("APP_INSIGHTS_CONNECTION_STRING", "").strip()

if IS_WINDOWS:
    try:
        handlers = [_file_handler()]
    except OSError:
        pass  # can't open the log file: keep the console rather than crash
elif app_insights:
    try:
        logger_provider = LoggerProvider()
        set_logger_provider(logger_provider)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(AzureMonitorLogExporter(connection_string=app_insights))
        )
        azure_handler = LoggingHandler()
        azure_handler.setFormatter(JsonFormatter())
        handlers = [console_handler, azure_handler]
    except Exception as exc:
        setup_warning = str(exc)

logger = logging.getLogger("mcit_logger")
logger.handlers = handlers
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
logger.propagate = False
if setup_warning:
    logger.warning(
        "app_insights_setup_failed",
        extra={"event": "app_insights_setup_failed", "error_details": setup_warning, "status": "failed"},
    )


def setup_logging(level: str = "INFO", to_file: bool = False) -> None:
    """Apply the app settings to the logger. Called once from app.py at import time."""
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    already_has_file = any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers)
    if to_file and not IS_WINDOWS and not already_has_file:
        try:
            logger.addHandler(_file_handler())
        except OSError as exc:
            logger.warning("log_file_setup_failed", extra={"error_details": str(exc), "status": "failed"})

    # config.py logs through logging.getLogger("app"); give it the same handlers.
    app_logger = logging.getLogger("app")
    app_logger.handlers = list(logger.handlers)
    app_logger.setLevel(logger.level)
    app_logger.propagate = False

    # Uvicorn's own logs (startup, errors, one access line per request) go to the same console/file.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = list(logger.handlers)
        uvicorn_logger.setLevel(logger.level)
        uvicorn_logger.propagate = False

    for noisy in ("urllib3", "httpx", "httpcore", "elastic_transport", "oci", "circuitbreaker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
