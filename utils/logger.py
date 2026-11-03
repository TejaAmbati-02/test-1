import contextvars
import logging
import functools
import inspect
import json
import os
import platform
import sys
import time
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from types import ModuleType
 
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
 
# Set OpenTelemetry limits
os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "4096"
os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "50"
 
IS_WINDOWS = platform.system() == "Windows"

# Per-request id, set by the request middleware so it appears on every log line of that request.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def set_log_context(request_id: str) -> None:
    _request_id.set(request_id)


def clear_log_context() -> None:
    _request_id.set("")

 
class JsonFormatter(logging.Formatter):
    # The "Z" suffix means UTC, so the timestamp must actually be UTC.
    converter = time.gmtime
 
    def format(self, record):
        def safe(val):
            try:
                json.dumps(val)
                return val
            except Exception:
                return str(val)
 
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", None) or _request_id.get(),
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
            "request_payload": safe(getattr(record, "request_payload", {})),
            "response_payload": safe(getattr(record, "response_payload", {})),
            "function_name": getattr(record, "function_name", ""),
            "input_data": safe(getattr(record, "input_data", "")),
            "error_details": safe(getattr(record, "error_details", "")),
            "status": getattr(record, "status", "Success"),
            "latency": getattr(record, "latency", ""),
            "latency_seconds": getattr(record, "latency_seconds", ""),
            "metric_type": getattr(record, "metric_type", ""),
            "environment": getattr(record, "environment", os.environ.get("ENV_NAME")),
            "logger_name": "[ASMOS BUYER ASSISTANT CHATBOT]",
        }
 
        return json.dumps(log_entry, ensure_ascii=False)
 
 
def _configure_logger(handlers) -> logging.Logger:
    logger = logging.getLogger("mcit_logger")
    logger.handlers.clear()
 
    for handler in handlers:
        logger.addHandler(handler)
 
    configured_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, configured_level, logging.INFO))
    logger.propagate = False
    return logger
 
 
def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    return handler
 
 
def _file_handler() -> logging.Handler:
    # <repo>/logs, regardless of the working directory the app is started from.
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
 
    log_filename = datetime.now().strftime("buyer-assistant-logs-%Y_%m_%d %H_%M_%S.log")
 
    handler = TimedRotatingFileHandler(
        logs_dir / log_filename,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    return handler
 
def setup_local_logger() -> logging.Logger:
    """Windows: log file only. Any other OS: console only."""
    if IS_WINDOWS:
        try:
            return _configure_logger([_file_handler()])
        except OSError:
            # Can't create/open the log file: fall back to console rather than crash.
            return _configure_logger([_console_handler()])
    return _configure_logger([_console_handler()])
 
 
def setup_app_insights_logger() -> logging.Logger:
    connection_string = os.getenv("APP_INSIGHTS_CONNECTION_STRING", "").strip()
    if not connection_string:
        return setup_local_logger()
 
    try:
        logger_provider = LoggerProvider()
        set_logger_provider(logger_provider)
        exporter = AzureMonitorLogExporter(connection_string=connection_string)
        logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
 
        azure_handler = LoggingHandler()
        azure_handler.setFormatter(JsonFormatter())
        return _configure_logger([_console_handler(), azure_handler])
 
    except Exception as exc:
        fallback = setup_local_logger()
        fallback.warning(
            "app_insights_setup_failed",
            extra={"event": "app_insights_setup_failed", "error_details": str(exc), "status": "failed"},
        )
        return fallback
 
 
# Initialize the logger
if IS_WINDOWS:
    logger = setup_local_logger()
else:
    logger = setup_app_insights_logger()


def setup_logging(level: str = "INFO", to_file: bool = False) -> None:
    """Apply app settings to the logger built at import time.

    Windows always logs to the file only. On other OSes logs go to the console, and
    ``to_file=True`` (LOG_TO_FILE=true) also writes the log file.
    """
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if to_file and not IS_WINDOWS and not any(
        isinstance(h, TimedRotatingFileHandler) for h in logger.handlers
    ):
        try:
            logger.addHandler(_file_handler())
        except OSError as exc:
            logger.warning("log_file_setup_failed", extra={"error_details": str(exc), "status": "failed"})

    # Modules that use logging.getLogger("app") (e.g. config.py) share the same handlers.
    app_logger = logging.getLogger("app")
    app_logger.handlers = list(logger.handlers)
    app_logger.setLevel(logger.level)
    app_logger.propagate = False

    for name in ("urllib3", "httpx", "httpcore", "elastic_transport", "oci", "circuitbreaker"):
        logging.getLogger(name).setLevel(logging.WARNING)