"""The application logger: one JSON object per line.

Where logs go:
  * Windows                              -> log file only (console if the file can't be opened)
  * APP_INSIGHTS_CONNECTION_STRING set   -> console + Azure Application Insights
  * otherwise                            -> console (LOG_TO_FILE=true also writes the log file)

Use the ready-made objects:

    from utils.logger import logger          # log anywhere
    from utils.logger import app_logging     # app.py: app_logging.setup(level, to_file) once at startup

The logging module calls ``JsonFormatter.format`` synchronously and the handlers are built when
this module is imported, so nothing here is async.
"""
import contextvars
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

from config import settings

# OpenTelemetry limits
os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "4096"
os.environ["OTEL_ATTRIBUTE_COUNT_LIMIT"] = "50"

# The current HTTP request's id. The request middleware sets it, so it appears on every log line of that request.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    """Formats every log line as one JSON object. A class because Python's logging requires one."""

    converter = time.gmtime  # the timestamp ends in "Z" (UTC), so it must be UTC
    # Attributes every LogRecord has. Anything else on a record came from ``extra={...}`` and is logged too.
    STANDARD_RECORD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}

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
            "environment": getattr(record, "environment", settings.env_name),
            "logger_name": "[ASMOS BUYER ASSISTANT CHATBOT]",
            "source": f"{record.name}:{record.funcName}:{record.lineno}",
        }
        # Every other extra={...} field (e.g. search_query, channel, results) — previously dropped.
        for key, value in record.__dict__.items():
            if key not in self.STANDARD_RECORD_ATTRS and key not in log_entry:
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


class AppLogging:
    def __init__(self, logs_dir: Path, is_windows: bool, level: str, app_insights_connection_string: str) -> None:
        self.logs_dir = logs_dir        # log files go here: <logs_dir>/asmo-YYYY-MM-DD-HH-MM-SS.log
        self.is_windows = is_windows    # Windows logs to the file only
        self.file_handler_name = "asmo_log_file"  # how our file handler is recognised (pytest adds FileHandlers too)

        # Build the app's logger once: console, or (Windows) a new log file, or console + Azure.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
        handlers: list[logging.Handler] = [console_handler]
        setup_warning: str | None = None
        if is_windows:
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)
                # A new file for every app start, named with the start time to the second.
                # "a" (append): never wipe a file another process started in the same second.
                file_handler = logging.FileHandler(logs_dir / datetime.now().strftime("asmo-%Y-%m-%d-%H-%M-%S.log"),
                                                   mode="a", encoding="utf-8")
                file_handler.setFormatter(JsonFormatter())
                file_handler.set_name(self.file_handler_name)
                handlers = [file_handler]
            except OSError:
                pass  # can't open the log file: keep the console rather than crash
        elif app_insights_connection_string.strip():
            try:
                logger_provider = LoggerProvider()
                set_logger_provider(logger_provider)
                logger_provider.add_log_record_processor(
                    BatchLogRecordProcessor(AzureMonitorLogExporter(connection_string=app_insights_connection_string.strip()))
                )
                azure_handler = LoggingHandler()
                azure_handler.setFormatter(JsonFormatter())
                handlers = [console_handler, azure_handler]
            except Exception as exc:
                setup_warning = str(exc)

        self.logger = logging.getLogger("mcit_logger")
        self.logger.handlers = handlers
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.propagate = False
        if setup_warning:
            self.logger.warning("app_insights_setup_failed", extra={
                "event": "app_insights_setup_failed", "error_details": setup_warning, "status": "failed"})

    def setup(self, level: str = "INFO", to_file: bool = False) -> None:
        """Apply the app settings to the logger. Called once from app.py at import time."""
        logger = self.logger
        logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))

        # 1. LOG_TO_FILE: add a new log file for this app start (Windows already has one).
        already_has_file = any(h.get_name() == self.file_handler_name for h in logger.handlers)
        if to_file and not self.is_windows and not already_has_file:
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.logs_dir / datetime.now().strftime("asmo-%Y-%m-%d-%H-%M-%S.log"),
                                                   mode="a", encoding="utf-8")
                file_handler.setFormatter(JsonFormatter())
                file_handler.set_name(self.file_handler_name)
                logger.addHandler(file_handler)
            except OSError as exc:
                logger.warning("log_file_setup_failed", extra={"error_details": str(exc), "status": "failed"})

        # 2. config.py logs through logging.getLogger("app"); give it the same handlers.
        app_logger = logging.getLogger("app")
        app_logger.handlers = list(logger.handlers)
        app_logger.setLevel(logger.level)
        app_logger.propagate = False

        # Uvicorn's own logs (server startup, access lines) are left alone: they stay in the
        # cmd window only and are not written to the .log file.

        # 3. Everything else (OpenSearch client, OCI SDK, asyncio background-task errors, ...) reaches
        #    the root logger; without handlers there it would print to stderr only, never to the file.
        root_logger = logging.getLogger()
        root_logger.handlers = list(logger.handlers)
        root_logger.setLevel(logging.WARNING)  # only libraries' warnings/errors, not their chatter
        # Python warnings (DeprecationWarning, InsecureRequestWarning, ...) go through logging too.
        logging.captureWarnings(True)

        for noisy in ("urllib3", "httpx", "httpcore", "elastic_transport", "oci", "circuitbreaker"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# The one object the app uses — import it where needed:  from utils.logger import app_logging
app_logging = AppLogging(
    logs_dir=Path(__file__).resolve().parent.parent / "logs",  # <repo>/logs, whatever the working directory
    is_windows=platform.system() == "Windows",
    level=settings.log_level,
    app_insights_connection_string=settings.app_insights_connection_string,
)
logger = app_logging.logger  # the logger itself: from utils.logger import logger
