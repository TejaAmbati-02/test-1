from __future__ import annotations

import functools
import inspect
import json
import logging
import logging.handlers
import os
import sys
import threading
from datetime import datetime
from pathlib import Path


# Per-request context 
# Set once at the start of a request so request_id flows into every log line
# produced by that thread without being passed explicitly.

_log_context = threading.local()


def set_log_context(request_id: str) -> None:
    _log_context.request_id = request_id


def clear_log_context() -> None:
    _log_context.request_id = ""


# JSON formatter 

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        def safe(val):
            try:
                json.dumps(val)
                return val
            except Exception:
                return str(val)

        entry = {
            "timestamp":        self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level":            record.levelname,
            "request_id":       getattr(record, "request_id", None) or getattr(_log_context, "request_id", ""),
            "message":          record.getMessage(),
            "search_query":     getattr(record, "search_query", ""),
            "language":         getattr(record, "language", ""),
            "channel":          getattr(record, "channel", ""),
            "function_name":    getattr(record, "function_name", ""),
            "input_data":       safe(getattr(record, "input_data", "")),
            "response_payload": safe(getattr(record, "response_payload", {})),
            "error_details":    safe(getattr(record, "error_details", "")),
            "status":           getattr(record, "status", "success"),
            "latency":          getattr(record, "latency", ""),
            "environment":      getattr(record, "environment", os.environ.get("ENV_NAME", "local")),
            "logger_name":      "[ASMO Search]",
        }
        return json.dumps(entry, ensure_ascii=False)


# Logging service 

class LoggingService:
    def __init__(self) -> None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self._configured = False
        self.logger = logging.getLogger("app")

    def setup(self, level: int = logging.INFO) -> None:
        if self._configured:
            return

        root = logging.getLogger()
        root.setLevel(level)

        log_file = self.log_dir / f"asmo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

        # Rotating file handler — new file each midnight, keeps 7 days
        fh = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        fh.setFormatter(JsonFormatter())
        fh.setLevel(level)

        # Console handler — human-readable for dev
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        ch.setLevel(level)

        root.addHandler(fh)
        root.addHandler(ch)

        for name in ("urllib3", "httpx", "httpcore", "elastic_transport", "oci", "circuitbreaker"):
            logging.getLogger(name).setLevel(logging.WARNING)

        self._configured = True
        root.info(f"Logging initialised — file: {log_file}")

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)


# Decorator: auto-log latency for any function 

def log_function(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = datetime.now()
        logger.debug(f"→ {func.__name__} called")
        try:
            result = await func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "success",
                },
            )
            return result
        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s — {exc}",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                },
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = datetime.now()
        logger.debug(f"→ {func.__name__} called")
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "success",
                },
            )
            return result
        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s — {exc}",
                extra={
                    "function_name": func.__name__,
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                },
            )
            raise

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


# Singleton 

logging_service = LoggingService()

logger = logging_service.logger
setup_logging = logging_service.setup
