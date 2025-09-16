"""Structured logging helpers with contextual information."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace(self, msg: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, msg, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]

_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "vectr_lab_log_context", default={}
)


class ContextFilter(logging.Filter):
    """Inject context variables into each record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        ctx = _CONTEXT.get()
        for key, value in ctx.items():
            setattr(record, key, value)
        if not hasattr(record, "run_id"):
            setattr(record, "run_id", ctx.get("run_id", ""))
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in ("run_id", "strategy", "ticker"):
            value = getattr(record, attr, None)
            if value:
                payload[attr] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base = super().format(record)
        ctx_parts = []
        for attr in ("run_id", "strategy", "ticker"):
            value = getattr(record, attr, None)
            if value:
                ctx_parts.append(f"{attr}={value}")
        ctx_str = f" [{' '.join(ctx_parts)}]" if ctx_parts else ""
        return f"{base}{ctx_str}"


def configure_logging(*, debug: bool = False, json_logs: bool = False, trace: bool = False) -> None:
    """Configure global logging for console output."""

    root = logging.getLogger()
    root.handlers.clear()
    level = TRACE_LEVEL if trace else (logging.DEBUG if debug else logging.INFO)
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextFilter())
    if json_logs:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = PrettyFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s", "[%X]")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def add_file_handler(path: Path, *, json_logs: bool = False) -> None:
    handler = logging.FileHandler(path)
    handler.addFilter(ContextFilter())
    handler.setFormatter(JSONFormatter() if json_logs else PrettyFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
    logging.getLogger().addHandler(handler)


def push_context(**kwargs: Any) -> contextvars.Token:
    ctx = _CONTEXT.get().copy()
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    return _CONTEXT.set(ctx)


def pop_context(token: contextvars.Token) -> None:
    _CONTEXT.reset(token)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)


__all__ = [
    "TRACE_LEVEL",
    "configure_logging",
    "add_file_handler",
    "push_context",
    "pop_context",
    "get_logger",
]
