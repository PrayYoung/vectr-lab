"""Logging helpers using rich."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

_LOG_FORMAT = "%(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure global logging with rich handler."""

    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
