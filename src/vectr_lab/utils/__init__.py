"""Shared utilities for vectr_lab."""

from .log import configure_logging, get_logger
from .time import now_utc

__all__ = ["configure_logging", "get_logger", "now_utc"]
