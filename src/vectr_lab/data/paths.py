"""Utilities for working with local data paths."""

from __future__ import annotations

from pathlib import Path

_CACHE_ROOT = Path(".cache")


def cache_root() -> Path:
    """Return the root directory for cached data."""

    return (_CACHE_ROOT if _CACHE_ROOT.is_absolute() else Path.cwd() / _CACHE_ROOT).resolve()


def cache_path_for(ticker: str, timeframe: str) -> Path:
    """Return the cache path for a given ticker/timeframe pair."""

    root = cache_root() / timeframe
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{ticker}.parquet"


__all__ = ["cache_path_for", "cache_root"]
