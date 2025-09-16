"""Data ingestion utilities using yfinance with local caching."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from vectr_lab.config.models import UniverseConfig
from vectr_lab.data.paths import cache_path_for
from vectr_lab.utils.log import get_logger

_LOG = get_logger(__name__)


_REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
_OPTIONAL_COLUMNS = ["adj_close"]
_CACHE_SCHEMA_VERSION = 1


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


_DEBUG = _env_flag("VECTR_LAB_DEBUG")
_STRICT = _env_flag("VECTR_LAB_STRICT")


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _write_cache(cache_path: Path, frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    meta = {"schema_version": _CACHE_SCHEMA_VERSION}
    _meta_path(cache_path).write_text(json.dumps(meta))


def _read_cache(cache_path: Path) -> tuple[pd.DataFrame | None, dict | None]:
    try:
        frame = pd.read_parquet(cache_path)
    except Exception as exc:  # pragma: no cover - defensive read
        _LOG.warning("Failed to load cache %s: %s", cache_path, exc)
        return None, None
    meta = None
    meta_file = _meta_path(cache_path)
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:  # pragma: no cover - defensive read
            _LOG.warning("Invalid cache metadata at %s", meta_file)
    return frame, meta


def _normalize_label(value: object) -> str:
    label = str(value).strip().lower().replace(" ", "_")
    replacements = {"adjclose": "adj_close"}
    return replacements.get(label, label)


def _select_field(df: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    field_key = field.replace("_", "")
    ticker_key = ticker.lower().replace("_", "")
    best = None
    best_score = -1
    for col in df.columns:
        parts = col if isinstance(col, tuple) else (col,)
        norm_parts = [_normalize_label(part) for part in parts]
        condensed = [part.replace("_", "") for part in norm_parts]
        if field_key not in condensed:
            continue
        score = 0
        if ticker_key in condensed:
            score += 10
        if condensed[-1] == field_key:
            score += 5
        if len(parts) == 1:
            score += 2
        if score > best_score:
            best = col
            best_score = score
    if best is None:
        raise KeyError(f"'{field}' column missing for {ticker}")
    series = df[best]
    if isinstance(series, pd.DataFrame):
        if _STRICT:
            raise ValueError(f"{field} for {ticker} expanded to shape {series.shape}")
        if _DEBUG:
            _LOG.debug("Collapsing multi-column field %s[%s] shape=%s", field, ticker, series.shape)
        series = series.iloc[:, 0]
    return series


def _normalize_ohlcv(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        needed = _REQUIRED_COLUMNS + ["adj_close"]
        extracted: Dict[str, pd.Series] = {}
        for field in needed:
            try:
                series = _select_field(df, ticker, field)
            except KeyError:
                if field != "adj_close":
                    raise
                continue
            extracted[field] = series
        df = pd.DataFrame(extracted, index=df.index)
    else:
        df.columns = [_normalize_label(col) for col in df.columns]

    df.columns = [_normalize_label(col) for col in df.columns]
    available_columns = [col for col in _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS if col in df.columns]
    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Normalized frame missing columns {missing} for {ticker}; columns={list(df.columns)[:10]}")

    df = df[available_columns].copy()
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "timestamp"
    return df


def _download_ticker(ticker: str, universe: UniverseConfig) -> pd.DataFrame:
    params = {
        "tickers": ticker,
        "interval": universe.timeframe,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
    }
    if universe.start:
        params["start"] = universe.start
    if universe.end:
        params["end"] = universe.end

    retries = 3
    delay = 2.0
    for attempt in range(1, retries + 1):
        data = yf.download(**params)
        if not data.empty:
            return _normalize_ohlcv(data, ticker)
        _LOG.warning("No data returned for %s on attempt %s", ticker, attempt)
        time.sleep(delay)
    raise RuntimeError(f"Failed to download data for {ticker} after {retries} attempts")


def load_cached_or_download(ticker: str, universe: UniverseConfig, force: bool = False) -> pd.DataFrame:
    """Load cached parquet data or fetch it from yfinance."""

    cache_path = cache_path_for(ticker, universe.timeframe)
    if cache_path.exists() and not force:
        frame, meta = _read_cache(cache_path)
        if frame is not None:
            needs_rewrite = meta is None or meta.get("schema_version") != _CACHE_SCHEMA_VERSION
            if isinstance(frame.columns, pd.MultiIndex):
                needs_rewrite = True
            try:
                normalized = _normalize_ohlcv(frame, ticker)
            except Exception as exc:
                _LOG.warning("Cached data for %s invalid, rebuilding: %s", ticker, exc)
            else:
                frame = normalized
                if needs_rewrite:
                    _LOG.info("Rewriting cache for %s to schema %s", ticker, _CACHE_SCHEMA_VERSION)
                    _write_cache(cache_path, frame)
                return frame

    _LOG.info("Downloading %s", ticker)
    frame = _download_ticker(ticker, universe)
    _write_cache(cache_path, frame)
    return frame


def download_universe(universe: UniverseConfig, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Download and cache OHLCV data for all universe tickers."""

    data: Dict[str, pd.DataFrame] = {}
    for ticker in universe.tickers:
        data[ticker] = load_cached_or_download(ticker, universe, force=force)
    return data


__all__ = ["download_universe", "load_cached_or_download"]
