"""Data ingestion utilities using yfinance with local caching."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Iterable

import pandas as pd
import yfinance as yf

from vectr_lab.config.models import UniverseConfig
from vectr_lab.data.paths import cache_path_for
from vectr_lab.utils.log import get_logger

_LOG = get_logger(__name__)

_REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(str.lower, axis="columns")
    missing = {col for col in _REQUIRED_COLUMNS if col not in frame.columns}
    if missing:
        raise ValueError(f"Downloaded data missing columns: {missing}")
    frame = frame[_REQUIRED_COLUMNS].copy()
    frame.index.name = "timestamp"
    return frame


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
            return _normalize_frame(data)
        _LOG.warning("No data returned for %s on attempt %s", ticker, attempt)
        time.sleep(delay)
    raise RuntimeError(f"Failed to download data for {ticker} after {retries} attempts")


def load_cached_or_download(ticker: str, universe: UniverseConfig, force: bool = False) -> pd.DataFrame:
    """Load cached parquet data or fetch it from yfinance."""

    cache_path = cache_path_for(ticker, universe.timeframe)
    if cache_path.exists() and not force:
        _LOG.info("Loading %s from cache", ticker)
        return pd.read_parquet(cache_path)

    _LOG.info("Downloading %s", ticker)
    frame = _download_ticker(ticker, universe)
    frame.to_parquet(cache_path)
    return frame


def download_universe(universe: UniverseConfig, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Download and cache OHLCV data for all universe tickers."""

    data: Dict[str, pd.DataFrame] = {}
    for ticker in universe.tickers:
        data[ticker] = load_cached_or_download(ticker, universe, force=force)
    return data


__all__ = ["download_universe", "load_cached_or_download"]
