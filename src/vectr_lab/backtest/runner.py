"""Backtest runner built on vectorbt."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Mapping

import numpy as np
import pandas as pd
import vectorbt as vbt

from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.risk.rules import apply_risk_rules
from vectr_lab.risk.sizer import sizes_fixed_risk
from vectr_lab.strategies.base import BaseStrategy
from vectr_lab.utils.log import get_logger


@dataclass
class BacktestResult:
    """Container returned by the backtest runner."""

    portfolio: vbt.Portfolio
    entries: pd.DataFrame
    exits: pd.DataFrame
    signals: Mapping[str, pd.DataFrame]

    @property
    def equity_curve(self) -> pd.Series:
        value = self.portfolio.value()
        if isinstance(value, pd.DataFrame):
            return value.sum(axis=1)
        return value

    @property
    def trades(self) -> pd.DataFrame:
        return self.portfolio.trades.records

    def stats(self) -> pd.Series:
        return self.portfolio.stats()


_STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


_DEBUG = _env_flag("VECTR_LAB_DEBUG")
_STRICT = _env_flag("VECTR_LAB_STRICT")
_LOG = get_logger(__name__)


def _standardize_series(
    ticker: str,
    label: str,
    value: object,
    index: pd.Index,
    dtype: type,
    fill_value: bool | float,
):
    original_type = type(value).__name__
    series: pd.Series

    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            msg = f"{label}[{ticker}]: expected single column, received shape {value.shape}."
            raise ValueError(msg)
        value = value.iloc[:, 0]

    if isinstance(value, pd.Series):
        series = value.copy()
    elif np.isscalar(value) or value is None:
        series = pd.Series(fill_value, index=index)
    else:
        try:
            series = pd.Series(value)
        except Exception as exc:  # pragma: no cover - defensive conversion
            if _STRICT:
                raise
            _LOG.warning(
                "%s[%s]: failed to coerce %s (%s); using fill value",
                label,
                ticker,
                original_type,
                exc,
            )
            series = pd.Series(fill_value, index=index)

    if not series.index.equals(index):
        series = series.reindex(index)

    if dtype is bool:
        series = series.fillna(False).astype(bool)
    else:
        series = series.astype(float)
    series.name = ticker

    if _DEBUG:
        _LOG.debug(
            "standardize %s[%s]: input=%s -> dtype=%s nulls=%s",
            label,
            ticker,
            original_type,
            series.dtype,
            int(series.isna().sum()),
        )

    return series


def _build_matrix(
    label: str,
    data_map: Dict[str, object],
    tickers: list[str],
    index: pd.Index,
    dtype: type,
):
    fill_value: bool | float = False if dtype is bool else np.nan
    standardized: Dict[str, pd.Series] = {}
    for ticker in tickers:
        value = data_map.get(ticker)
        standardized[ticker] = _standardize_series(
            ticker,
            label,
            value,
            index,
            dtype,
            fill_value,
        )
    frame = pd.DataFrame(standardized, index=index)
    if dtype is bool:
        frame = frame.fillna(False).astype(bool)
    else:
        frame = frame.astype(float)

    if _STRICT:
        if frame.index.difference(index).any() or len(frame.columns) != len(tickers):
            raise ValueError(f"{label} matrix shape mismatch: {frame.shape}")

    if _DEBUG:
        _LOG.debug("%s matrix -> shape %s", label, frame.shape)

    return frame


def register_strategy(strategy_cls: type[BaseStrategy]) -> None:
    _STRATEGY_REGISTRY[strategy_cls.name] = strategy_cls


def get_strategy(strategy_config: StrategyConfig) -> BaseStrategy:
    cls = _STRATEGY_REGISTRY.get(strategy_config.name)
    if cls is None:
        raise KeyError(f"Strategy {strategy_config.name} not registered")
    return cls.from_config(strategy_config)


def run_backtest(
    data: Dict[str, pd.DataFrame],
    strategy: BaseStrategy,
    universe: UniverseConfig,
    risk: RiskConfig,
) -> BacktestResult:
    """Run a multi-asset backtest using vectorbt."""

    close_map: Dict[str, pd.Series] = {}
    entry_map: Dict[str, pd.Series] = {}
    exit_map: Dict[str, pd.Series] = {}
    sl_map: Dict[str, pd.Series] = {}
    tp_map: Dict[str, pd.Series] = {}
    signals: Dict[str, pd.DataFrame] = {}

    for ticker, raw_df in data.items():
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            needed = ['open', 'high', 'low', 'close', 'volume']
            mapping = {str(col).lower(): col for col in df.columns.get_level_values(-1)}
            extracted = {}
            for field in needed:
                actual = mapping.get(field)
                if actual is None:
                    raise KeyError(f"'{field}' column missing for {ticker}")
                series = df.xs(actual, axis=1, level=-1)
                if isinstance(series, pd.DataFrame) and 'Ticker' in series.columns:
                    series = series['Ticker']
                if isinstance(series, pd.DataFrame):
                    if _STRICT:
                        raise ValueError(f"{field} for {ticker} expanded to shape {series.shape}")
                    series = series.iloc[:, 0]
                extracted[field] = series
            df = pd.DataFrame(extracted, index=df.index)

        if 'close' not in df.columns:
            raise KeyError(f"'close' column missing for {ticker} after flattening")

        assert isinstance(df['close'], pd.Series), f"close column for {ticker} is not Series: {type(df['close'])}"
        features = strategy.compute_features(df).reindex(df.index)
        joined = df.copy()
        for column in features.columns:
            joined[column] = features[column]
        ticker_signals = strategy.generate_signals(joined)
        if _DEBUG:
            _LOG.debug("strategy signals[%s]: type=%s shape=%s columns=%s", ticker, type(ticker_signals).__name__, getattr(ticker_signals, 'shape', None), getattr(ticker_signals, 'columns', None))

        signals[ticker] = ticker_signals
        close_map[ticker] = df["close"]

        entry_series = _standardize_series(
            ticker,
            "entries_manual",
            ticker_signals.get("entry_long", pd.Series(False, index=df.index)),
            df.index,
            bool,
            False,
        )
        entry_map[ticker] = entry_series

        exit_manual = _standardize_series(
            ticker,
            "exits_manual",
            ticker_signals.get("exit_long", pd.Series(False, index=df.index)),
            df.index,
            bool,
            False,
        )

        stop_series = _standardize_series(
            ticker,
            "stop_loss_raw",
            ticker_signals.get("sl_price", pd.Series(np.nan, index=df.index)),
            df.index,
            float,
            np.nan,
        ).ffill()
        sl_map[ticker] = stop_series

        target_series = _standardize_series(
            ticker,
            "take_profit_raw",
            ticker_signals.get("tp_price", pd.Series(np.nan, index=df.index)),
            df.index,
            float,
            np.nan,
        ).ffill()
        tp_map[ticker] = target_series

        exit_sl = df["close"].le(stop_series)
        exit_tp = df["close"].ge(target_series)

        exit_map[ticker] = (exit_manual | exit_sl | exit_tp).shift(1, fill_value=False)

        if _DEBUG:
            _LOG.debug("exit components[%s]: manual=%s sl=%s tp=%s result=%s",
                ticker, type(exit_manual).__name__, type(exit_sl).__name__, type(exit_tp).__name__, type(exit_map[ticker]).__name__)

    close_df = pd.concat(close_map, axis=1)
    close_df.columns = pd.Index(close_map.keys())
    close_df = close_df.sort_index()
    tickers = list(close_df.columns)
    master_index = close_df.index

    entries_df = _build_matrix("entries", entry_map, tickers, master_index, bool)
    exits_df = _build_matrix("exits", exit_map, tickers, master_index, bool)
    sl_df = _build_matrix("stop_loss", sl_map, tickers, master_index, float)
    tp_df = _build_matrix("take_profit", tp_map, tickers, master_index, float)

    if _DEBUG:
        _LOG.debug(
            "aligned matrices: entries=%s exits=%s stop=%s take=%s",
            entries_df.shape,
            exits_df.shape,
            sl_df.shape,
            tp_df.shape,
        )

    returns = close_df.pct_change().fillna(0.0)
    masked_entries = apply_risk_rules(entries_df, exits_df, returns, risk)

    size_df = sizes_fixed_risk(close_df, sl_df, risk.risk_pct)
    size_df = size_df.where(masked_entries, other=0.0)

    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=masked_entries,
        exits=exits_df,
        size=size_df,
        fees=universe.fees_pct,
        slippage=universe.slippage_pct,
    )

    return BacktestResult(
        portfolio=portfolio,
        entries=masked_entries,
        exits=exits_df,
        signals=signals,
    )


__all__ = ["BacktestResult", "run_backtest", "register_strategy", "get_strategy"]
