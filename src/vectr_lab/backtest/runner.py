"""Backtest runner built on vectorbt."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import os
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.risk.rules import apply_risk_rules
from vectr_lab.risk.sizer import sizes_fixed_risk
from vectr_lab.strategies.base import BaseStrategy
from vectr_lab.utils.log import get_logger, pop_context, push_context


@dataclass
class BacktestResult:
    """Container returned by the backtest runner."""

    portfolio: vbt.Portfolio
    entries: pd.DataFrame
    exits: pd.DataFrame
    signals: Mapping[str, pd.DataFrame]
    trace_data: Optional[Mapping[str, pd.DataFrame]] = None

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

    if _LOG.isEnabledFor(logging.DEBUG):
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

    if _LOG.isEnabledFor(logging.DEBUG):
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
    *,
    run_id: Optional[str] = None,
    trace_tickers: Optional[Iterable[str]] = None,
) -> BacktestResult:
    """Run a multi-asset backtest using vectorbt."""

    close_map: Dict[str, pd.Series] = {}
    entry_map: Dict[str, pd.Series] = {}
    exit_map: Dict[str, pd.Series] = {}
    sl_map: Dict[str, pd.Series] = {}
    tp_map: Dict[str, pd.Series] = {}
    signals: Dict[str, pd.DataFrame] = {}

    trace_set = {t.upper() for t in trace_tickers} if trace_tickers else set()

    run_token = push_context(run_id=run_id, strategy=strategy.name)
    try:
        for ticker, raw_df in data.items():
            ticker_token = push_context(ticker=ticker)
            try:
                df = raw_df.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    raise ValueError(
                        f"Data for {ticker} still has MultiIndex columns; ensure ingest normalization ran. Columns sample: {list(df.columns)[:5]}"
                    )

                if "close" not in df.columns:
                    raise KeyError(
                        f"'close' column missing for {ticker}; columns={list(df.columns)[:10]}"
                    )

                if _LOG.isEnabledFor(logging.INFO):
                    _LOG.info(
                        "ticker_loaded rows=%d start=%s end=%s",
                        len(df),
                        df.index.min(),
                        df.index.max(),
                    )

                assert isinstance(df["close"], pd.Series), (
                    f"close column for {ticker} is not Series: {type(df['close'])}"
                )
                features = strategy.compute_features(df).reindex(df.index)
                joined = df.copy()
                for column in features.columns:
                    joined[column] = features[column]
                ticker_signals = strategy.generate_signals(joined)
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        "strategy_signals type=%s shape=%s columns=%s",
                        type(ticker_signals).__name__,
                        getattr(ticker_signals, "shape", None),
                        getattr(ticker_signals, "columns", None),
                    )

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

                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug(
                        "signal_counts entries=%d exits=%d sl_nan=%d tp_nan=%d",
                        int(entry_series.sum()),
                        int(exit_manual.sum()),
                        int(stop_series.isna().sum()),
                        int(target_series.isna().sum()),
                    )
            finally:
                pop_context(ticker_token)

        close_df = pd.concat(close_map, axis=1)
        close_df.columns = pd.Index(close_map.keys())
        close_df = close_df.sort_index()
        tickers = list(close_df.columns)
        master_index = close_df.index

        entries_df = _build_matrix("entries", entry_map, tickers, master_index, bool)
        exits_df = _build_matrix("exits", exit_map, tickers, master_index, bool)
        sl_df = _build_matrix("stop_loss", sl_map, tickers, master_index, float)
        tp_df = _build_matrix("take_profit", tp_map, tickers, master_index, float)

        _LOG.info(
            "assembled_matrices entries=%s exits=%s stop=%s take=%s",
            entries_df.shape,
            exits_df.shape,
            sl_df.shape,
            tp_df.shape,
        )

        returns = close_df.pct_change().fillna(0.0)

        size_result = sizes_fixed_risk(
            close_df,
            sl_df,
            risk.risk_pct,
            return_diagnostics=_LOG.isEnabledFor(logging.DEBUG),
        )
        if isinstance(size_result, tuple):
            size_df, size_diag = size_result
            for ticker, diag in size_diag.items():
                _LOG.debug("sizer_stats %s %s", ticker, diag)
        else:
            size_df = size_result

        masked_entries, risk_stats = apply_risk_rules(
            entries_df,
            exits_df,
            returns,
            risk,
            return_stats=True,
        )
        _LOG.info("risk_mask_stats %s", risk_stats)

        size_df = size_df.where(masked_entries, other=0.0)

        portfolio = vbt.Portfolio.from_signals(
            close=close_df,
            entries=masked_entries,
            exits=exits_df,
            size=size_df,
            fees=universe.fees_pct,
            slippage=universe.slippage_pct,
        )

        trades_records = portfolio.trades.records
        trades_count = len(trades_records)
        total_return_obj = portfolio.total_return()
        if isinstance(total_return_obj, (pd.Series, pd.DataFrame)):
            total_return_value = float(total_return_obj.sum()) if not total_return_obj.empty else 0.0
        else:
            total_return_value = float(total_return_obj)
        exposure = float(masked_entries.any(axis=1).mean()) if not masked_entries.empty else 0.0
        _LOG.info(
            "portfolio_summary trades=%d total_return=%.4f exposure=%.4f",
            trades_count,
            total_return_value,
            exposure,
        )

        trace_outputs: Dict[str, pd.DataFrame] = {}
        if trace_set:
            for ticker in tickers:
                if ticker.upper() not in trace_set:
                    continue
                trace_outputs[ticker] = pd.DataFrame(
                    {
                        "open": data[ticker]["open"],
                        "high": data[ticker]["high"],
                        "low": data[ticker]["low"],
                        "close": close_df[ticker],
                        "volume": data[ticker]["volume"],
                        "entry_raw": entries_df[ticker],
                        "entry_final": masked_entries[ticker],
                        "exit": exits_df[ticker],
                        "stop": sl_df[ticker],
                        "target": tp_df[ticker] if ticker in tp_df.columns else pd.Series(index=master_index, dtype=float),
                        "size": size_df[ticker],
                    }
                )

        return BacktestResult(
            portfolio=portfolio,
            entries=masked_entries,
            exits=exits_df,
            signals=signals,
            trace_data=trace_outputs or None,
        )
    finally:
        pop_context(run_token)


__all__ = ["BacktestResult", "run_backtest", "register_strategy", "get_strategy"]
