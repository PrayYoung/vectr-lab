"""Backtest runner built on vectorbt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd
import vectorbt as vbt

from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.risk.rules import apply_risk_rules
from vectr_lab.risk.sizer import sizes_fixed_risk
from vectr_lab.strategies.base import BaseStrategy


@dataclass
class BacktestResult:
    """Container returned by the backtest runner."""

    portfolio: vbt.Portfolio
    entries: pd.DataFrame
    exits: pd.DataFrame
    signals: Mapping[str, pd.DataFrame]

    @property
    def equity_curve(self) -> pd.Series:
        return self.portfolio.value()

    @property
    def trades(self) -> pd.DataFrame:
        return self.portfolio.trades.records

    def stats(self) -> pd.Series:
        return self.portfolio.stats()


_STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {}


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

    for ticker, df in data.items():
        features = strategy.compute_features(df)
        joined = features.join(df, how="left")
        ticker_signals = strategy.generate_signals(joined)

        signals[ticker] = ticker_signals
        close_map[ticker] = df["close"]
        entry_map[ticker] = ticker_signals["entry_long"].astype(bool)

        stop_losses = ticker_signals.get("sl_price", pd.Series(index=df.index, dtype=float))
        take_profit = ticker_signals.get("tp_price", pd.Series(index=df.index, dtype=float))
        sl_map[ticker] = stop_losses.ffill()
        tp_map[ticker] = take_profit.ffill()

        exit_long = ticker_signals.get("exit_long", pd.Series(False, index=df.index))
        exit_sl = (df["close"] <= sl_map[ticker])
        exit_tp = (df["close"] >= tp_map[ticker])
        exit_map[ticker] = (exit_long | exit_sl | exit_tp).shift(1, fill_value=False)

    close_df = pd.DataFrame(close_map).sort_index()
    entries_df = pd.DataFrame(entry_map).reindex(close_df.index, fill_value=False)
    exits_df = pd.DataFrame(exit_map).reindex(close_df.index, fill_value=False)
    sl_df = pd.DataFrame(sl_map).reindex(close_df.index)

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
