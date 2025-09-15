"""Metrics helpers for backtest results."""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from vectr_lab.backtest.runner import BacktestResult


def _cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    if start_value <= 0:
        return 0.0
    elapsed_days = (equity.index[-1] - equity.index[0]).days
    years = elapsed_days / 365.25 if elapsed_days > 0 else len(equity) / 252
    years = max(years, 1e-9)
    return (end_value / start_value) ** (1 / years) - 1


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    return math.sqrt(periods_per_year) * returns.mean() / returns.std()


def compute_portfolio_metrics(result: BacktestResult) -> Dict[str, float]:
    equity = result.equity_curve
    returns = equity.pct_change().dropna()

    trades = result.trades
    wins = 0
    total_trades = len(trades)
    avg_r = 0.0

    if total_trades > 0:
        pnl_column = next(
            (col for col in ["return", "return_pct", "pnl"] if col in trades.columns),
            trades.columns[-1],
        )
        wins = int((trades[pnl_column] > 0).sum())
        avg_r = float(trades[pnl_column].mean())

    exposure = float(result.entries.any(axis=1).mean()) if not result.entries.empty else 0.0

    turnover = float(result.portfolio.returns().abs().sum().sum())

    return {
        "cagr": _cagr(equity),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _sharpe(returns),
        "win_rate": wins / total_trades if total_trades else 0.0,
        "avg_r": avg_r,
        "exposure": exposure,
        "turnover": turnover,
    }


def compute_per_ticker_metrics(result: BacktestResult) -> pd.DataFrame:
    portfolio = result.portfolio
    total_return = portfolio.total_return()
    max_dd = portfolio.max_drawdown()
    exposure = result.entries.mean()
    metrics = pd.DataFrame({
        "total_return": total_return,
        "max_drawdown": max_dd,
        "exposure": exposure,
    })
    metrics.index.name = "ticker"
    return metrics


__all__ = ["compute_portfolio_metrics", "compute_per_ticker_metrics"]
