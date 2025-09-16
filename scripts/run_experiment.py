#!/usr/bin/env python3
"""One-stop runner for vectr-lab experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from plotly.io import to_html

from vectr_lab.backtest.metrics import (
    compute_per_ticker_metrics,
    compute_portfolio_metrics,
)
from vectr_lab.backtest.plots import (
    contribution_bar,
    drawdown_figure,
    equity_curve_figure,
)
from vectr_lab.backtest.runner import run_backtest
from vectr_lab.config.loader import load_config, load_default
from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.data.ingest import download_universe
from vectr_lab.reports import render_report
from vectr_lab.strategies.breakout import BreakoutStrategy
from vectr_lab.strategies.ma_cross import MovingAverageCrossStrategy
from vectr_lab.utils.log import configure_logging
from vectr_lab.utils.time import now_utc

_STRATEGIES = {
    MovingAverageCrossStrategy.name: MovingAverageCrossStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
}


def _resolve_config(path: Optional[Path], model):
    if path is None:
        name = model.__name__.replace("Config", "").lower()
        return load_default(name, model)
    return load_config(path, model)


def _instantiate_strategy(config: StrategyConfig):
    cls = _STRATEGIES.get(config.name)
    if cls is None:
        raise ValueError(f"Strategy '{config.name}' is not registered")
    return cls.from_config(config)


def run_pipeline(
    universe_path: Optional[Path],
    strategy_path: Optional[Path],
    risk_path: Optional[Path],
    out_dir: Optional[Path],
    force_download: bool,
) -> Path:
    configure_logging()
    universe = _resolve_config(universe_path, UniverseConfig)
    strategy_config = _resolve_config(strategy_path, StrategyConfig)
    risk_config = _resolve_config(risk_path, RiskConfig)

    market_data = download_universe(universe, force=force_download)
    strategy = _instantiate_strategy(strategy_config)
    result = run_backtest(market_data, strategy, universe, risk_config)

    metrics = compute_portfolio_metrics(result)
    per_ticker = compute_per_ticker_metrics(result)

    timestamp = now_utc().strftime("%Y%m%d-%H%M%S")
    output_path = out_dir or Path("artifacts/runs") / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    equity = result.equity_curve
    trades = result.trades
    equity.to_frame("equity").to_parquet(output_path / "equity.parquet")
    trades.to_parquet(output_path / "trades.parquet")
    per_ticker.to_parquet(output_path / "per_ticker.parquet")
    with (output_path / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    equity_fig = equity_curve_figure(equity)
    drawdown_fig = drawdown_figure(equity)
    contribution_fig = contribution_bar(per_ticker)

    context = {
        "generated_at": now_utc().isoformat(),
        "metrics": metrics,
        "per_ticker": per_ticker.reset_index().to_dict(orient="records"),
        "equity_html": to_html(equity_fig, full_html=False, include_plotlyjs="cdn"),
        "drawdown_html": to_html(drawdown_fig, full_html=False, include_plotlyjs=False),
        "contribution_html": to_html(contribution_fig, full_html=False, include_plotlyjs=False),
    }
    render_report(context, output_path / "report.html")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vectr-lab end-to-end experiment")
    parser.add_argument("--universe", type=Path, default=None, help="Path to universe YAML")
    parser.add_argument("--strategy", type=Path, default=None, help="Path to strategy YAML")
    parser.add_argument("--risk", type=Path, default=None, help="Path to risk YAML")
    parser.add_argument("--out", type=Path, default=None, help="Output directory root")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cache and redownload market data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_pipeline(
        universe_path=args.universe,
        strategy_path=args.strategy,
        risk_path=args.risk,
        out_dir=args.out,
        force_download=args.force_download,
    )
    print(f"Artifacts written to {out_path}")


if __name__ == "__main__":
    main()
