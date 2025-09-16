#!/usr/bin/env python3
"""One-stop runner for vectr-lab experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

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
from vectr_lab.utils.log import add_file_handler, configure_logging, pop_context, push_context
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
    parser.add_argument("--rebuild-cache", action="store_true", help="Normalize and rewrite cached data")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON logs")
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help="Dump per-ticker trace parquet (can be repeated)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = [t.upper() for t in args.trace]
    configure_logging(debug=args.debug or bool(trace), json_logs=args.json_logs, trace=bool(trace))
    run_id = now_utc().strftime("%Y%m%d-%H%M%S-%f")
    output_path = args.out or Path("artifacts/runs") / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "run.log", json_logs=args.json_logs)

    run_token = push_context(run_id=run_id)
    strategy_token = None
    try:
        universe = _resolve_config(args.universe, UniverseConfig)
        strategy_config = _resolve_config(args.strategy, StrategyConfig)
        strategy_token = push_context(strategy=strategy_config.name)
        risk_config = _resolve_config(args.risk, RiskConfig)

        market_data = download_universe(universe, force=args.force_download or args.rebuild_cache)
        strategy = _instantiate_strategy(strategy_config)
        result = run_backtest(
            market_data,
            strategy,
            universe,
            risk_config,
            run_id=run_id,
            trace_tickers=trace,
        )

        metrics = compute_portfolio_metrics(result)
        per_ticker = compute_per_ticker_metrics(result)

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

        if result.trace_data:
            for ticker, trace_df in result.trace_data.items():
                trace_df.to_parquet(output_path / f"trace_{ticker}.parquet")

        print(f"Artifacts written to {output_path} (run_id={run_id})")
    finally:
        if strategy_token is not None:
            pop_context(strategy_token)
        pop_context(run_token)


if __name__ == "__main__":
    main()
