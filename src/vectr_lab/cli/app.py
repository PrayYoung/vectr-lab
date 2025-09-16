"""Typer-based command line interface for vectr_lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer
from plotly.io import to_html
from rich.console import Console

from vectr_lab.backtest.metrics import compute_per_ticker_metrics, compute_portfolio_metrics
from vectr_lab.backtest.plots import contribution_bar, drawdown_figure, equity_curve_figure
from vectr_lab.backtest.runner import run_backtest
from vectr_lab.backtest.scan import run_grid_scan
from vectr_lab.config.loader import load_config, load_default
from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.data.ingest import download_universe
from vectr_lab.reports import render_report
from vectr_lab.strategies.breakout import BreakoutStrategy
from vectr_lab.strategies.ma_cross import MovingAverageCrossStrategy
from vectr_lab.utils.log import add_file_handler, configure_logging, pop_context, push_context
from vectr_lab.utils.time import now_utc

app = typer.Typer(help="vectr-lab research CLI")
data_app = typer.Typer(help="Data operations")
backtest_app = typer.Typer(help="Backtest operations")
scan_app = typer.Typer(help="Parameter scans")
report_app = typer.Typer(help="Reporting")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(scan_app, name="scan")
app.add_typer(report_app, name="report")

console = Console()
_STRATEGIES = {
    MovingAverageCrossStrategy.name: MovingAverageCrossStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
}


def _load_universe(path: Optional[Path]) -> UniverseConfig:
    if path is None:
        return load_default("universe", UniverseConfig)
    return load_config(path, UniverseConfig)


def _load_strategy(path: Optional[Path]) -> StrategyConfig:
    if path is None:
        return load_default("strategy", StrategyConfig)
    return load_config(path, StrategyConfig)


def _load_risk(path: Optional[Path]) -> RiskConfig:
    if path is None:
        return load_default("risk", RiskConfig)
    return load_config(path, RiskConfig)


def _make_strategy(config: StrategyConfig):
    cls = _STRATEGIES.get(config.name)
    if cls is None:
        raise typer.BadParameter(f"Unknown strategy: {config.name}")
    return cls.from_config(config)


@data_app.command("pull")
def data_pull(
    universe_path: Optional[Path] = typer.Option(None, "--universe", help="Universe YAML path"),
    force: bool = typer.Option(False, "--force", help="Force re-download"),
    rebuild_cache: bool = typer.Option(False, "--rebuild-cache", help="Normalize and rewrite cached data"),
) -> None:
    """Download and cache OHLCV data for a universe."""

    configure_logging()
    universe = _load_universe(universe_path)
    console.print(f"Downloading data for {len(universe.tickers)} tickers...")
    download_universe(universe, force=force or rebuild_cache)
    console.print("Done.")


@backtest_app.command("run")
def backtest_run(
    universe_path: Optional[Path] = typer.Option(None, "--universe"),
    strategy_path: Optional[Path] = typer.Option(None, "--strategy"),
    risk_path: Optional[Path] = typer.Option(None, "--risk"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON logs"),
    trace: List[str] = typer.Option([], "--trace", help="Dump per-ticker trace parquet", show_default=False, multiple=True),
    rebuild_cache: bool = typer.Option(False, "--rebuild-cache", help="Normalize and rewrite cached data"),
) -> None:
    """Run a backtest for the given universe/strategy/risk pair."""

    run_id = now_utc().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = out or Path("artifacts/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(debug=debug or bool(trace), json_logs=json_logs, trace=bool(trace))
    add_file_handler(out_dir / "run.log", json_logs=json_logs)

    run_token = push_context(run_id=run_id)
    strategy_token = None
    try:
        universe = _load_universe(universe_path)
        strategy_config = _load_strategy(strategy_path)
        strategy_token = push_context(strategy=strategy_config.name)
        risk_config = _load_risk(risk_path)

        console.print("Preparing data...")
        market_data = download_universe(universe, force=rebuild_cache)

        strategy = _make_strategy(strategy_config)
        result = run_backtest(
            market_data,
            strategy,
            universe,
            risk_config,
            run_id=run_id,
            trace_tickers=[t.upper() for t in trace],
        )

        metrics = compute_portfolio_metrics(result)
        per_ticker = compute_per_ticker_metrics(result)

        console.print(f"Writing artifacts to {out_dir}")
        equity = result.equity_curve
        trades = result.trades
        equity.to_frame("equity").to_parquet(out_dir / "equity.parquet")
        trades.to_parquet(out_dir / "trades.parquet")
        with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        per_ticker.to_parquet(out_dir / "per_ticker.parquet")

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

        render_report(context, out_dir / "report.html")

        if result.trace_data:
            for trace_ticker, trace_df in result.trace_data.items():
                trace_df.to_parquet(out_dir / f"trace_{trace_ticker}.parquet")

        console.print(f"Backtest complete. Run ID: {run_id}")
    finally:
        if strategy_token is not None:
            pop_context(strategy_token)
        pop_context(run_token)


@scan_app.command("grid")
def scan_grid(
    universe_path: Optional[Path] = typer.Option(None, "--universe"),
    strategy_path: Optional[Path] = typer.Option(None, "--strategy"),
    risk_path: Optional[Path] = typer.Option(None, "--risk"),
    param_grid: str = typer.Option(..., help="JSON string mapping parameter to values"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Execute a parameter grid search."""

    configure_logging()
    universe = _load_universe(universe_path)
    strategy_config = _load_strategy(strategy_path)
    risk_config = _load_risk(risk_path)

    grid: Dict[str, list] = json.loads(param_grid)
    console.print(f"Running grid scan over {grid}")
    market_data = download_universe(universe)
    run_grid_scan(market_data, universe, risk_config, strategy_config, grid, output_root=out)
    console.print("Scan complete.")


@report_app.command("export")
def report_export(
    run_path: Path = typer.Argument(..., exists=True, dir_okay=True, file_okay=False),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Regenerate report for an existing run directory."""

    stats_path = run_path / "stats.json"
    equity_path = run_path / "equity.parquet"
    per_ticker_path = run_path / "per_ticker.parquet"

    if not stats_path.exists() or not equity_path.exists():
        raise typer.BadParameter("Run directory missing required artifacts")

    with stats_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    equity = pd.read_parquet(equity_path)["equity"]
    per_ticker = pd.read_parquet(per_ticker_path)

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
    output_path = out or run_path / "report.html"
    render_report(context, output_path)
    console.print(f"Report exported to {output_path}")


if __name__ == "__main__":
    app()
