"""Parameter scan utilities."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd

from vectr_lab.backtest.metrics import compute_portfolio_metrics
from vectr_lab.backtest.plots import param_heatmap
from vectr_lab.backtest.runner import run_backtest
from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.strategies.ma_cross import MovingAverageCrossStrategy


def _expand_grid(grid: Dict[str, Iterable]) -> Iterator[Dict[str, object]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combination in product(*values):
        yield dict(zip(keys, combination))


def run_grid_scan(
    data: Dict[str, pd.DataFrame],
    universe: UniverseConfig,
    risk: RiskConfig,
    base_config: StrategyConfig,
    param_grid: Dict[str, Iterable],
    output_root: Path | None = None,
) -> pd.DataFrame:
    """Run a grid search across strategy parameters."""

    results: List[Dict[str, object]] = []
    metric_keys: List[str] = []
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = output_root or Path("artifacts/scans") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for params in _expand_grid(param_grid):
        config = StrategyConfig(name=base_config.name, parameters={**base_config.parameters, **params})
        strategy = MovingAverageCrossStrategy.from_config(config)
        bt_result = run_backtest(data, strategy, universe, risk)
        metrics = compute_portfolio_metrics(bt_result)
        if not metric_keys:
            metric_keys = list(metrics.keys())
        results.append({**params, **metrics})

    result_df = pd.DataFrame(results)
    parquet_path = out_dir / "results.parquet"
    result_df.to_parquet(parquet_path)

    if len(param_grid) >= 2:
        first, second = list(param_grid.keys())[:2]
        heatmap = param_heatmap(result_df, x=first, y=second, value="cagr")
        heatmap.write_html(out_dir / "heatmap.html")

    with (out_dir / "grid.json").open("w", encoding="utf-8") as handle:
        json.dump({"timestamp": timestamp, "param_grid": param_grid}, handle, indent=2)

    return result_df


__all__ = ["run_grid_scan"]
