import pandas as pd

from vectr_lab.backtest.runner import run_backtest
from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.strategies.ma_cross import MovingAverageCrossStrategy


def _build_synthetic() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2022-01-01", periods=60, freq="H")
    frames = {}
    for offset, ticker in enumerate(["AAA", "BBB", "CCC"]):
        base = pd.Series(range(60), index=index).astype(float) + offset
        frames[ticker] = pd.DataFrame({
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.5,
            "volume": 1_000,
        })
    return frames


def test_smoke_backtest_runs():
    data = _build_synthetic()
    universe = UniverseConfig(tickers=list(data.keys()), timeframe="1h")
    risk = RiskConfig(risk_pct=0.02)
    strategy_config = StrategyConfig(name="ma_cross", parameters={"ma_fast": 5, "ma_slow": 15, "atr_len": 7})
    strategy = MovingAverageCrossStrategy.from_config(strategy_config)

    result = run_backtest(data, strategy, universe, risk)
    equity = result.equity_curve
    assert len(equity) == 60
    assert equity.iloc[-1] >= 0
