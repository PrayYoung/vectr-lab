import pandas as pd

from vectr_lab.backtest.runner import run_backtest
from vectr_lab.config.models import RiskConfig, StrategyConfig, UniverseConfig
from vectr_lab.strategies.ma_cross import MovingAverageCrossStrategy


def _make_data() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2022-01-01", periods=50, freq="D")
    base = pd.Series(range(50), index=index).astype(float)
    data = {}
    for ticker in ["AAA", "BBB"]:
        df = pd.DataFrame({
            "open": base + 0.1,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": 1_000,
        })
        data[ticker] = df
    return data


def test_runner_no_lookahead():
    data = _make_data()
    universe = UniverseConfig(tickers=list(data.keys()), timeframe="1d")
    risk = RiskConfig(risk_pct=0.01)
    strategy_config = StrategyConfig(name="ma_cross", parameters={"ma_fast": 3, "ma_slow": 5, "atr_len": 5})
    strategy = MovingAverageCrossStrategy.from_config(strategy_config)

    result = run_backtest(data, strategy, universe, risk)
    assert not result.entries.iloc[0].any()
    assert result.entries.shape == (50, 2)
    assert result.exits.shape == (50, 2)
