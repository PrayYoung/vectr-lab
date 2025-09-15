"""Placeholder breakout strategy for future extension."""

from __future__ import annotations

import pandas as pd

from vectr_lab.strategies.base import BaseStrategy, StrategyOutput


class BreakoutStrategy(BaseStrategy):
    """Simple price breakout placeholder."""

    name = "breakout"

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        features = pd.DataFrame(index=data.index)
        features["rolling_max"] = data["high"].rolling(lookback, min_periods=lookback).max()
        features["rolling_min"] = data["low"].rolling(lookback, min_periods=lookback).min()
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        breakout_up = data["close"] > data["rolling_max"].shift(1)
        exit_down = data["close"] < data["rolling_min"].shift(1)

        df["entry_long"] = breakout_up.shift(1).fillna(False)
        df["exit_long"] = exit_down.shift(1).fillna(False)
        df["sl_price"] = data["rolling_min"].fillna(method="ffill")
        df["tp_price"] = data["rolling_max"].fillna(method="ffill")
        return df


__all__ = ["BreakoutStrategy"]
