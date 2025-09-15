"""Moving average cross strategy with ATR based stops."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from vectr_lab.strategies.base import BaseStrategy


def _atr(data: pd.DataFrame, length: int) -> pd.Series:
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift(1)).abs()
    low_close = (data["low"] - data["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


class MovingAverageCrossStrategy(BaseStrategy):
    """Simple MA cross strategy with ATR-based trailing stop and R-multiple target."""

    name = "ma_cross"

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        ma_fast = int(self.params.get("ma_fast", 10))
        ma_slow = int(self.params.get("ma_slow", 50))
        atr_len = int(self.params.get("atr_len", 14))

        features = pd.DataFrame(index=data.index)
        features["sma_fast"] = data["close"].rolling(window=ma_fast, min_periods=ma_fast).mean()
        features["sma_slow"] = data["close"].rolling(window=ma_slow, min_periods=ma_slow).mean()
        features["atr"] = _atr(data, atr_len)
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        features_cols = {col for col in ["sma_fast", "sma_slow", "atr"] if col in data.columns}
        if not features_cols.issuperset({"sma_fast", "sma_slow", "atr"}):
            raise ValueError("Features missing from dataframe; run compute_features first")

        df = pd.DataFrame(index=data.index)
        fast = data["sma_fast"]
        slow = data["sma_slow"]
        close = data["close"]
        atr = data["atr"]

        cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        df["entry_long"] = cross_up.shift(1).fillna(False)
        exit_cross = cross_down.shift(1).fillna(False)

        trail_mult = float(self.params.get("trail_mult", 3.0))
        r_multiple_tp = float(self.params.get("r_multiple_tp", 2.0))
        time_stop_bars = int(self.params.get("time_stop_bars", 0) or 0)

        df["sl_price"] = close - trail_mult * atr
        df["tp_price"] = close + r_multiple_tp * (close - df["sl_price"])
        df["sl_price"] = df["sl_price"].clip(lower=0).fillna(method="ffill")
        df["tp_price"] = df["tp_price"].fillna(method="ffill")

        if time_stop_bars > 0:
            df["exit_time"] = df["entry_long"].shift(time_stop_bars).fillna(False)
        else:
            df["exit_time"] = False

        df["exit_long"] = exit_cross | df["exit_time"]
        return df[["entry_long", "exit_long", "sl_price", "tp_price"]]


__all__ = ["MovingAverageCrossStrategy"]
