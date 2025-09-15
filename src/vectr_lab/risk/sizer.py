"""Position sizing helpers."""

from __future__ import annotations

import pandas as pd


def sizes_fixed_risk(prices: pd.DataFrame, sl_prices: pd.DataFrame, risk_pct: float, equity: float = 1.0) -> pd.DataFrame:
    """Compute position sizes so that a stop loss equates to the provided risk percentage."""

    if not 0 < risk_pct < 1:
        raise ValueError("risk_pct must be between 0 and 1")

    if prices.shape != sl_prices.shape:
        raise ValueError("prices and sl_prices must have the same shape")

    risk_per_trade = equity * risk_pct
    distance = prices - sl_prices
    distance = distance.mask(distance <= 0)
    size = (risk_per_trade / distance).where(distance.notna(), other=0.0)
    size = size.fillna(0.0)
    return size


__all__ = ["sizes_fixed_risk"]
