"""Position sizing helpers."""

from __future__ import annotations

import pandas as pd


def sizes_fixed_risk(
    prices: pd.DataFrame,
    sl_prices: pd.DataFrame,
    risk_pct: float,
    equity: float = 1.0,
    return_diagnostics: bool = False,
):
    """Compute position sizes so that a stop loss equates to the provided risk percentage."""

    if not 0 < risk_pct < 1:
        raise ValueError("risk_pct must be between 0 and 1")

    if prices.shape != sl_prices.shape:
        raise ValueError("prices and sl_prices must have the same shape")

    risk_per_trade = equity * risk_pct
    distance = prices - sl_prices
    invalid = (distance <= 0) | distance.isna()
    size = (risk_per_trade / distance).where(~invalid, other=0.0)
    size = size.fillna(0.0)

    if not return_diagnostics:
        return size

    diagnostics = {}
    for column in prices.columns:
        col_size = size[column]
        col_invalid = invalid[column]
        diagnostics[column] = {
            "non_zero": int((col_size != 0).sum()),
            "zero_invalid": int(col_invalid.sum()),
            "nan_sl": int(sl_prices[column].isna().sum()),
            "nan_price": int(prices[column].isna().sum()),
        }
    return size, diagnostics


__all__ = ["sizes_fixed_risk"]
