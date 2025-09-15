"""Risk rule helpers for masking entries."""

from __future__ import annotations

from datetime import date

import pandas as pd

from vectr_lab.config.models import RiskConfig


def apply_risk_rules(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    returns: pd.DataFrame,
    risk_config: RiskConfig,
) -> pd.DataFrame:
    """Mask entry signals when risk thresholds are violated."""

    masked = entries.copy().astype(bool)
    open_positions = {col: False for col in entries.columns}
    cooldown_remaining = 0
    consecutive_losses = 0
    current_day: date | None = None
    daily_pnl = 0.0
    drawdown_triggered = False

    for timestamp in entries.index:
        day = timestamp.date() if hasattr(timestamp, "date") else timestamp
        if day != current_day:
            current_day = day
            daily_pnl = 0.0
            drawdown_triggered = False

        # close positions if exits fire
        for col in exits.columns:
            if exits.loc[timestamp, col]:
                open_positions[col] = False

        realized = returns.loc[timestamp].where(exits.loc[timestamp]).fillna(0.0)
        pnl_sum = float(realized.sum()) * 100  # convert to percentage points
        if pnl_sum < 0:
            consecutive_losses += 1
        elif pnl_sum > 0:
            consecutive_losses = 0

        daily_pnl += pnl_sum

        if drawdown_triggered:
            masked.loc[timestamp] = False
            continue

        if cooldown_remaining > 0:
            masked.loc[timestamp] = False
            cooldown_remaining -= 1
            continue

        if daily_pnl <= -risk_config.max_daily_drawdown_pct:
            drawdown_triggered = True
            masked.loc[timestamp] = False
            continue

        if consecutive_losses >= risk_config.max_consecutive_losses:
            cooldown_remaining = risk_config.cooldown_bars
            masked.loc[timestamp] = False
            consecutive_losses = 0
            continue

        available_slots = risk_config.max_concurrent_positions - sum(open_positions.values())
        row = entries.loc[timestamp].copy()
        desired = row[row].index.tolist()
        if available_slots <= 0:
            row[:] = False
        elif len(desired) > available_slots:
            for symbol in desired[available_slots:]:
                row[symbol] = False
        masked.loc[timestamp] = row

        for symbol in masked.columns:
            if masked.loc[timestamp, symbol]:
                open_positions[symbol] = True

    return masked


__all__ = ["apply_risk_rules"]
