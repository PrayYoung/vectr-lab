"""Risk rule helpers for masking entries."""

from __future__ import annotations

from datetime import date

import pandas as pd

from vectr_lab.config.models import RiskConfig
from vectr_lab.utils.log import get_logger


_LOG = get_logger(__name__)


def apply_risk_rules(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    returns: pd.DataFrame,
    risk_config: RiskConfig,
    return_stats: bool = False,
):
    """Mask entry signals when risk thresholds are violated."""

    masked = entries.copy().astype(bool)
    open_positions = {col: False for col in entries.columns}
    cooldown_remaining = 0
    consecutive_losses = 0
    current_day: date | None = None
    daily_pnl = 0.0
    drawdown_triggered = False

    stats = {
        "total_entries": int(entries.sum().sum()),
        "daily_dd_masked": 0,
        "cooldown_masked": 0,
        "loss_streak_masked": 0,
        "concurrency_masked": 0,
    }

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

        original_row = entries.loc[timestamp]
        if drawdown_triggered:
            masked.loc[timestamp] = False
            stats["daily_dd_masked"] += int(original_row.sum())
            continue

        if cooldown_remaining > 0:
            masked.loc[timestamp] = False
            cooldown_remaining -= 1
            stats["cooldown_masked"] += int(original_row.sum())
            continue

        if daily_pnl <= -risk_config.max_daily_drawdown_pct:
            drawdown_triggered = True
            masked.loc[timestamp] = False
            stats["daily_dd_masked"] += int(original_row.sum())
            continue

        if consecutive_losses >= risk_config.max_consecutive_losses:
            cooldown_remaining = risk_config.cooldown_bars
            masked.loc[timestamp] = False
            consecutive_losses = 0
            stats["loss_streak_masked"] += int(original_row.sum())
            continue

        available_slots = risk_config.max_concurrent_positions - sum(open_positions.values())
        row = entries.loc[timestamp].copy()
        desired = row[row].index.tolist()
        if available_slots <= 0:
            row[:] = False
            stats["concurrency_masked"] += int(original_row.sum())
        elif len(desired) > available_slots:
            for symbol in desired[available_slots:]:
                row[symbol] = False
            stats["concurrency_masked"] += int((original_row & ~row).sum())
        masked.loc[timestamp] = row

        for symbol in masked.columns:
            if masked.loc[timestamp, symbol]:
                open_positions[symbol] = True

    stats["final_entries"] = int(masked.sum().sum())

    if return_stats:
        return masked, stats
    return masked


__all__ = ["apply_risk_rules"]
