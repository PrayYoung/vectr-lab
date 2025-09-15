"""Timezone and session helpers."""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Iterable

import pandas as pd


def now_utc() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc)


def within_session(index: pd.DatetimeIndex, session: str | None) -> pd.DatetimeIndex:
    """Filter an index to a trading session expressed as HHMM-HHMM."""

    if not session:
        return index
    start_str, end_str = session.split("-")
    start = time(int(start_str[:2]), int(start_str[2:]))
    end = time(int(end_str[:2]), int(end_str[2:]))
    mask = (index.time >= start) & (index.time <= end)
    return index[mask]


__all__ = ["now_utc", "within_session"]
