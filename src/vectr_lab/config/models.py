"""Pydantic models for vectr_lab configuration."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, validator


class UniverseConfig(BaseModel):
    """Universe definition used for data downloads and alignment."""

    tickers: List[str] = Field(..., min_items=1, description="Symbol list")
    timeframe: str = Field("1d", description="Bar size string understood by data source")
    start: Optional[datetime] = Field(None, description="Inclusive start timestamp")
    end: Optional[datetime] = Field(None, description="Inclusive end timestamp")
    session: Optional[str] = Field(None, description="Session string, e.g. 0930-1600")
    calendar: Optional[str] = Field(None, description="Trading calendar name")
    fees_pct: float = Field(0.0, ge=0.0, description="Per-trade fee percentage")
    slippage_pct: float = Field(0.0, ge=0.0, description="Slippage percentage per trade")

    @validator("tickers")
    def _tickers_non_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one ticker must be specified")
        return value

    @validator("start", "end", pre=True)
    def _coerce_datetime(cls, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as exc:  # pragma: no cover - invalid formats bubble up
                raise ValueError(f"Invalid datetime string: {value}") from exc
        raise ValueError("Unsupported datetime value")


class StrategyConfig(BaseModel):
    """Strategy parameters shared across implementations."""

    name: str = Field("ma_cross", description="Strategy identifier")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def get_param(self, key: str, default: Any = None) -> Any:
        """Return a single parameter with an optional default."""

        return self.parameters.get(key, default)


class RiskConfig(BaseModel):
    """Account level risk controls."""

    risk_pct: PositiveFloat = Field(..., description="Risk per trade as fraction of equity")
    max_daily_drawdown_pct: PositiveFloat = Field(
        10.0, description="Daily drawdown limit expressed as percent"
    )
    max_concurrent_positions: PositiveInt = Field(5, description="Cap on simultaneous positions")
    max_consecutive_losses: PositiveInt = Field(
        3, description="Loss streak limit before cooldown"
    )
    cooldown_bars: PositiveInt = Field(5, description="Bars to skip entries after rule triggers")


__all__ = ["UniverseConfig", "StrategyConfig", "RiskConfig"]
