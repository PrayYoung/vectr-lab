"""Strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from vectr_lab.config.models import StrategyConfig


@dataclass
class StrategyOutput:
    """Container for features and signals produced by a strategy."""

    features: pd.DataFrame
    signals: pd.DataFrame


class BaseStrategy(ABC):
    """Common interface for strategy implementations."""

    name: str

    def __init__(self, parameters: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = parameters or {}

    @classmethod
    def from_config(cls, config: StrategyConfig) -> "BaseStrategy":
        """Construct a strategy from configuration."""

        return cls(parameters=config.parameters)

    @abstractmethod
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a feature dataframe derived from raw OHLCV data."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the signal dataframe consumed by the backtest runner."""

    def run(self, data: pd.DataFrame) -> StrategyOutput:
        """Compute features and signals in a single call."""

        features = self.compute_features(data)
        signals = self.generate_signals(features.join(data, how="left"))
        return StrategyOutput(features=features, signals=signals)


__all__ = ["BaseStrategy", "StrategyOutput"]
