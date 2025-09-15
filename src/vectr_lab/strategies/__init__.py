"""Strategy implementations."""

from .base import BaseStrategy
from .ma_cross import MovingAverageCrossStrategy

__all__ = ["BaseStrategy", "MovingAverageCrossStrategy"]
