"""vectr_lab package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vectr-lab")
except PackageNotFoundError:  # pragma: no cover - during local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]
