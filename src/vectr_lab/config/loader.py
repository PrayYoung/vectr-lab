"""YAML configuration loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_DEFAULT_ROOT = Path(__file__).resolve().parent


def _resolve_path(path: Path | str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    candidate = Path.cwd() / raw_path
    if candidate.exists():
        return candidate
    return (_DEFAULT_ROOT / raw_path).resolve()


def load_config(path: Path | str, model: Type[T]) -> T:
    """Load a YAML file into the provided Pydantic model type."""

    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload: Any = yaml.safe_load(handle) or {}

    return model.parse_obj(payload)


def load_default(name: str, model: Type[T]) -> T:
    """Load a default configuration from the package defaults folder."""

    return load_config(Path("defaults") / f"{name}.yaml", model)


__all__ = ["load_config", "load_default"]
