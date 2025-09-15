"""Reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_report(context: Dict[str, object], output_path: Path) -> None:
    template = _TEMPLATE_ENV.get_template("summary.md.j2")
    html = template.render(**context)
    output_path.write_text(html, encoding="utf-8")


__all__ = ["render_report"]
