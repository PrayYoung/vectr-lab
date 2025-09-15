"""Plotly visualisations for backtest outputs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def equity_curve_figure(equity: pd.Series) -> go.Figure:
    fig = px.line(equity, title="Equity Curve")
    fig.update_layout(xaxis_title="Date", yaxis_title="Equity")
    return fig


def drawdown_figure(equity: pd.Series) -> go.Figure:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    fig = px.area(drawdown, title="Drawdown")
    fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown")
    return fig


def param_heatmap(data: pd.DataFrame, x: str, y: str, value: str) -> go.Figure:
    pivot = data.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig = px.imshow(pivot, text_auto=True, title=f"Parameter Heatmap ({value})")
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig


def contribution_bar(data: pd.DataFrame, column: str = "total_return") -> go.Figure:
    fig = px.bar(data.reset_index(), x="ticker", y=column, title="Per-ticker Contribution")
    fig.update_layout(xaxis_title="Ticker", yaxis_title=column.replace("_", " ").title())
    return fig


__all__ = ["equity_curve_figure", "drawdown_figure", "param_heatmap", "contribution_bar"]
