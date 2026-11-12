"""
Financial Visualization & Reporting Utilities.
Provides Plotly/Matplotlib charting and tabular ASCII metrics formatting.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def format_currency(val: float) -> str:
    """Formats numeric float as USD currency string."""
    return f"${val:,.2f}"


def format_percent(val: float) -> str:
    """Formats numeric decimal as percentage string."""
    return f"{val * 100:.2f}%"


def format_bps(val: float) -> str:
    """Formats decimal as basis points."""
    return f"{val * 10000:.1f} bps"


def print_metrics_table(metrics, title: str = "Backtest Performance Report") -> str:
    """
    Renders an institutional performance table.
    """
    data = [
        ("Initial Capital", format_currency(metrics.initial_capital)),
        ("Final Capital", format_currency(metrics.final_capital)),
        ("Cumulative Return", format_percent(metrics.total_return)),
        ("CAGR (Annualized)", format_percent(metrics.cagr)),
        ("Annualized Volatility", format_percent(metrics.annualized_volatility)),
        ("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}"),
        ("Sortino Ratio", f"{metrics.sortino_ratio:.3f}"),
        ("Calmar Ratio", f"{metrics.calmar_ratio:.3f}"),
        ("Max Drawdown (MDD)", format_percent(metrics.max_drawdown)),
        ("Max Drawdown Duration", f"{metrics.max_drawdown_duration_days} days"),
        ("Value at Risk (VaR 95%)", format_percent(metrics.var_95)),
        ("Conditional VaR (CVaR 95%)", format_percent(metrics.cvar_95)),
        ("Jensen's Alpha (vs SPY)", format_percent(metrics.alpha)),
        ("Beta (vs SPY)", f"{metrics.beta:.3f}"),
        ("R-Squared (R^2)", f"{metrics.r_squared:.3f}"),
        ("Tracking Error", format_percent(metrics.tracking_error)),
        ("Information Ratio", f"{metrics.information_ratio:.3f}"),
        ("Win Rate (Daily)", format_percent(metrics.win_rate)),
        ("Profit Factor", f"{metrics.profit_factor:.2f}"),
    ]

    border = "=" * 52
    lines = [border, f" {title.center(50)} ", border]
    for metric, val in data:
        lines.append(f"  {metric:<30} : {val:>16}")
    lines.append(border)
    output = "\n".join(lines)
    return output


def create_plotly_efficient_frontier(
    frontier_df: pd.DataFrame,
    max_sharpe_res,
    min_vol_res,
    asset_stats_df: pd.DataFrame,
    risk_free_rate: float = 0.045,
):
    """
    Builds an interactive Plotly Efficient Frontier chart with CAL and asset markers.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    fig = go.Figure()

    # 1. Frontier curve
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Risk (Vol): %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # 2. Capital Allocation Line (CAL)
    max_vol = frontier_df["volatility"].max() * 1.15
    cal_x = np.linspace(0, max_vol, 50)
    cal_y = risk_free_rate + (max_sharpe_res.sharpe_ratio) * cal_x
    fig.add_trace(
        go.Scatter(
            x=cal_x,
            y=cal_y,
            mode="lines",
            name="Capital Allocation Line (CAL)",
            line=dict(color="#2ca02c", dash="dash", width=1.5),
        )
    )

    # 3. Individual Assets
    fig.add_trace(
        go.Scatter(
            x=asset_stats_df["Volatility"],
            y=asset_stats_df["Expected Return"],
            mode="markers+text",
            name="Component Equities",
            text=asset_stats_df["Ticker"],
            textposition="top center",
            marker=dict(size=9, color="#ff7f0e", symbol="circle"),
            hovertemplate="<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    # 4. Tangency Portfolio (Max Sharpe)
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_res.volatility],
            y=[max_sharpe_res.expected_return],
            mode="markers",
            name=f"Max Sharpe ({max_sharpe_res.sharpe_ratio:.2f})",
            marker=dict(size=14, color="#d62728", symbol="star"),
            hovertemplate="<b>Max Sharpe Portfolio</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: "
            + f"{max_sharpe_res.sharpe_ratio:.2f}<extra></extra>",
        )
    )

    # 5. Minimum Volatility Portfolio
    fig.add_trace(
        go.Scatter(
            x=[min_vol_res.volatility],
            y=[min_vol_res.expected_return],
            mode="markers",
            name="Min Volatility",
            marker=dict(size=12, color="#9467bd", symbol="diamond"),
            hovertemplate="<b>Min Volatility Portfolio</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Markowitz Modern Portfolio Theory - Efficient Frontier</b>",
        xaxis_title="Annualized Volatility (Risk σ)",
        yaxis_title="Annualized Expected Return (μ)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
