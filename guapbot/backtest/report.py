"""
guapbot/backtest/report.py

BacktestReport — text summary and optional Plotly HTML charts.

Text report always works (no extra dependencies).
HTML report requires plotly (pip install plotly) — raises ImportError otherwise.

Charts produced:
    1. Equity curve vs buy-and-hold
    2. Underwater (drawdown) chart
    3. Rolling 30-day Sharpe ratio
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from guapbot.backtest.engine import BacktestResult
from guapbot.utils.logging import get_logger

logger = get_logger(__name__)

_PERIODS_PER_YEAR = 8_760   # 1h bars
_ROLLING_WINDOW   = 720     # 30 days × 24 h


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult, title: str = "GuapBot Backtest") -> None:
    """Print a formatted performance summary to stdout."""
    m = result.metrics
    n_bars = len(result.equity_curve)

    lines = [
        "",
        f"  {'─' * 50}",
        f"  {title}",
        f"  {'─' * 50}",
        f"  Bars simulated:  {n_bars:,}",
        f"  Trades logged:   {int(m.get('n_trades', 0)):,}",
        f"  {'─' * 50}",
        f"  Total return:    {m.get('total_return', 0) * 100:+.2f} %",
        f"  Ann. return:     {m.get('ann_return',   0) * 100:+.2f} %",
        f"  Sharpe ratio:    {m.get('sharpe',       0):+.3f}",
        f"  Sortino ratio:   {m.get('sortino',      0):+.3f}",
        f"  Calmar ratio:    {m.get('calmar',       0):+.3f}",
        f"  Max drawdown:    {m.get('max_drawdown', 0) * 100:.2f} %",
        f"  Win rate:        {m.get('win_rate',     0) * 100:.1f} %",
        f"  Profit factor:   {_fmt_pf(m.get('profit_factor', 0))}",
        f"  {'─' * 50}",
        "",
    ]
    print("\n".join(lines))


def _fmt_pf(pf: float) -> str:
    if math.isinf(pf):
        return "∞"
    return f"{pf:.2f}"


# ---------------------------------------------------------------------------
# HTML report (requires plotly)
# ---------------------------------------------------------------------------

def save_html(
    result: BacktestResult,
    path: str,
    title: str = "GuapBot Backtest",
) -> None:
    """
    Save a standalone HTML report with three Plotly charts.

    Args:
        result: BacktestResult from BacktestEngine.run()
        path:   Output file path (e.g. '/tmp/report.html')
        title:  Report title shown in charts

    Raises:
        ImportError: if plotly is not installed
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for HTML reports. "
            "Install it with: pip install plotly"
        ) from exc

    equity  = result.equity_curve
    idx     = equity.index

    # Buy-and-hold benchmark
    if not result.log_returns.empty and len(result.log_returns) == len(equity):
        bnh_returns = result.log_returns.values
        bnh_equity  = equity.iloc[0] * np.exp(np.cumsum(bnh_returns))
        bnh_series  = pd.Series(bnh_equity, index=idx)
    else:
        bnh_series = pd.Series(equity.iloc[0], index=idx)

    # Drawdown series
    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max * 100  # percent

    # Rolling 30-day Sharpe
    pct_returns = equity.pct_change().fillna(0.0)
    roll_sharpe = (
        pct_returns.rolling(_ROLLING_WINDOW).mean()
        / pct_returns.rolling(_ROLLING_WINDOW).std().replace(0, np.nan)
        * np.sqrt(_PERIODS_PER_YEAR)
    )

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Equity Curve vs Buy-and-Hold",
            "Drawdown (%)",
            "Rolling 30-Day Sharpe",
        ],
        vertical_spacing=0.10,
        row_heights=[0.45, 0.25, 0.30],
    )

    # --- Chart 1: equity ---
    fig.add_trace(
        go.Scatter(x=idx, y=equity.values, name="Strategy", line=dict(color="royalblue", width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=idx, y=bnh_series.values, name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")),
        row=1, col=1,
    )

    # --- Chart 2: drawdown ---
    fig.add_trace(
        go.Scatter(
            x=idx, y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="crimson", width=1),
            fillcolor="rgba(220,20,60,0.15)",
        ),
        row=2, col=1,
    )

    # --- Chart 3: rolling Sharpe ---
    fig.add_trace(
        go.Scatter(x=idx, y=roll_sharpe.values, name="Sharpe (30d)", line=dict(color="seagreen", width=1.5)),
        row=3, col=1,
    )
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.8), row=3, col=1)

    m = result.metrics
    subtitle = (
        f"Sharpe {m.get('sharpe', 0):+.2f} | "
        f"Total {m.get('total_return', 0) * 100:+.1f}% | "
        f"MaxDD {m.get('max_drawdown', 0) * 100:.1f}% | "
        f"Trades {int(m.get('n_trades', 0))}"
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sub>{subtitle}</sub>", x=0.5),
        height=900,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)",    row=2, col=1)
    fig.update_yaxes(title_text="Sharpe",    row=3, col=1)

    fig.write_html(path, include_plotlyjs="cdn")
    logger.info("HTML report saved to %s", path)
    print(f"Report saved → {path}")
