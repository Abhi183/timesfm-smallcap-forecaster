"""Charting utilities for forecast output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from forecaster.data import StockData
from forecaster.model import ForecastResult

logger = logging.getLogger(__name__)

# ── Style ───────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
GRID = "#21262d"
TEXT = "#e6edf3"
MUTED = "#8b949e"


def _apply_dark_style(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--")


def plot_forecast(
    stock: StockData,
    result: ForecastResult,
    history_days: int = 120,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path | None:
    """Generate a publication-quality forecast chart.

    Layout:
      Top panel  — price history + forecast with confidence bands
      Bottom panel — volume bars

    Returns the saved file path, or None if not saved.
    """
    df = stock.df.iloc[-history_days:]
    dates_hist = pd.to_datetime(df.index)
    close_hist = df["Close"].values

    # Build future date index (business days only)
    last_date = dates_hist[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=result.horizon)

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=False,
    )
    _apply_dark_style(fig, [ax_price, ax_vol])

    # ── Price history ────────────────────────────────────────────────────────
    ax_price.plot(
        dates_hist, close_hist,
        color=ACCENT, linewidth=1.5, label="Historical Close",
        zorder=3,
    )

    # Divider
    ax_price.axvline(x=last_date, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)

    # ── Forecast bands ───────────────────────────────────────────────────────
    # 80% CI
    ax_price.fill_between(
        future_dates,
        result.lower_80,
        result.upper_80,
        color=ACCENT, alpha=0.15, label="80% Confidence Interval",
        zorder=1,
    )

    # Median
    ax_price.plot(
        future_dates, result.median,
        color=ACCENT, linewidth=1.0, linestyle=":", alpha=0.7, label="Median (50%)",
        zorder=2,
    )

    # Point forecast
    fc_color = GREEN if result.point_forecast[-1] >= result.last_known_price else RED
    ax_price.plot(
        future_dates, result.point_forecast,
        color=fc_color, linewidth=2.0, label="Point Forecast",
        zorder=4,
    )

    # Endpoint annotation
    end_price = result.point_forecast[-1]
    ret_pct = result.expected_return_pct
    ret_label = f"${end_price:.2f} ({ret_pct:+.1f}%)"
    ax_price.annotate(
        ret_label,
        xy=(future_dates[-1], end_price),
        xytext=(10, 0), textcoords="offset points",
        color=fc_color, fontsize=8, fontweight="bold",
        va="center",
    )

    ax_price.set_ylabel("Price (USD)", color=MUTED)
    ax_price.set_title(
        f"{result.ticker}  —  {result.horizon}-Day TimesFM Forecast",
        color=TEXT, fontsize=13, fontweight="bold", pad=12,
    )
    ax_price.legend(
        loc="upper left", fontsize=8,
        facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT,
    )
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # ── Volume bars ──────────────────────────────────────────────────────────
    if "Volume" in df.columns:
        vol = df["Volume"].values
        bar_colors = [
            GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED
            for i in range(len(df))
        ]
        ax_vol.bar(dates_hist, vol / 1e6, color=bar_colors, alpha=0.7, width=0.8)
        ax_vol.set_ylabel("Vol (M)", color=MUTED, fontsize=8)
        ax_vol.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=30, ha="right")
    else:
        ax_vol.set_visible(False)

    # Subtitle with key stats
    upside = result.upside_pct
    downside = result.downside_pct
    subtitle = (
        f"Last: ${result.last_known_price:.2f}  |  "
        f"80% CI: {downside:+.1f}% / {upside:+.1f}%  |  "
        f"Powered by Google TimesFM"
    )
    fig.text(0.5, 0.97, subtitle, ha="center", va="top", color=MUTED, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.08)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Chart saved to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_comparison(
    results: list[ForecastResult],
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path | None:
    """Plot expected returns comparison bar chart across multiple tickers."""
    tickers = [r.ticker for r in results]
    exp_rets = [r.expected_return_pct for r in results]
    upsides = [r.upside_pct for r in results]
    downsides = [r.downside_pct for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(tickers) * 1.8), 5))
    _apply_dark_style(fig, [ax])

    x = np.arange(len(tickers))
    w = 0.25

    colors = [GREEN if v >= 0 else RED for v in exp_rets]
    ax.bar(x - w, downsides, width=w, color=RED, alpha=0.6, label="Downside (10%)")
    ax.bar(x, exp_rets, width=w, color=colors, alpha=0.9, label="Point Forecast")
    ax.bar(x + w, upsides, width=w, color=GREEN, alpha=0.6, label="Upside (90%)")

    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, color=TEXT, fontsize=10)
    ax.set_ylabel("Expected Return %", color=MUTED)
    ax.set_title("TimesFM Forecast Comparison", color=TEXT, fontsize=12, fontweight="bold")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    for i, (exp, up, dn) in enumerate(zip(exp_rets, upsides, downsides)):
        ax.text(i, exp + 0.5, f"{exp:+.1f}%", ha="center", color=TEXT, fontsize=8)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Comparison chart saved to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
