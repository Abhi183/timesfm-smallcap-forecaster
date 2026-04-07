"""
Streamlit dashboard for TimesFM SmallCap Forecaster.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TimesFM SmallCap Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 TimesFM SmallCap")
    st.caption("Google TimesFM  •  Stock Prediction")

    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="GRRR",
        placeholder="e.g. GRRR, SOUN, NVTS",
        help="Enter a US stock ticker. Works best for small/micro-caps.",
    ).upper().strip()

    horizon = st.slider("Forecast Horizon (trading days)", 5, 90, 30, step=5)
    history_days = st.slider("History to fetch (days)", 180, 730, 365, step=30)
    context_len = st.select_slider(
        "TimesFM Context Length",
        options=[128, 256, 512, 1024],
        value=512,
    )

    st.divider()
    run_btn = st.button("🚀 Run Forecast", use_container_width=True, type="primary")
    risk_only = st.checkbox("Risk metrics only (no model)", value=False)

    st.divider()
    st.caption(
        "⚠️ **NOT financial advice.** This tool is for educational and research "
        "purposes only. Always consult a qualified financial advisor."
    )

# ── Main area ────────────────────────────────────────────────────────────────
st.title("TimesFM SmallCap Forecaster")
st.markdown(
    "Powered by **[Google TimesFM 2.5](https://github.com/google-research/timesfm)** "
    "— a 200M-parameter time-series foundation model pretrained by Google Research."
)

if not run_btn:
    st.info("👈 Enter a ticker in the sidebar and click **Run Forecast**.")
    st.stop()


# ── Run pipeline ─────────────────────────────────────────────────────────────
from forecaster.data import fetch_stock_data, prepare_context
from forecaster.analysis import compute_risk_metrics, generate_signal

with st.spinner(f"Fetching {ticker_input} price history …"):
    try:
        stock = fetch_stock_data(ticker_input, period_days=history_days)
    except Exception as exc:
        st.error(f"Could not fetch data for **{ticker_input}**: {exc}")
        st.stop()

name = stock.metadata.get("longName", ticker_input)
mc = stock.metadata.get("marketCap")
mc_str = f"${mc / 1e6:.1f}M" if mc else "N/A"

# ── Company overview ──────────────────────────────────────────────────────────
st.subheader(f"{name} ({ticker_input})")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last Close", f"${stock.last_price:.4f}")
c2.metric("Market Cap", mc_str)
c3.metric("Category", stock.market_cap_category)
c4.metric("Sector", stock.metadata.get("sector", "N/A"))
c5.metric(
    "52w Range",
    f"${stock.metadata.get('52wLow', '?')} – ${stock.metadata.get('52wHigh', '?')}",
)

st.divider()

# ── Risk metrics ──────────────────────────────────────────────────────────────
with st.spinner("Computing risk metrics …"):
    risk = compute_risk_metrics(stock)

st.subheader("Risk Metrics")
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("Ann. Volatility", f"{risk.annualised_volatility:.1f}%")
r2.metric("Sharpe Ratio", f"{risk.sharpe_ratio:.2f}")
r3.metric("Max Drawdown", f"{risk.max_drawdown:.1f}%")
r4.metric("20d Momentum", f"{risk.momentum_20d:+.1f}%")
r5.metric("RSI (14)", f"{risk.rsi_14:.1f}")
r6.metric("Vol Spike", "Yes ⚠️" if risk.volume_spike else "No")

if risk_only:
    st.info("Risk-only mode — no TimesFM inference.")
    st.stop()

# ── TimesFM inference ─────────────────────────────────────────────────────────
try:
    from forecaster.model import TimesFMForecaster
except ImportError:
    st.error(
        "TimesFM is not installed. Please run:\n\n"
        "```bash\npip install timesfm[torch]\n```"
    )
    st.stop()

with st.spinner("Loading TimesFM model from HuggingFace (first run may take a minute) …"):
    forecaster = TimesFMForecaster(max_context=context_len)

with st.spinner(f"Running {horizon}-day forecast …"):
    context, s_min, s_max = prepare_context(stock.close, context_len=context_len)
    try:
        result = forecaster.forecast(
            context=context,
            horizon=horizon,
            s_min=s_min,
            s_max=s_max,
            ticker=ticker_input,
        )
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()

signal = generate_signal(result, risk)

# ── Forecast metrics ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Forecast")

f1, f2, f3, f4 = st.columns(4)
f1.metric("Expected Return", f"{result.expected_return_pct:+.2f}%")
f2.metric("Point Forecast (end)", f"${result.point_forecast[-1]:.4f}")
f3.metric("Upside (90th pct)", f"{result.upside_pct:+.1f}%")
f4.metric("Downside (10th pct)", f"{result.downside_pct:+.1f}%")

# Signal badge
sig_color = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "orange"}.get(signal.signal, "gray")
st.markdown(
    f"**Signal:** :{sig_color}[{signal.signal}] — {signal.confidence} confidence"
)
for r in signal.rationale:
    st.markdown(f"- {r}")

# ── Chart ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Price Chart")

import matplotlib
matplotlib.use("Agg")
from forecaster.visualize import plot_forecast

buf = io.BytesIO()
import matplotlib.pyplot as plt

# Generate the chart to a buffer
tmp_path = Path("/tmp/_streamlit_chart.png")
plot_forecast(stock, result, history_days=min(120, history_days), output_path=tmp_path, show=False)
st.image(str(tmp_path), use_column_width=True)

# ── Historical price table ────────────────────────────────────────────────────
with st.expander("Historical Price Data (last 30 days)"):
    display_df = stock.df.tail(30)[["Open", "High", "Low", "Close", "Volume"]].copy()
    display_df.index = pd.to_datetime(display_df.index).strftime("%Y-%m-%d")
    st.dataframe(display_df.style.format({"Open": "${:.4f}", "High": "${:.4f}", "Low": "${:.4f}", "Close": "${:.4f}"}))

# ── Download JSON ─────────────────────────────────────────────────────────────
from forecaster.report import generate_json_report, generate_markdown_report

json_data = generate_json_report(stock, result, risk, signal)
md_report = generate_markdown_report(stock, result, risk, signal)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(json_data, indent=2),
        file_name=f"{ticker_input.lower()}_forecast.json",
        mime="application/json",
    )
with col_dl2:
    st.download_button(
        "⬇️ Download Markdown Report",
        data=md_report,
        file_name=f"{ticker_input.lower()}_report.md",
        mime="text/markdown",
    )

st.divider()
st.caption(
    "⚠️ This tool is for **educational and research purposes only**. "
    "It is NOT financial advice. TimesFM is a general-purpose time-series model — "
    "it has no knowledge of company fundamentals, news, or market microstructure. "
    "Use at your own risk."
)
