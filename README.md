# TimesFM SmallCap Forecaster 📈

> **AI-powered small-cap equity price forecasting using Google's TimesFM foundation model.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TimesFM 2.5](https://img.shields.io/badge/TimesFM-2.5-orange.svg)](https://github.com/google-research/timesfm)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

> ⚠️ **DISCLAIMER**: This project is for **educational and research purposes only**.  
> It is **NOT financial advice**. AI models cannot reliably predict future stock prices.  
> Always do your own research and consult a qualified financial advisor before investing.

---

## What is this?

This project wraps [Google's TimesFM](https://github.com/google-research/timesfm) — a pretrained 200M-parameter time-series foundation model — to forecast small-cap stock prices.

**Why small-caps?**
- Small-cap stocks (< $2B market cap) are often under-analysed
- Higher signal-to-noise ratio for momentum and mean-reversion patterns
- Micro-caps (< $300M) like GRRR can exhibit strong trending behaviour

**What it does:**
- Downloads price history via `yfinance`
- Computes risk metrics: volatility, Sharpe, drawdown, RSI, momentum
- Feeds closing prices into TimesFM for point + quantile (probabilistic) forecasts
- Generates a forecast chart with 80% confidence bands
- Emits a `BULLISH / BEARISH / NEUTRAL` signal (heuristic, not financial advice)
- Saves Markdown + JSON reports

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Abhi183/timesfm-smallcap-forecaster.git
cd timesfm-smallcap-forecaster

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install TimesFM (PyTorch — works on CPU and CUDA)
pip install timesfm[torch]
```

### 2. Run the CLI

```bash
# Forecast GRRR for the next 30 trading days
python cli.py --ticker GRRR

# Custom horizon + output directory
python cli.py --ticker GRRR --horizon 60 --output reports/

# Compare multiple small-cap tickers
python cli.py --ticker GRRR SOUN NVTS --horizon 30 --compare

# Risk metrics only (no TimesFM model needed)
python cli.py --ticker GRRR --no-model
```

### 3. Run the Streamlit dashboard

```bash
pip install streamlit
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run the GRRR example script

```bash
python examples/grrr_analysis.py
```

---

## CLI Reference

```
python cli.py --help

Arguments:
  --ticker / -t     Ticker symbol(s), e.g. GRRR SOUN NVTS
  --horizon         Forecast horizon in trading days (default: 30)
  --context         TimesFM context length (default: 512)
  --history         Days of price history to download (default: 730)
  --output / -o     Output directory for charts/reports (default: reports/)
  --compare         Generate comparison chart across tickers
  --no-model        Skip TimesFM — show risk metrics only
  --show            Display charts interactively
  --verbose / -v    Verbose logging
```

---

## Output

For each ticker, the tool saves to `reports/<ticker>/`:

| File | Description |
|------|-------------|
| `<ticker>_forecast.png` | Forecast chart with confidence bands |
| `<ticker>_report.md` | Full Markdown report |
| `<ticker>_report.json` | Machine-readable JSON report |

When `--compare` is used: `reports/comparison.png`

---

## How it works

```
yfinance (price history)
        │
        ▼
   Normalise to [0,1]      ← last N closing prices
        │
        ▼
  TimesFM 2.5 (200M)       ← Google's pretrained foundation model
        │
        ▼
   Point forecast           ← most likely price path
   Quantile forecasts       ← 10th / 20th / 50th / 80th / 90th percentile
        │
        ▼
   Denormalise → USD        ← back to price units
        │
        ├── Chart (matplotlib)
        ├── Risk metrics
        ├── Signal heuristic
        └── Reports (MD + JSON)
```

**Important caveats:**
- TimesFM is trained on general time-series data, not specifically on equities
- It has no access to news, fundamentals, SEC filings, or earnings data
- Normalisation to [0,1] means the model sees *relative* price movements, not absolute levels
- The model can extrapolate beyond [0,1] — we clip to prevent extreme artefacts
- Short-horizon forecasts (5–30 days) are generally more reliable than long ones

---

## Supported Tickers

Any US-listed ticker available on Yahoo Finance. The tool works best for:

| Category | Examples |
|----------|---------|
| Micro-cap AI/tech | GRRR, SOUN, BBAI |
| Small-cap EV | NKLA, WKHS |
| Small-cap biotech | AGEN, OCGN |
| Small-cap energy | PSHG, GATO |

---

## Project Structure

```
timesfm-smallcap-forecaster/
├── forecaster/
│   ├── __init__.py       # Package version
│   ├── data.py           # yfinance data fetching + normalisation
│   ├── model.py          # TimesFM wrapper (ForecastResult dataclass)
│   ├── analysis.py       # Risk metrics + signal generation
│   ├── visualize.py      # Matplotlib charts (dark theme)
│   └── report.py         # Markdown + JSON report generation
├── cli.py                # CLI entry point
├── app.py                # Streamlit dashboard
├── examples/
│   └── grrr_analysis.py  # Full GRRR example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Based on

- **[Google TimesFM](https://github.com/google-research/timesfm)** — Ansari et al., ICML 2024  
  *"A decoder-only foundation model for time-series forecasting"*
- **[HuggingFace: google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch)**
- **[yfinance](https://github.com/ranaroussi/yfinance)** for market data

---

## License

Apache 2.0 — same as the upstream TimesFM project.

---

*Built by [Abhi183](https://github.com/Abhi183) using [Google TimesFM](https://github.com/google-research/timesfm).*
