"""Data ingestion and preprocessing via yfinance."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class StockData:
    ticker: str
    df: pd.DataFrame
    close: np.ndarray
    log_returns: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def last_price(self) -> float:
        return float(self.df["Close"].iloc[-1])

    @property
    def market_cap_category(self) -> str:
        mc = self.metadata.get("marketCap", 0)
        if mc < 300_000_000:
            return "Micro-cap"
        if mc < 2_000_000_000:
            return "Small-cap"
        if mc < 10_000_000_000:
            return "Mid-cap"
        return "Large-cap"


def fetch_stock_data(
    ticker: str,
    period_days: int = 365 * 2,
    interval: str = "1d",
) -> StockData:
    """Download historical OHLCV data for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'GRRR').
        period_days: How many calendar days of history to fetch.
        interval: yfinance interval string ('1d', '1h', etc.).

    Returns:
        StockData with cleaned price series.
    """
    end = datetime.today()
    start = end - timedelta(days=period_days)

    logger.info("Fetching %s from %s to %s", ticker, start.date(), end.date())
    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(subset=["Close"], inplace=True)
    df.sort_index(inplace=True)

    close = df["Close"].values.astype(np.float32)
    log_returns = np.diff(np.log(close + 1e-8)).astype(np.float32)

    # Grab fundamental metadata
    try:
        info = yf.Ticker(ticker).info
        metadata = {
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "longName": info.get("longName", ticker),
            "exchange": info.get("exchange"),
            "currency": info.get("currency", "USD"),
            "beta": info.get("beta"),
            "52wHigh": info.get("fiftyTwoWeekHigh"),
            "52wLow": info.get("fiftyTwoWeekLow"),
            "avgVolume": info.get("averageVolume"),
        }
    except Exception:
        metadata = {}

    return StockData(
        ticker=ticker.upper(),
        df=df,
        close=close,
        log_returns=log_returns,
        metadata=metadata,
    )


def prepare_context(
    close: np.ndarray,
    context_len: int = 512,
    use_log: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Prepare a price series as a TimesFM input context.

    Normalises the last `context_len` prices to [0, 1] so the model sees
    relative movements rather than absolute dollar amounts.

    Returns:
        (context_array, scale_min, scale_max) — scale values needed to
        invert the normalisation on the forecast output.
    """
    seq = close[-context_len:]
    if use_log:
        seq = np.log(seq + 1e-8)

    s_min = float(seq.min())
    s_max = float(seq.max())
    if s_max - s_min < 1e-8:
        s_max = s_min + 1.0

    normalised = (seq - s_min) / (s_max - s_min)
    return normalised.astype(np.float32), s_min, s_max


def denormalise(values: np.ndarray, s_min: float, s_max: float, use_log: bool = False) -> np.ndarray:
    """Invert the normalisation applied by prepare_context."""
    out = values * (s_max - s_min) + s_min
    if use_log:
        out = np.exp(out)
    return out
