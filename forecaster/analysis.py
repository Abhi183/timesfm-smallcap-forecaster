"""Risk metrics and signal analysis for small-cap equities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecaster.data import StockData
from forecaster.model import ForecastResult


TRADING_DAYS = 252


@dataclass
class RiskMetrics:
    ticker: str
    annualised_volatility: float      # Historical vol (annualised %)
    sharpe_ratio: float               # Annualised Sharpe (risk-free = 4.5%)
    max_drawdown: float               # Maximum peak-to-trough % drawdown
    avg_daily_range_pct: float        # Average daily high-low % range (liquidity proxy)
    beta_to_spy: float | None         # Beta vs SPY (None if unavailable)
    momentum_20d: float               # 20-day price momentum %
    rsi_14: float                     # 14-period RSI
    volume_spike: bool                # True if latest volume > 2x 20-day avg

    def summary(self) -> str:
        lines = [
            f"  Annualised Volatility : {self.annualised_volatility:.1f}%",
            f"  Sharpe Ratio          : {self.sharpe_ratio:.2f}",
            f"  Max Drawdown          : {self.max_drawdown:.1f}%",
            f"  20d Momentum          : {self.momentum_20d:+.1f}%",
            f"  RSI(14)               : {self.rsi_14:.1f}",
            f"  Avg Daily Range       : {self.avg_daily_range_pct:.2f}%",
            f"  Volume Spike          : {'YES ⚠' if self.volume_spike else 'No'}",
        ]
        if self.beta_to_spy is not None:
            lines.append(f"  Beta (vs SPY)         : {self.beta_to_spy:.2f}")
        return "\n".join(lines)


def _rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _max_drawdown(prices: np.ndarray) -> float:
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / (peak + 1e-10)
    return float(drawdown.min() * 100)


def compute_risk_metrics(stock: StockData) -> RiskMetrics:
    """Compute risk and momentum metrics from historical price data."""
    df = stock.df
    close = stock.close
    lr = stock.log_returns

    # Volatility
    ann_vol = float(np.std(lr) * np.sqrt(TRADING_DAYS) * 100)

    # Sharpe (risk-free approx 4.5% / 252)
    rf_daily = 0.045 / TRADING_DAYS
    excess = lr - rf_daily
    sharpe = float(np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(TRADING_DAYS))

    # Max drawdown
    mdd = _max_drawdown(close)

    # Average daily range %
    if "High" in df.columns and "Low" in df.columns:
        adr = float(((df["High"] - df["Low"]) / (df["Close"] + 1e-10)).mean() * 100)
    else:
        adr = 0.0

    # 20-day momentum
    if len(close) >= 21:
        mom = float((close[-1] / close[-21] - 1.0) * 100)
    else:
        mom = 0.0

    # RSI
    rsi = _rsi(close)

    # Volume spike
    if "Volume" in df.columns and len(df) >= 21:
        avg_vol_20 = float(df["Volume"].iloc[-21:-1].mean())
        latest_vol = float(df["Volume"].iloc[-1])
        vol_spike = latest_vol > 2.0 * avg_vol_20
    else:
        vol_spike = False

    # Beta vs SPY — optional, gracefully skip on error
    beta = stock.metadata.get("beta", None)
    if isinstance(beta, float) and np.isnan(beta):
        beta = None

    return RiskMetrics(
        ticker=stock.ticker,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        avg_daily_range_pct=adr,
        beta_to_spy=beta,
        momentum_20d=mom,
        rsi_14=rsi,
        volume_spike=vol_spike,
    )


@dataclass
class ForecastSignal:
    """Synthesised trading signal from forecast + risk metrics."""

    ticker: str
    signal: str          # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: str      # "HIGH", "MEDIUM", "LOW"
    rationale: list[str]

    def summary(self) -> str:
        icon = {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "─"}.get(self.signal, "?")
        lines = [f"  Signal     : {icon} {self.signal}  [{self.confidence} confidence]"]
        for r in self.rationale:
            lines.append(f"  • {r}")
        return "\n".join(lines)


def generate_signal(forecast: ForecastResult, risk: RiskMetrics) -> ForecastSignal:
    """Combine forecast direction + risk to emit a high-level signal.

    This is a *heuristic* signal for informational purposes only.
    It is NOT financial advice.
    """
    rationale: list[str] = []
    bullish_pts = 0
    bearish_pts = 0

    # Expected return direction
    if forecast.expected_return_pct > 5:
        bullish_pts += 2
        rationale.append(f"Model projects +{forecast.expected_return_pct:.1f}% over {forecast.horizon} days")
    elif forecast.expected_return_pct < -5:
        bearish_pts += 2
        rationale.append(f"Model projects {forecast.expected_return_pct:.1f}% over {forecast.horizon} days")
    else:
        rationale.append(f"Model projects modest {forecast.expected_return_pct:+.1f}% over {forecast.horizon} days")

    # Asymmetric upside vs downside
    if forecast.upside_pct > abs(forecast.downside_pct) * 1.5:
        bullish_pts += 1
        rationale.append(f"Asymmetric upside: +{forecast.upside_pct:.1f}% vs {forecast.downside_pct:.1f}% downside (80% CI)")
    elif abs(forecast.downside_pct) > forecast.upside_pct * 1.5:
        bearish_pts += 1
        rationale.append(f"Downside skew: {forecast.downside_pct:.1f}% vs +{forecast.upside_pct:.1f}% (80% CI)")

    # Momentum
    if risk.momentum_20d > 10:
        bullish_pts += 1
        rationale.append(f"Positive 20d momentum: +{risk.momentum_20d:.1f}%")
    elif risk.momentum_20d < -10:
        bearish_pts += 1
        rationale.append(f"Negative 20d momentum: {risk.momentum_20d:.1f}%")

    # RSI
    if risk.rsi_14 < 30:
        bullish_pts += 1
        rationale.append(f"RSI {risk.rsi_14:.0f} — oversold territory")
    elif risk.rsi_14 > 70:
        bearish_pts += 1
        rationale.append(f"RSI {risk.rsi_14:.0f} — overbought territory")

    # Volatility risk flag
    if risk.annualised_volatility > 80:
        rationale.append(f"High volatility ({risk.annualised_volatility:.0f}% ann.) — position-size carefully")

    if risk.volume_spike:
        rationale.append("Volume spike detected — unusual activity")

    # Resolve signal
    if bullish_pts > bearish_pts + 1:
        signal = "BULLISH"
    elif bearish_pts > bullish_pts + 1:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    diff = abs(bullish_pts - bearish_pts)
    confidence = "HIGH" if diff >= 3 else "MEDIUM" if diff >= 2 else "LOW"

    return ForecastSignal(
        ticker=forecast.ticker,
        signal=signal,
        confidence=confidence,
        rationale=rationale,
    )
