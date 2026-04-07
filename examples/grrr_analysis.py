"""
Example: Full GRRR forecast analysis.

Run from the project root:
    python examples/grrr_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecaster.data import fetch_stock_data, prepare_context
from forecaster.analysis import compute_risk_metrics, generate_signal
from forecaster.report import generate_markdown_report, generate_json_report


def main():
    print("=" * 60)
    print("  GRRR — Gorilla Technology Group | TimesFM Analysis")
    print("=" * 60)
    print()

    # 1. Fetch data
    print("Fetching GRRR price history (2 years)...")
    stock = fetch_stock_data("GRRR", period_days=730)
    print(f"  Company   : {stock.metadata.get('longName', 'GRRR')}")
    print(f"  Last Close: ${stock.last_price:.4f}")
    print(f"  Category  : {stock.market_cap_category}")
    print(f"  Market Cap: ${stock.metadata.get('marketCap', 0)/1e6:.1f}M")
    print()

    # 2. Risk metrics (no model needed)
    risk = compute_risk_metrics(stock)
    print("Risk Metrics:")
    print(risk.summary())
    print()

    # 3. TimesFM forecast
    try:
        from forecaster.model import TimesFMForecaster
        from forecaster.visualize import plot_forecast

        print("Loading TimesFM 2.5 model (may take a moment on first run)...")
        forecaster = TimesFMForecaster(max_context=512)

        context, s_min, s_max = prepare_context(stock.close, context_len=512)
        print("Running 30-day forecast...")
        result = forecaster.forecast(
            context=context,
            horizon=30,
            s_min=s_min,
            s_max=s_max,
            ticker="GRRR",
        )

        print(f"\nForecast Results (30 days):")
        print(f"  Last price     : ${result.last_known_price:.4f}")
        print(f"  Point forecast : ${result.point_forecast[-1]:.4f}")
        print(f"  Expected return: {result.expected_return_pct:+.2f}%")
        print(f"  80% CI         : [{result.downside_pct:+.1f}%, {result.upside_pct:+.1f}%]")

        # Signal
        signal = generate_signal(result, risk)
        print(f"\nSignal: {signal.signal} ({signal.confidence} confidence)")
        for r in signal.rationale:
            print(f"  • {r}")

        # Save chart
        out = Path("reports/grrr")
        out.mkdir(parents=True, exist_ok=True)
        chart_path = out / "grrr_forecast.png"
        plot_forecast(stock, result, output_path=chart_path)
        print(f"\nChart saved to: {chart_path}")

        # Save reports
        md_path = out / "grrr_report.md"
        generate_markdown_report(stock, result, risk, signal, output_path=md_path, chart_path=chart_path)
        json_path = out / "grrr_report.json"
        generate_json_report(stock, result, risk, signal, output_path=json_path)
        print(f"Report  saved to: {md_path}")
        print(f"JSON    saved to: {json_path}")

    except ImportError:
        print("TimesFM not installed — showing risk metrics only.")
        print("Install with: pip install timesfm[torch]")

        signal = None

    print()
    print("=" * 60)
    print("  DISCLAIMER: NOT FINANCIAL ADVICE.")
    print("  For research & educational use only.")
    print("=" * 60)


if __name__ == "__main__":
    main()
