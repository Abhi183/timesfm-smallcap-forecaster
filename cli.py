#!/usr/bin/env python3
"""
TimesFM SmallCap Forecaster CLI
================================
Predict small-cap stock prices using Google's TimesFM foundation model.

Usage examples
--------------
  python cli.py --ticker GRRR
  python cli.py --ticker GRRR --horizon 60 --output reports/
  python cli.py --ticker GRRR SOUN NVTS --horizon 30 --compare
  python cli.py --ticker GRRR --no-model   # risk metrics only (no TimesFM)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Rich console (optional but pretty) ──────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

from forecaster.data import fetch_stock_data, prepare_context
from forecaster.analysis import compute_risk_metrics, generate_signal

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _print(msg: str, style: str = "") -> None:
    if RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def _banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       TimesFM SmallCap Forecaster  •  Powered by Google      ║
║    Predicting small-cap equities with AI foundation models   ║
╚══════════════════════════════════════════════════════════════╝
    """.strip()
    _print(banner, style="bold cyan")
    _print("")


def run_single(
    ticker: str,
    horizon: int,
    context_len: int,
    history_days: int,
    output_dir: Path,
    no_model: bool,
    show_chart: bool,
    verbose: bool,
) -> int:
    """Run a full forecast for one ticker. Returns 0 on success, 1 on error."""
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    _print(f"\n[bold]▶  {ticker}[/bold]" if RICH else f"\n>> {ticker}")

    # ── Data fetch ───────────────────────────────────────────────────────────
    try:
        _print("  Fetching price history …", style="dim")
        stock = fetch_stock_data(ticker, period_days=history_days)
        name = stock.metadata.get("longName", ticker)
        _print(f"  {name}  |  Last: ${stock.last_price:.4f}  |  {stock.market_cap_category}")
    except Exception as exc:
        _print(f"  [red]ERROR fetching {ticker}: {exc}[/red]" if RICH else f"  ERROR: {exc}")
        return 1

    # ── Risk metrics ─────────────────────────────────────────────────────────
    risk = compute_risk_metrics(stock)
    _print("\n  [bold]Risk Metrics[/bold]" if RICH else "\n  Risk Metrics")
    _print(risk.summary())

    if no_model:
        _print("\n  [dim](--no-model: skipping TimesFM inference)[/dim]" if RICH else "  (skipping forecast)")
        return 0

    # ── TimesFM inference ────────────────────────────────────────────────────
    from forecaster.model import TimesFMForecaster
    from forecaster.visualize import plot_forecast
    from forecaster.report import generate_markdown_report, generate_json_report

    try:
        _print(f"\n  Loading TimesFM model …", style="dim")
        forecaster = TimesFMForecaster(max_context=context_len)
        context, s_min, s_max = prepare_context(stock.close, context_len=context_len)
        _print(f"  Running forecast (horizon={horizon} days) …", style="dim")
        result = forecaster.forecast(
            context=context,
            horizon=horizon,
            s_min=s_min,
            s_max=s_max,
            ticker=ticker,
        )
    except ImportError:
        _print(
            "\n  [yellow]TimesFM not installed.[/yellow]\n"
            "  Run: [bold]pip install timesfm[torch][/bold]\n"
            "  Or see README for setup instructions." if RICH else
            "\n  TimesFM not installed. Run: pip install timesfm[torch]"
        )
        return 1
    except Exception as exc:
        _print(f"  [red]Forecast error: {exc}[/red]" if RICH else f"  Forecast error: {exc}")
        logger.exception("Forecast failed for %s", ticker)
        return 1

    # ── Signal ───────────────────────────────────────────────────────────────
    signal = generate_signal(result, risk)
    _print("\n  [bold]Forecast Signal[/bold]" if RICH else "\n  Forecast Signal")
    _print(signal.summary())

    _print(f"\n  Expected return ({horizon}d): [bold]{result.expected_return_pct:+.2f}%[/bold]" if RICH
           else f"\n  Expected return ({horizon}d): {result.expected_return_pct:+.2f}%")
    _print(f"  80% CI: [{result.downside_pct:+.1f}%  →  {result.upside_pct:+.1f}%]")

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = output_dir / ticker.lower()
    ticker_dir.mkdir(exist_ok=True)

    chart_path = ticker_dir / f"{ticker.lower()}_forecast.png"
    plot_forecast(stock, result, history_days=120, output_path=chart_path, show=show_chart)
    _print(f"\n  Chart → {chart_path}")

    md_path = ticker_dir / f"{ticker.lower()}_report.md"
    generate_markdown_report(stock, result, risk, signal, output_path=md_path, chart_path=chart_path)
    _print(f"  Report → {md_path}")

    json_path = ticker_dir / f"{ticker.lower()}_report.json"
    generate_json_report(stock, result, risk, signal, output_path=json_path)
    _print(f"  JSON → {json_path}")

    return 0


def run_comparison(
    tickers: list[str],
    horizon: int,
    context_len: int,
    history_days: int,
    output_dir: Path,
    show_chart: bool,
) -> None:
    """Run forecasts for multiple tickers and emit a comparison chart."""
    from forecaster.model import TimesFMForecaster
    from forecaster.data import prepare_context
    from forecaster.visualize import plot_comparison

    forecaster = TimesFMForecaster(max_context=context_len)
    results = []

    for ticker in tickers:
        _print(f"\n  Forecasting {ticker} …")
        try:
            stock = fetch_stock_data(ticker, period_days=history_days)
            context, s_min, s_max = prepare_context(stock.close, context_len=context_len)
            result = forecaster.forecast(context, horizon, s_min, s_max, ticker=ticker)
            results.append(result)
        except Exception as exc:
            _print(f"  Skipping {ticker}: {exc}")

    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        comp_path = output_dir / "comparison.png"
        plot_comparison(results, output_path=comp_path, show=show_chart)
        _print(f"\n  Comparison chart → {comp_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TimesFM SmallCap Forecaster — AI-powered price forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker", "-t", nargs="+", required=True,
        help="Stock ticker(s) to forecast (e.g. GRRR SOUN NVTS)",
    )
    parser.add_argument(
        "--horizon", type=int, default=30,
        help="Forecast horizon in trading days (default: 30)",
    )
    parser.add_argument(
        "--context", type=int, default=512,
        help="Context length fed to TimesFM (default: 512)",
    )
    parser.add_argument(
        "--history", type=int, default=730,
        help="Days of price history to download (default: 730)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("reports"),
        help="Output directory for charts and reports (default: reports/)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate a side-by-side comparison chart (requires multiple tickers)",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip TimesFM inference — only show risk metrics (useful for quick checks)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display charts interactively (requires display)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    _banner()
    _print("[yellow]⚠  NOT FINANCIAL ADVICE — for research purposes only.[/yellow]\n" if RICH
           else "WARNING: NOT FINANCIAL ADVICE — for research purposes only.\n")

    exit_code = 0
    for ticker in args.ticker:
        code = run_single(
            ticker=ticker.upper(),
            horizon=args.horizon,
            context_len=args.context,
            history_days=args.history,
            output_dir=args.output,
            no_model=args.no_model,
            show_chart=args.show,
            verbose=args.verbose,
        )
        if code != 0:
            exit_code = code

    if args.compare and len(args.ticker) > 1 and not args.no_model:
        _print("\n[bold]▶  Generating comparison chart …[/bold]" if RICH else "\n>> Comparison chart")
        run_comparison(
            tickers=[t.upper() for t in args.ticker],
            horizon=args.horizon,
            context_len=args.context,
            history_days=args.history,
            output_dir=args.output,
            show_chart=args.show,
        )

    _print("\n[green]Done.[/green]\n" if RICH else "\nDone.\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
