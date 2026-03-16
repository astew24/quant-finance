"""
CLI entrypoint: python -m crypto_volatility [OPTIONS]

Examples:
    python -m crypto_volatility
    python -m crypto_volatility --symbols BTC-USD ETH-USD --days 365
    python -m crypto_volatility --symbols BTC-USD --horizon 20 --output results
"""

import argparse
import logging
import sys

from crypto_volatility.config import DEFAULT_DAYS_BACK, DEFAULT_SYMBOLS, FORECAST_HORIZON
from crypto_volatility.src.pipeline import generate_report, run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="crypto_volatility",
        description="Crypto volatility forecasting pipeline (GARCH)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="Crypto tickers (default: BTC-USD ETH-USD)",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS_BACK,
                        help="Days of history to fetch")
    parser.add_argument("--horizon", type=int, default=FORECAST_HORIZON,
                        help="Forecast horizon in days")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*60}")
    print("  Crypto Volatility Forecasting Pipeline")
    print(f"{'='*60}")
    print(f"  Symbols  : {args.symbols}")
    print(f"  History  : {args.days} days")
    print(f"  Horizon  : {args.horizon} days")
    print(f"  Output   : {args.output}/")
    print(f"{'='*60}\n")

    results = run_pipeline(
        symbols=args.symbols,
        days_back=args.days,
        forecast_horizon=args.horizon,
        output_dir=args.output,
    )

    if not results:
        print("No results produced -- check your network connection.")
        sys.exit(1)

    if not args.no_plots:
        from crypto_volatility.src.visualize import plot_results
        plot_results(results, output_dir=args.output)

    report = generate_report(results)
    print(report)

    report_path = f"{args.output}/report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
