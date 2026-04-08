"""CLI entrypoint for the factor research project."""

from __future__ import annotations

import argparse
import logging

from factor_risk_model.src.pipeline import generate_report, run_factor_research


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="factor_risk_model",
        description="Cross-sectional equity factor research pipeline",
    )
    parser.add_argument("--symbols", nargs="+", help="Override the default universe")
    parser.add_argument("--start-date", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end-date", help="Backtest end date")
    parser.add_argument("--output", default="factor_risk_model/output", help="Output directory")
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=63,
        help="Trading days between rebalances",
    )
    parser.add_argument(
        "--selection-quantile",
        type=float,
        default=0.2,
        help="Top/bottom bucket size for long-short construction",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="One-way transaction cost in basis points",
    )
    parser.add_argument(
        "--backtest-universe",
        choices=["default", "sp500"],
        default="default",
        help="Select the historical backtest universe",
    )
    parser.add_argument(
        "--backtest-limit",
        type=int,
        help="Optional cap on backtest universe size",
    )
    parser.add_argument(
        "--screening-symbols",
        nargs="+",
        help="Override the default screening universe",
    )
    parser.add_argument(
        "--screening-universe",
        choices=["default", "sp500"],
        default="default",
        help="Select the screening universe",
    )
    parser.add_argument(
        "--screening-limit",
        type=int,
        help="Optional cap on screening universe size",
    )
    parser.add_argument(
        "--thesis-top-n",
        type=int,
        default=5,
        help="Number of top ideas to include in the thesis report",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_factor_research(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        rebalance_frequency=args.rebalance_frequency,
        selection_quantile=args.selection_quantile,
        transaction_cost_bps=args.transaction_cost_bps,
        backtest_universe=args.backtest_universe,
        backtest_limit=args.backtest_limit,
        screening_symbols=args.screening_symbols,
        screening_universe=args.screening_universe,
        screening_limit=args.screening_limit,
        thesis_top_n=args.thesis_top_n,
    )
    print(generate_report(result))


if __name__ == "__main__":
    main()
