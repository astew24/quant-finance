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
    )
    print(generate_report(result))


if __name__ == "__main__":
    main()
