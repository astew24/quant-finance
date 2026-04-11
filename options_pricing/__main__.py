"""CLI entrypoint for the options pricing toolkit."""

from __future__ import annotations

import argparse

from options_pricing.src.black_scholes import black_scholes_greeks, black_scholes_price
from options_pricing.src.numerical_methods import (
    american_option_binomial,
    implied_volatility,
    monte_carlo_price,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="options_pricing",
        description="Analytical and numerical option pricing toolkit",
    )
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--option-type", choices=["call", "put"], default="call")
    args = parser.parse_args(argv)

    analytic = black_scholes_price(
        args.spot,
        args.strike,
        args.rate,
        args.sigma,
        args.maturity,
        option_type=args.option_type,
    )
    greeks = black_scholes_greeks(
        args.spot,
        args.strike,
        args.rate,
        args.sigma,
        args.maturity,
        option_type=args.option_type,
    )
    implied = implied_volatility(
        analytic,
        args.spot,
        args.strike,
        args.rate,
        args.maturity,
        option_type=args.option_type,
    )
    mc = monte_carlo_price(
        args.spot,
        args.strike,
        args.rate,
        args.sigma,
        args.maturity,
        option_type=args.option_type,
        n_paths=50_000,
        seed=42,
    )
    american = american_option_binomial(
        args.spot,
        args.strike,
        args.rate,
        args.sigma,
        args.maturity,
        option_type=args.option_type,
        n_steps=200,
    )

    print("Options Pricing Toolkit")
    print("=" * 60)
    print(f"Black-Scholes price : {analytic:.4f}")
    print(f"Implied volatility  : {implied:.4%}")
    print(f"Monte Carlo price   : {mc.price:.4f} +/- {1.96 * mc.standard_error:.4f}")
    print(f"American tree price : {american:.4f}")
    print("Greeks:")
    for name, value in greeks.items():
        print(f"  {name:<6} {value: .4f}")


if __name__ == "__main__":
    main()
