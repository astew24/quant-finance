"""Numerical methods for pricing and calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from options_pricing.src.black_scholes import black_scholes_greeks, black_scholes_price


@dataclass
class MonteCarloResult:
    """Monte Carlo price estimate with sampling error."""

    price: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Solve for the implied volatility using Newton, then bisection fallback."""

    sigma = max(initial_guess, 1e-4)
    for _ in range(max_iterations):
        price = black_scholes_price(
            spot,
            strike,
            rate,
            sigma,
            maturity,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        vega = black_scholes_greeks(
            spot,
            strike,
            rate,
            sigma,
            maturity,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )["vega"] * 100.0
        error = price - market_price
        if abs(error) < tolerance:
            return float(sigma)
        if vega <= 1e-8:
            break
        sigma = max(sigma - error / vega, 1e-6)

    # Newton failed (near-zero vega or divergence) — fall back to bisection
    low, high = 1e-6, 5.0
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        price = black_scholes_price(
            spot,
            strike,
            rate,
            mid,
            maturity,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )
        if abs(price - market_price) < tolerance:
            return float(mid)
        if price > market_price:
            high = mid
        else:
            low = mid
    return float(0.5 * (low + high))


def monte_carlo_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> MonteCarloResult:
    """Monte Carlo estimate for a European option under GBM."""

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    n_draws = n_paths // 2 if antithetic else n_paths
    normals = rng.standard_normal(n_draws)
    if antithetic:
        shocks = np.concatenate([normals, -normals])
    else:
        shocks = normals

    drift = (rate - dividend_yield - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * math.sqrt(maturity) * shocks
    terminal_prices = spot * np.exp(drift + diffusion)

    if option_type == "call":
        payoff = np.maximum(terminal_prices - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - terminal_prices, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-rate * maturity) * payoff
    price = float(discounted_payoff.mean())
    standard_error = float(discounted_payoff.std(ddof=1) / np.sqrt(len(discounted_payoff)))
    confidence_radius = 1.96 * standard_error
    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_low=price - confidence_radius,
        confidence_interval_high=price + confidence_radius,
    )


def american_option_binomial(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    option_type: str = "put",
    dividend_yield: float = 0.0,
    n_steps: int = 200,
) -> float:
    """Cox-Ross-Rubinstein tree for an American option."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = maturity / n_steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1.0 / up
    discount = math.exp(-rate * dt)
    growth = math.exp((rate - dividend_yield) * dt)
    prob = (growth - down) / (up - down)
    if not 0 <= prob <= 1:
        raise ValueError("binomial tree produced an invalid risk-neutral probability")

    spots = np.array(
        [spot * (up ** j) * (down ** (n_steps - j)) for j in range(n_steps + 1)],
        dtype=float,
    )
    if option_type == "call":
        values = np.maximum(spots - strike, 0.0)
    elif option_type == "put":
        values = np.maximum(strike - spots, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    for step in range(n_steps - 1, -1, -1):
        spots = spots[:-1] / down
        continuation = discount * (
            prob * values[1:] + (1.0 - prob) * values[:-1]
        )
        if option_type == "call":
            exercise = np.maximum(spots - strike, 0.0)
        else:
            exercise = np.maximum(strike - spots, 0.0)
        values = np.maximum(continuation, exercise)

    return float(values[0])
