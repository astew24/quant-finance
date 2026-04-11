"""Closed-form Black-Scholes pricing and Greeks."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from scipy.stats import norm


def _validate_inputs(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")


def _d1_d2(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    _validate_inputs(spot, strike, rate, sigma, maturity)
    numerator = math.log(spot / strike) + (
        rate - dividend_yield + 0.5 * sigma ** 2
    ) * maturity
    denominator = sigma * math.sqrt(maturity)
    d1 = numerator / denominator
    d2 = d1 - sigma * math.sqrt(maturity)
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes price for a European option."""

    d1, d2 = _d1_d2(spot, strike, rate, sigma, maturity, dividend_yield)
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)

    if option_type == "call":
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    if option_type == "put":
        return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """Return Delta, Gamma, Vega, Theta, and Rho."""

    d1, d2 = _d1_d2(spot, strike, rate, sigma, maturity, dividend_yield)
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    pdf = norm.pdf(d1)

    gamma = math.exp(-dividend_yield * maturity) * pdf / (
        spot * sigma * math.sqrt(maturity)
    )
    vega = discounted_spot * pdf * math.sqrt(maturity) / 100.0

    if option_type == "call":
        delta = math.exp(-dividend_yield * maturity) * norm.cdf(d1)
        theta = (
            -discounted_spot * pdf * sigma / (2.0 * math.sqrt(maturity))
            - rate * discounted_strike * norm.cdf(d2)
            + dividend_yield * discounted_spot * norm.cdf(d1)
        ) / 365.0
        rho = discounted_strike * maturity * norm.cdf(d2) / 100.0
    elif option_type == "put":
        delta = math.exp(-dividend_yield * maturity) * (norm.cdf(d1) - 1.0)
        theta = (
            -discounted_spot * pdf * sigma / (2.0 * math.sqrt(maturity))
            + rate * discounted_strike * norm.cdf(-d2)
            - dividend_yield * discounted_spot * norm.cdf(-d1)
        ) / 365.0
        rho = -discounted_strike * maturity * norm.cdf(-d2) / 100.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
