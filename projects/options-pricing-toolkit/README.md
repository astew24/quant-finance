# Options Pricing Toolkit

Implementation: [`/options_pricing`](/Users/andrewstewart/quant-finance/quant-finance-1/options_pricing)

## Problem Statement

This project implements a compact derivatives pricing stack covering closed-form valuation, calibration, and numerical pricing methods in one package.

## Methodology

- Black-Scholes pricing for European calls and puts
- Closed-form Greeks: Delta, Gamma, Vega, Theta, and Rho
- Implied volatility inversion with Newton-Raphson and bisection fallback
- Monte Carlo pricing with antithetic variates
- Cox-Ross-Rubinstein binomial tree for American options

## Results

Sample contract:

- `S = 100`
- `K = 100`
- `r = 3%`
- `sigma = 20%`
- `T = 1.0`

| Output | Value |
| --- | --- |
| Black-Scholes price | `9.4134` |
| Monte Carlo estimate | `9.4158 +/- 0.1241` |
| Implied volatility recovery | `20.00%` |
| American tree price | `9.4035` |

## Tools Used

- Python
- `numpy`
- `scipy`

## Why This Project Matters

- Shows closed-form pricing and numerical approximations in the same package
- Demonstrates calibration logic, not just valuation formulas
- Captures the core building blocks used in options research and derivatives tooling
