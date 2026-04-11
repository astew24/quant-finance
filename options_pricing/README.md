# Options Pricing Toolkit

## Problem Statement

This project implements the pricing and calibration primitives that show up repeatedly in derivatives work:

- analytical European option pricing
- Greeks for sensitivity analysis
- implied volatility inversion
- Monte Carlo pricing
- American option valuation with a binomial tree

The goal is to show numerical finance fundamentals in a clean, testable codebase rather than a notebook-only implementation.

## Methodology

### Analytical Layer

- Black-Scholes closed-form pricing for European calls and puts
- Closed-form Greeks: Delta, Gamma, Vega, Theta, and Rho

### Calibration Layer

- Newton-Raphson implied volatility solver
- Bisection fallback when Vega becomes too small

### Numerical Layer

- Monte Carlo pricing under geometric Brownian motion
- Antithetic variates to reduce estimator variance
- Confidence interval reporting from the sampling error
- Cox-Ross-Rubinstein binomial tree for American options

## Results

Sample contract:

- Spot `S = 100`
- Strike `K = 100`
- Rate `r = 3%`
- Volatility `sigma = 20%`
- Maturity `T = 1.0`

### Pricing Snapshot

| Method | Output |
| --- | --- |
| Black-Scholes price | `9.4134` |
| Implied volatility recovery | `20.0000%` |
| Monte Carlo price | `9.4158 +/- 0.1241` |
| American tree price | `9.4035` |

### Greeks

| Greek | Value |
| --- | --- |
| Delta | `0.5987` |
| Gamma | `0.0193` |
| Vega | `0.3867` |
| Theta | `-0.0147` |
| Rho | `0.5046` |

Validation:

- Unit tests verify put-call parity
- Implied vol recovers the input sigma to four decimal places
- Monte Carlo prices converge to Black-Scholes within tolerance
- American puts price above their European counterparts
- Sample CLI output is committed in [`output_sample.txt`](./output_sample.txt)

## Tools Used

- Python
- `numpy`
- `scipy`

## Why It Matters for Finance

- Shows derivatives math fluency, not just data engineering
- Demonstrates both closed-form and numerical methods
- Includes calibration logic, which is the bridge between theory and market data

## How To Run

```bash
python -m options_pricing
```
