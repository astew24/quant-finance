# Crypto Volatility Risk Engine

Implementation: [`/crypto_volatility`](/Users/andrewstewart/quant-finance/quant-finance-1/crypto_volatility)

## Problem Statement

Crypto markets are structurally noisy: volatility clusters, tail events matter, and the difference between a decent forecast and a naive one changes how aggressively a strategy should size risk. This project asks a practical desk question: can short-horizon volatility forecasts improve over simple baselines, and can those forecasts be translated into usable risk controls?

## Methodology

1. Pull BTC and ETH daily market data from Yahoo Finance
2. Compute log returns and 30-day realized volatility
3. Fit `GARCH(1,1)` models on returns scaled in percent
4. Run walk-forward backtests against:
   - historical volatility
   - EWMA (`lambda = 0.94`)
   - random-walk volatility
5. Evaluate with RMSE, MAE, direction accuracy, and Diebold-Mariano tests
6. Feed forecast volatility into a volatility-targeted overlay with leverage caps and turnover costs

## Results

Evaluation window: **April 9, 2024 to April 8, 2026**

| Asset | GARCH RMSE | Random Walk RMSE | Improvement vs Random Walk | Vol-Managed Realized Vol |
| --- | --- | --- | --- | --- |
| BTC-USD | `1.863` | `2.172` | `14.2% lower` | `36.1%` |
| ETH-USD | `3.060` | `3.673` | `16.7% lower` | `36.1%` |

Risk results:

- BTC buy-and-hold annualized volatility was `45.8%`; the volatility-managed overlay reduced it to `36.1%`
- ETH buy-and-hold annualized volatility was `74.7%`; the overlay reduced it to `36.1%`
- BTC realized vs conditional volatility correlation was `0.663`

## Tools Used

- Python
- `pandas`, `numpy`, `scipy`
- `arch`
- `yfinance`
- Plotly and Streamlit

## Why It Matters for Finance

- Connects forecasting research to actual risk sizing
- Uses walk-forward testing instead of relying on in-sample fit
- Looks like systematic risk research, not a generic dashboard toy

## Portfolio Value

This project signals that you can move from raw market data to a forecast, benchmark it properly, and convert it into a decision variable a PM or risk manager would care about.
