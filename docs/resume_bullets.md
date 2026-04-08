# Resume Bullets

## Crypto Volatility Forecasting and Risk Engineering

- Built a Python research pipeline for BTC and ETH volatility forecasting with `GARCH(1,1)`, walk-forward validation, VaR/CVaR, and a Streamlit dashboard for interactive monitoring.
- Benchmarked GARCH forecasts against historical volatility, EWMA, and random-walk baselines on data from April 9, 2024 to April 8, 2026; reduced BTC RMSE from `2.172` to `1.863` and ETH RMSE from `3.673` to `3.060`.
- Implemented a volatility-targeted overlay with capped leverage and turnover costs, cutting realized annualized volatility from `45.8%` to `36.1%` on BTC and from `74.7%` to `36.1%` on ETH over the evaluation window.
- Productionized the project with CLI entrypoints, committed sample outputs, unit tests, and a Streamlit front end that turns raw forecasts into desk-style risk views.

## Cross-Sectional Equity Factor Research

- Rebuilt a lightweight factor-regression repo into a cross-sectional equity research engine that constructs momentum, short-term reversal, and low-volatility signals across a liquid 12-name U.S. equity universe.
- Designed a market-neutral long-short strategy with quarterly rebalancing and `10 bps` transaction costs; generated `67.9%` total return, `10.9%` annualized return, and `0.62` Sharpe from January 3, 2020 to April 7, 2026.
- Added portfolio diagnostics used in quant workflows, including turnover, max drawdown, mean information coefficient (`0.129`), and factor attribution with near-zero market beta (`-0.041`).
- Exposed the research stack through a clean CLI pipeline that writes prices, returns, factor scores, weights, factor returns, and exposure summaries for reproducible analysis.

## Options Pricing Toolkit

- Implemented an options pricing library covering Black-Scholes valuation, Greeks, implied volatility inversion, Monte Carlo pricing, and American option valuation via a Cox-Ross-Rubinstein tree.
- Built a calibration workflow that recovered a `20.00%` implied volatility from a sample ATM 1Y call and matched Black-Scholes pricing with Monte Carlo (`9.4134` vs `9.4158 +/- 0.1241`).
- Added numerical validation tests for put-call parity, implied-vol recovery, Monte Carlo convergence, and American put early-exercise consistency to keep the toolkit trustworthy.
- Structured the project as a reusable Python package with CLI execution, making it present like production-grade quantitative software rather than a one-off notebook.
