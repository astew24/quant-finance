# Resume Bullets

## Crypto Volatility Forecasting and Risk Engineering

- Built a Python research pipeline for BTC and ETH volatility forecasting with `GARCH(1,1)`, walk-forward validation, VaR/CVaR, and a Streamlit dashboard for interactive monitoring.
- Benchmarked GARCH forecasts against historical volatility, EWMA, and random-walk baselines on data from April 9, 2024 to April 8, 2026; reduced BTC RMSE from `2.172` to `1.863` and ETH RMSE from `3.673` to `3.060`.
- Implemented a volatility-targeted overlay with capped leverage and turnover costs, cutting realized annualized volatility from `45.8%` to `36.1%` on BTC and from `74.7%` to `36.1%` on ETH over the evaluation window.
- Productionized the project with CLI entrypoints, committed sample outputs, unit tests, and a Streamlit front end that turns raw forecasts into desk-style risk views.

## Equity Factor Screening Pipeline and Alpha Research

- Built an automated equity factor screener across a 24-name large-cap universe, ranking securities on value, momentum, and quality using Yahoo Finance fundamentals and cross-sectional price signals.
- Applied a scikit-learn logistic classifier to predict next-month cross-sectional outperformance, reaching `54.9%` holdout accuracy and `0.537` ROC AUC over the November 4, 2024 to February 9, 2026 holdout window.
- Designed a linked market-neutral factor backtest with quarterly rebalancing and `10 bps` transaction costs; generated `67.9%` total return, `10.9%` annualized return, and `0.62` Sharpe from January 3, 2020 to April 7, 2026.
- Added portfolio diagnostics used in quant workflows, including turnover, max drawdown, information coefficient (`0.129`), and factor attribution with near-zero market beta (`-0.041`).

## Options Pricing Toolkit

- Implemented an options pricing library covering Black-Scholes valuation, Greeks, implied volatility inversion, Monte Carlo pricing, and American option valuation via a Cox-Ross-Rubinstein tree.
- Built a calibration workflow that recovered a `20.00%` implied volatility from a sample ATM 1Y call and matched Black-Scholes pricing with Monte Carlo (`9.4134` vs `9.4158 +/- 0.1241`).
- Added numerical validation tests for put-call parity, implied-vol recovery, Monte Carlo convergence, and American put early-exercise consistency to keep the toolkit trustworthy.
- Structured the project as a reusable Python package with CLI execution, making it present like production-grade quantitative software rather than a one-off notebook.
