# Crypto Volatility Forecasting and Risk Engineering

## Problem Statement

Crypto markets exhibit volatility clustering, fat tails, and rapid regime shifts. The goal of this project is to forecast short-horizon volatility, evaluate whether the forecast is better than simple baselines, and translate that forecast into position-sizing and risk-monitoring outputs that would be useful in a trading or risk desk workflow.

## Methodology

1. Download daily BTC and ETH market data from Yahoo Finance
2. Compute log returns and 30-day realized volatility
3. Fit `GARCH(1,1)` models to returns scaled in percent for numerical stability
4. Run a walk-forward out-of-sample backtest against:
   - Historical volatility
   - EWMA (`lambda = 0.94`)
   - Random-walk volatility
5. Compare forecast errors with RMSE, MAE, direction accuracy, and Diebold-Mariano significance tests
6. Convert forecast volatility into a volatility-targeted overlay using a `35%` annualized risk target, capped leverage, and `10 bps` turnover costs
7. Produce CSV outputs, a text report, and a Streamlit dashboard

## Results

Evaluation window: **April 9, 2024 to April 8, 2026**

### Forecasting Performance

| Asset | GARCH RMSE | Random Walk RMSE | Improvement vs Random Walk | Direction Accuracy |
| --- | --- | --- | --- | --- |
| BTC-USD | `1.863` | `2.172` | `14.2% lower` | `29.98%` |
| ETH-USD | `3.060` | `3.673` | `16.7% lower` | `32.08%` |

### Risk Overlay

| Asset | Buy-and-Hold Vol | Vol-Managed Vol | Vol Reduction | Avg Leverage | Max Drawdown |
| --- | --- | --- | --- | --- | --- |
| BTC-USD | `45.8%` | `36.1%` | `21.2% lower` | `0.83x` | `-50.2%` |
| ETH-USD | `74.7%` | `36.1%` | `51.6% lower` | `0.48x` | `-41.4%` |

Notes:

- Over this specific 2024-2026 window, the overlay improved risk compression more than return generation
- GARCH materially outperformed the random-walk benchmark on both BTC and ETH
- BTC in-sample realized vs conditional volatility correlation was `0.663`

Sample outputs:

- [`output_sample/summary.csv`](./output_sample/summary.csv)
- [`output_sample/report.txt`](./output_sample/report.txt)

## Tools Used

- Python
- `arch`
- `pandas`, `numpy`, `scipy`
- `yfinance`
- `scikit-learn`
- Plotly and Streamlit

## Why It Matters for Finance

- Mirrors the workflow used in market risk and systematic trading research
- Separates in-sample fit from walk-forward evaluation instead of stopping at a fitted model
- Connects forecasting directly to execution-relevant risk controls like leverage caps, VaR, and Expected Shortfall

## How To Run

```bash
python -m crypto_volatility --symbols BTC-USD ETH-USD --days 730 --horizon 10 --output crypto_volatility/output_sample
```

For the interactive interface:

```bash
streamlit run streamlit_app.py
```
