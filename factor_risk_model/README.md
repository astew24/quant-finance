# Equity Factor Screening Pipeline and Cross-Sectional Alpha Research

## Problem Statement

This project combines two workflows that actually belong together in a quant portfolio:

1. a live equity factor screener that ranks names on value, momentum, and quality
2. a research backtest that checks whether related factor signals survive transaction costs and portfolio construction rules

The point is not just to sort stocks. It is to show a full chain from raw data to ranking logic, ML-assisted return prediction, backtesting, and attribution.

## Methodology

### Live Screener Universe

- 24 liquid U.S. large-cap equities
- Price history from Yahoo Finance
- Current fundamental snapshot from Yahoo Finance for value and quality inputs

### Screener Factors

- Value:
  - trailing P/E
  - forward P/E
  - price-to-book
  - enterprise-to-EBITDA
- Momentum:
  - 12-1 momentum
  - 6-1 momentum
  - trailing 1-month return
- Quality:
  - return on equity
  - return on assets
  - profit margin
  - operating margin
  - current ratio
  - debt-to-equity

### Return Prediction

- Built a scikit-learn logistic classifier to predict next-month cross-sectional outperformance
- Features are derived from rolling price and volatility structure:
  - 1M, 3M, 6M, and 12-1 momentum
  - 3M and 6M volatility
  - 6M drawdown
  - market-relative strength
  - short-term reversal

### Research Backtest Universe

- 12 liquid U.S. large-cap names across technology, financials, energy, healthcare, and consumer sectors
- Evaluation window: **January 3, 2020 to April 7, 2026**

### Backtest Signals

- `momentum_12_1`
- `short_term_reversal`
- `low_volatility`

### Portfolio Construction

- Cross-sectional z-scoring each rebalance date
- Composite score weights:
  - `50%` momentum
  - `25%` short-term reversal
  - `25%` low volatility
- Long top `20%`, short bottom `20%`
- Rebalance every `63` trading days
- Apply `10 bps` transaction cost on turnover

### Attribution

- Regress strategy returns on factor-mimicking portfolios plus the equal-weight market benchmark
- Report estimated factor betas and model `R^2`

## Results

### Screener Metrics

Holdout window: **November 4, 2024 to February 9, 2026**

| Metric | Value |
| --- | --- |
| Holdout accuracy | `54.9%` |
| Holdout ROC AUC | `0.537` |
| Holdout precision | `51.3%` |
| Holdout recall | `44.1%` |

Latest top-ranked names:

1. `NVDA`
2. `GOOGL`
3. `CAT`
4. `COP`
5. `GS`

### Strategy Performance

| Metric | Value |
| --- | --- |
| Total return | `67.9%` |
| Annualized return | `10.9%` |
| Annualized volatility | `17.5%` |
| Sharpe ratio | `0.62` |
| Sortino ratio | `0.91` |
| Max drawdown | `-38.2%` |
| Mean information coefficient | `0.129` |

### Factor Exposures

| Factor | Beta |
| --- | --- |
| Momentum 12-1 | `0.727` |
| Short-term reversal | `0.406` |
| Low volatility | `0.175` |
| Market | `-0.041` |

Additional attribution detail:

- Regression `R^2`: `0.691`
- Average turnover per rebalance: `1.25`
- The strategy maintained near-zero market beta while keeping a positive exposure to the intended alpha factors

Sample outputs:

- [`output_sample/summary.csv`](./output_sample/summary.csv)
- [`output_sample/factor_exposures.csv`](./output_sample/factor_exposures.csv)
- [`output_sample/information_coefficient.csv`](./output_sample/information_coefficient.csv)
- [`output_sample/latest_screen.csv`](./output_sample/latest_screen.csv)
- [`output_sample/screening_model_metrics.csv`](./output_sample/screening_model_metrics.csv)
- [`output_sample/screening_model_coefficients.csv`](./output_sample/screening_model_coefficients.csv)

## Tools Used

- Python
- `pandas`, `numpy`, `scipy`
- `yfinance`
- `scikit-learn`

## Why It Matters for Finance

- Combines a live screen and a research backtest in one coherent project
- Produces the metrics recruiters expect to see: Sharpe, drawdown, turnover, IC, factor exposures, and ML holdout metrics
- Reads like a practical quant research workflow rather than a classroom assignment

## How To Run

```bash
python -m factor_risk_model --output factor_risk_model/output_sample
```
