# Equity Factor Screening Pipeline

Implementation: [`/factor_risk_model`](/Users/andrewstewart/quant-finance/quant-finance-1/factor_risk_model)

## Problem Statement

Recruiters see a lot of generic “factor investing” repos that stop at ranking a few tickers or running a single regression. This project is built to look more like actual buy-side research:

1. screen names on value, momentum, and quality
2. estimate forward outperformance probability with scikit-learn
3. link the screen to a long-short research engine with turnover, drawdown, IC, and factor attribution

## Methodology

### Live Screener

- Universe: 24 liquid large-cap U.S. equities
- Data: Yahoo Finance prices plus a current fundamental snapshot
- Value signals:
  - trailing P/E
  - forward P/E
  - price-to-book
  - enterprise-to-EBITDA
- Momentum signals:
  - 12-1 momentum
  - 6-1 momentum
  - trailing 1-month return
- Quality signals:
  - return on equity
  - return on assets
  - profit margin
  - operating margin
  - current ratio
  - debt-to-equity
- Rank names cross-sectionally and blend the factor score with a scikit-learn logistic classifier trained to predict next-month cross-sectional outperformance

### Research Backtest

- Separate 12-name liquid universe for reproducible factor testing
- Signals:
  - `momentum_12_1`
  - `short_term_reversal`
  - `low_volatility`
- Long top `20%`, short bottom `20%`
- Rebalance every `63` trading days
- Apply `10 bps` transaction costs

## Results

### Screener

Holdout window for the classifier: **November 4, 2024 to February 9, 2026**

| Metric | Value |
| --- | --- |
| Holdout accuracy | `54.95%` |
| Holdout ROC AUC | `0.537` |
| Holdout precision | `51.3%` |
| Holdout recall | `44.1%` |

Latest top-ranked names from the live screen:

1. `NVDA`
2. `GOOGL`
3. `CAT`
4. `COP`
5. `GS`

### Backtest

Evaluation window: **January 3, 2020 to April 7, 2026**

| Metric | Value |
| --- | --- |
| Total return | `67.9%` |
| Annualized return | `10.9%` |
| Annualized volatility | `17.5%` |
| Sharpe ratio | `0.62` |
| Max drawdown | `-38.2%` |
| Mean information coefficient | `0.129` |
| Market beta | `-0.041` |

## Tools Used

- Python
- `pandas`, `numpy`, `scipy`
- `yfinance`
- `scikit-learn`

## Why It Matters for Finance

- Puts a current screener and a research backtest in the same project instead of treating them as unrelated exercises
- Shows awareness of both alpha generation and evaluation discipline
- Gives you a clean, human-sounding resume story: factor ranking, ML prediction, and backtest attribution in one codebase

## Notes

- The live screener uses the latest Yahoo Finance fundamental snapshot for value and quality inputs
- The historical classifier is trained on rolling price and volatility features, which keeps the prediction workflow reproducible
