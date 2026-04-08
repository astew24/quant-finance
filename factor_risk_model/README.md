# Cross-Sectional Equity Factor Research

## Problem Statement

The project asks a practical quant research question: can a small universe of liquid U.S. equities support a market-neutral multi-factor strategy built from transparent, price-based signals?

Rather than stopping at factor regression, this project builds the full research loop:

1. Construct signals
2. Form portfolios
3. Apply transaction costs
4. Measure information coefficient and performance
5. Explain returns through factor attribution

## Methodology

### Universe

- 12 liquid U.S. large-cap names across technology, financials, energy, healthcare, and consumer sectors
- Evaluation window: **January 3, 2020 to April 7, 2026**

### Signals

- `momentum_12_1`: 12-month momentum excluding the most recent month
- `short_term_reversal`: negative 21-day return
- `low_volatility`: negative 63-day realized volatility

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

## Tools Used

- Python
- `pandas`, `numpy`, `scipy`
- `yfinance`
- `scikit-learn`

## Why It Matters for Finance

- Demonstrates real cross-sectional factor research rather than a single regression notebook
- Includes a tangible portfolio construction layer with costs and rebalancing
- Produces metrics that recruiters expect to see in quant research projects: Sharpe, drawdown, turnover, IC, and factor exposures

## How To Run

```bash
python -m factor_risk_model --output factor_risk_model/output_sample
```
