# Quant Finance Projects

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://quant-finance.streamlit.app)

This repository showcases a small quant portfolio across three distinct workflows: time-series volatility modeling for crypto markets, cross-sectional factor research for equities, and derivatives pricing with both analytical and numerical methods. Together they demonstrate end-to-end quantitative engineering, from raw data and model design through validation, risk measurement, committed sample artifacts, and a stateless Streamlit showcase that an employer can review immediately.

This repository contains three quantitative finance projects. The short project pages live under [`projects/`](./projects/README.md), while the implementation directories remain at the repo root so the Python packages and CLI workflows stay stable.

## Portfolio Snapshot

| Project | Focus | Results Snapshot | Why It Matters |
| --- | --- | --- | --- |
| [`Crypto Volatility Risk Engine`](./projects/crypto-volatility-risk-engine/README.md) | GARCH volatility forecasting, walk-forward validation, VaR/CVaR, and volatility-targeted overlays | On data from April 9, 2024 to April 8, 2026, BTC GARCH RMSE was `1.863` vs `2.172` for a random-walk baseline and ETH GARCH RMSE was `3.060` vs `3.673` | Shows time-series modeling, market risk analytics, and research code that connects forecasts to position sizing |
| [`Quantamental Equity Research Platform`](./projects/equity-factor-screening-pipeline/README.md) | Value-momentum-quality ranking, scikit-learn return classification, quantamental idea generation, and long-short factor research | The long-short strategy delivered `67.9%` total return with `0.62` Sharpe from January 3, 2020 to April 7, 2026; the live screener classifier posted `54.9%` holdout accuracy and `0.537` ROC AUC from November 4, 2024 to February 9, 2026 | Shows cross-sectional research, factor construction, valuation-aware screening, and ML-assisted ranking |
| [`Options Pricing Toolkit`](./projects/options-pricing-toolkit/README.md) | Black-Scholes, Greeks, implied vol inversion, Monte Carlo, and American option trees | For an ATM 1Y call with `S=K=100`, `r=3%`, `sigma=20%`, Black-Scholes priced at `9.4134`, Monte Carlo estimated `9.4158 +/- 0.1241`, and implied vol recovered `20.00%` | Shows derivatives math, calibration, and numerical methods |

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`projects/`](./projects/README.md) | Short project pages with methods, results, and run instructions |
| `crypto_volatility/` | Crypto forecasting package, tests, notebooks, and sample outputs |
| `factor_risk_model/` | Equity factor backtesting and screening package, tests, and outputs |
| `options_pricing/` | Derivatives pricing toolkit and validation tests |
| `streamlit_app.py` | Interactive portfolio dashboard spanning all three projects |

## Quickstart

```bash
git clone https://github.com/astew24/quant-finance.git
cd quant-finance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
```

Run the projects:

```bash
# Crypto volatility dashboard
streamlit run streamlit_app.py

# Crypto volatility research pipeline
python -m crypto_volatility --symbols BTC-USD ETH-USD --days 730 --horizon 10 --output crypto_volatility/output_sample

# Equity factor screening + research pipeline
python -m factor_risk_model --output factor_risk_model/output_sample

# Options pricing toolkit
python -m options_pricing
```

For an offline-safe demo, open the dashboard and set the sidebar `Data source` control to `Demo sample`. That uses the committed BTC and ETH sample artifacts in `crypto_volatility/output_sample/` instead of relying on a live Yahoo Finance connection.

Run all tests:

```bash
python -m pytest -q
```

## Notes

- The live portfolio dashboard is at [quant-finance.streamlit.app](https://quant-finance.streamlit.app)
- Sample research artifacts are committed under `crypto_volatility/output_sample/`, `factor_risk_model/output_sample/`, and `options_pricing/output_sample.txt`
- Exact dates are used throughout the repository so the claims are auditable
- Additional project ideas are listed in [`projects/next-project-ideas.md`](./projects/next-project-ideas.md)
