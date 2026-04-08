# Quant Finance Portfolio

This repository is structured as a targeted portfolio for quant finance, data science, and software engineering roles. It now covers three complementary areas:

1. Time-series volatility forecasting and risk engineering in crypto markets
2. Cross-sectional equity factor research with backtesting and attribution
3. Derivatives pricing with analytical and numerical methods

## Portfolio Snapshot

| Project | Focus | Results Snapshot | Why It Matters |
| --- | --- | --- | --- |
| [`crypto_volatility`](./crypto_volatility/README.md) | GARCH volatility forecasting, walk-forward backtesting, VaR/CVaR, volatility-targeted overlays | On trailing data from April 9, 2024 to April 8, 2026, BTC GARCH RMSE was `1.863` vs `2.172` for a random-walk baseline and ETH GARCH RMSE was `3.060` vs `3.673` | Shows time-series modeling, risk analytics, backtesting discipline, and deployable research code |
| [`factor_risk_model`](./factor_risk_model/README.md) | Cross-sectional factor signals, long-short portfolio construction, IC analysis, factor attribution | From January 3, 2020 to April 7, 2026 the strategy delivered `67.9%` total return, `10.9%` annualized return, `0.62` Sharpe, and `0.129` mean IC with near-zero market beta | Shows portfolio construction, alpha research, and explainable factor exposure analysis |
| [`options_pricing`](./options_pricing/README.md) | Black-Scholes, Greeks, implied vol inversion, Monte Carlo, American option tree | For an ATM 1Y call with `S=K=100`, `r=3%`, `sigma=20%`, Black-Scholes priced at `9.4134`, Monte Carlo estimated `9.4158 +/- 0.1241`, and implied vol recovered `20.00%` | Shows derivatives math, calibration, and numerical methods |

## Repository Guide

| Path | Purpose |
| --- | --- |
| `streamlit_app.py` | Interactive dashboard for the crypto volatility project |
| `crypto_volatility/` | Forecasting pipeline, risk analytics, tests, notebooks, and sample outputs |
| `factor_risk_model/` | Cross-sectional factor research engine, outputs, and tests |
| `options_pricing/` | Derivatives pricing toolkit and validation tests |
| `docs/repository_audit.md` | Portfolio audit of the repository and upgrade decisions |
| `docs/resume_bullets.md` | Resume-ready bullets tailored for quant, data, and SWE roles |

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
# Streamlit demo
streamlit run streamlit_app.py

# Crypto volatility research pipeline
python -m crypto_volatility --symbols BTC-USD ETH-USD --days 730 --horizon 10 --output crypto_volatility/output_sample

# Cross-sectional equity factor research
python -m factor_risk_model --output factor_risk_model/output_sample

# Options pricing toolkit
python -m options_pricing
```

Run the full test suite:

```bash
python -m pytest -q
```

## Project Notes

- The crypto project includes a live Streamlit app: [quant-finance.streamlit.app](https://quant-finance.streamlit.app)
- Sample research artifacts are committed under `crypto_volatility/output_sample/` and `factor_risk_model/output_sample/`
- Results in the READMEs use exact historical windows rather than vague phrases like "recently"

## Resume Support

The repository now includes a dedicated resume bullets file:

- [`docs/resume_bullets.md`](./docs/resume_bullets.md)

It contains role-ready bullets for:

- Quant research / quant developer
- Data scientist / ML engineer
- Software engineer
