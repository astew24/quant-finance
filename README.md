# Quant Finance Projects

**[Live Demo →](https://astew24.github.io/quant-finance/)**

Three quant finance projects in one repo: GARCH volatility modeling for crypto, cross-sectional factor research for equities, and a derivatives pricing toolkit. Each has its own package, tests, and sample output artifacts.

## Projects

| Project | What it does | Numbers |
| --- | --- | --- |
| [Crypto Volatility Risk Engine](./projects/crypto-volatility-risk-engine/README.md) | GARCH forecasting, walk-forward validation, VaR/CVaR | BTC GARCH RMSE 1.863 vs 2.172 random-walk baseline |
| [Quantamental Equity Research Platform](./projects/equity-factor-screening-pipeline/README.md) | Value-momentum-quality ranking, long-short factor research, scikit-learn return classification | 67.9% total return, 0.62 Sharpe (Jan 2020–Apr 2026) |
| [Options Pricing Toolkit](./projects/options-pricing-toolkit/README.md) | Black-Scholes, Greeks, implied vol inversion, Monte Carlo, American option trees | IV recovered at 20.00% for ATM 1Y call |

## Quickstart

```bash
git clone https://github.com/astew24/quant-finance.git
cd quant-finance
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
```

```bash
# Run the portfolio dashboard
streamlit run streamlit_app.py

# Run individual pipelines
python -m crypto_volatility --symbols BTC-USD ETH-USD --days 730 --horizon 10
python -m factor_risk_model --output factor_risk_model/output_sample
python -m options_pricing

# Tests
python -m pytest -q
```

The Streamlit dashboard defaults to committed sample artifacts so it works offline — set the sidebar `Data source` to `Demo sample` to skip the live yfinance connection.

## Layout

| Path | |
| --- | --- |
| `crypto_volatility/` | GARCH forecasting package, notebooks, sample outputs |
| `factor_risk_model/` | Equity factor backtesting and screening package |
| `options_pricing/` | Derivatives pricing toolkit |
| `streamlit_app.py` | Portfolio dashboard spanning all three projects |
| `projects/` | Short writeups with methods and run instructions |
