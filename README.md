# Crypto Volatility Forecasting

GARCH-based volatility forecasting for BTC and ETH with an interactive Streamlit dashboard and CLI pipeline.

Fetches real market data, fits conditional volatility models, generates multi-day forecasts, and computes Value-at-Risk -- all from live Yahoo Finance data with no API key required.

**[Live Demo](https://quant-finance.streamlit.app)** (Streamlit Cloud)

## What it does

- Pulls daily OHLCV data for BTC-USD and ETH-USD from Yahoo Finance
- Computes log returns and rolling realised volatility
- Fits a GARCH(1,1) model to capture volatility clustering
- Generates n-day ahead volatility forecasts
- Computes Value-at-Risk at 95% and 99% confidence levels
- Compares multiple GARCH specifications (AIC/BIC)
- Displays interactive Plotly charts: price history, return distribution, realised vs conditional volatility, forecast

## Quickstart

```bash
git clone https://github.com/astew24/quant-finance.git
cd quant-finance
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run streamlit_app.py
```

### Run the CLI pipeline

```bash
python -m crypto_volatility
python -m crypto_volatility --symbols BTC-USD --days 365 --horizon 20
python -m crypto_volatility --no-plots -v
```

### Run tests

```bash
python -m pytest crypto_volatility/tests/ -v
```

## Project layout

```
streamlit_app.py               # interactive dashboard
crypto_volatility/
  __main__.py                  # CLI entrypoint
  config.py                    # pipeline defaults
  src/
    data_collector.py          # yfinance data fetching + returns/vol
    garch_model.py             # GARCH fitting, forecasting, comparison
    lstm_model.py              # LSTM model (optional, needs tensorflow)
    pipeline.py                # end-to-end orchestration
    visualize.py               # static matplotlib charts
    metrics.py                 # RMSE
    risk_utils.py              # VaR
    utils.py                   # data cleaning
  tests/                       # 21 unit tests
factor_risk_model/             # separate equity factor model project
```

## Deployment

Deployed on Streamlit Cloud. To deploy your own:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point it at your fork, branch `main`, file `streamlit_app.py`
4. Deploy -- dependencies install automatically from `requirements.txt`

## Tech stack

Python, arch (GARCH), yfinance, Streamlit, Plotly, scikit-learn, NumPy/Pandas

## Also in this repo

**Factor Risk Model** -- Multi-factor equity model (Market, SMB, HML, Momentum) with OLS/Ridge regression and risk attribution. See `factor_risk_model/`.

---

CI runs on Python 3.10-3.12 via GitHub Actions. License: MIT.
