# Crypto Market Analytics Platform

Volatility forecasting and risk monitoring for cryptocurrency markets, built with GARCH modeling, walk-forward backtesting, and interactive dashboards.

**[Live Demo](https://quant-finance.streamlit.app)**

## What it does

Fetches live BTC/ETH market data, fits GARCH(1,1) volatility models, and generates forward-looking forecasts -- validated against naive baselines with statistical significance testing.

- **Market Overview** -- multi-asset price history, returns, drawdown, skewness
- **Volatility Analysis** -- GARCH fitting, conditional vol vs realised, n-day forecasts, model comparison (AIC/BIC)
- **Risk Monitoring** -- VaR, CVaR (Expected Shortfall), volatility regime detection, alerting
- **Model Validation** -- walk-forward backtesting, comparison to Historical Vol / EWMA / Random Walk baselines, Diebold-Mariano test for statistical significance
- **Methodology** -- documented approach, assumptions, and limitations

## Quickstart

```bash
git clone https://github.com/astew24/quant-finance.git && cd quant-finance
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run the dashboard
streamlit run streamlit_app.py

# or run the CLI pipeline
python -m crypto_volatility --symbols BTC-USD ETH-USD --days 730 --horizon 10
```

## Project layout

```
streamlit_app.py                 # interactive dashboard (5 tabs)
crypto_volatility/
  __main__.py                    # CLI entrypoint
  config.py                     # pipeline defaults
  src/
    data_collector.py            # yfinance data + returns/vol
    garch_model.py               # GARCH fitting, forecasting, rolling window, comparison
    backtesting.py               # walk-forward eval, baselines, Diebold-Mariano test
    risk_utils.py                # VaR, CVaR, regime detection
    lstm_model.py                # LSTM model (optional, needs TensorFlow)
    pipeline.py                  # end-to-end orchestration
    visualize.py                 # static matplotlib charts
    metrics.py                   # RMSE
    utils.py                     # data cleaning
  tests/                         # 32 unit tests
factor_risk_model/               # separate equity factor model project
```

## Tests

```bash
python -m pytest crypto_volatility/tests/ -v   # 32 tests
python -m pytest factor_risk_model/tests/ -v    # 5 tests
```

## Deployment

Deployed on Streamlit Cloud. To deploy your own fork:

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point at your fork, branch `main`, file `streamlit_app.py`
4. Deploy

## Tech stack

Python, arch (GARCH), yfinance, SciPy, scikit-learn, Plotly, Streamlit, NumPy/Pandas

---

CI runs on Python 3.10-3.12 via GitHub Actions. License: MIT.
