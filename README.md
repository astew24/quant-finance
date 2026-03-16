# Quant Finance

Two quantitative finance projects: crypto volatility forecasting and equity factor risk modeling.

## Crypto Volatility Forecasting

GARCH(1,1) volatility forecasting for BTC and ETH with a CLI pipeline that fetches data, fits models, generates forecasts, and saves reports.

### Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r crypto_volatility/requirements.txt
```

### Usage

```bash
# run the full pipeline (BTC + ETH, 2yr history, 10-day forecast)
python -m crypto_volatility

# customize
python -m crypto_volatility --symbols BTC-USD --days 365 --horizon 20
python -m crypto_volatility --no-plots --output results
python -m crypto_volatility -v  # verbose
```

### What the pipeline does

1. Fetches daily OHLCV from Yahoo Finance (no API key needed)
2. Computes log returns and 30-day rolling realised volatility
3. Fits a GARCH(1,1) to model volatility clustering
4. Generates n-day ahead volatility forecasts
5. Evaluates in-sample fit (RMSE, correlation)
6. Computes VaR at 95% and 99%
7. Saves CSVs, charts, and a text report to `output/`

### Project layout

```
crypto_volatility/
  __main__.py           # CLI entrypoint
  config.py
  src/
    data_collector.py   # yfinance data + returns/vol
    garch_model.py      # GARCH fitting & forecasting
    lstm_model.py       # LSTM model (optional, needs tensorflow)
    pipeline.py         # end-to-end orchestration
    visualize.py        # matplotlib plots
    metrics.py          # RMSE
    risk_utils.py       # VaR
    utils.py            # data cleaning
  tests/
```

### Tests

```bash
python -m pytest crypto_volatility/tests/ -v
```

---

## Factor Risk Model

Multi-factor model (Market, SMB, HML, Momentum) for equities. OLS/Ridge regression with rolling exposures and risk attribution.

```bash
pip install -r factor_risk_model/requirements.txt
python -m pytest factor_risk_model/tests/ -v
```

---

CI runs on Python 3.10-3.12 via GitHub Actions.

License: MIT
