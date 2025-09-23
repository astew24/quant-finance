# Quant Finance Projects

This repository contains two comprehensive quant finance projects built with Python and Jupyter notebooks.

## 🧠 Project 1: Crypto Volatility Forecasting Platform

A predictive system that uses both GARCH and LSTM models to forecast short-term volatility of BTC and ETH.

### Features:
- Historical volatility calculation and GARCH(1,1) modeling
- LSTM-based volatility prediction using TensorFlow/Keras
- Model comparison (GARCH vs LSTM) with MSE and directionality metrics
- Interactive visualization dashboard
- VaR calculator with volatility threshold alerts

### Key Components:
- `crypto_volatility/notebooks/01_data_collection.ipynb` - Data fetching from Binance
- `crypto_volatility/notebooks/02_garch_modeling.ipynb` - GARCH model implementation
- `crypto_volatility/notebooks/03_lstm_forecasting.ipynb` - LSTM model training
- `crypto_volatility/notebooks/04_model_comparison.ipynb` - Model evaluation
- `crypto_volatility/notebooks/05_risk_management.ipynb` - VaR and alerts

---

## 📈 Project 2: Factor-Based Risk Model for Equities

A multi-factor risk model based on Fama-French + momentum signals with risk decomposition analysis.

### Features:
- Multi-factor model construction (Market, SMB, HML, Momentum)
- OLS and Ridge regression for factor exposure estimation
- Time-varying factor exposure analysis
- Portfolio-level risk attribution
- Interactive factor exposure charts
- Sharpe ratio optimization

### Key Components:
- `factor_risk_model/notebooks/01_data_collection.ipynb` - Stock data fetching
- `factor_risk_model/notebooks/02_factor_construction.ipynb` - Factor building
- `factor_risk_model/notebooks/03_factor_regression.ipynb` - Regression analysis
- `factor_risk_model/notebooks/04_risk_attribution.ipynb` - Risk decomposition
- `factor_risk_model/notebooks/05_portfolio_optimization.ipynb` - Sharpe optimization

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Projects

1. **Crypto Volatility Project:**
   ```bash
   cd crypto_volatility
   jupyter notebook notebooks/
   ```

2. **Factor Risk Model Project:**
   ```bash
   cd factor_risk_model
   jupyter notebook notebooks/
   ```

### Project Structure
```
quant-finance/
├── crypto_volatility/
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── plots/
│   └── requirements.txt
├── factor_risk_model/
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── models/
│   ├── plots/
│   └── requirements.txt
└── README.md
```

## 📊 Key Features

- **Modular Design**: Clean separation of data, models, and visualization
- **Reproducible Research**: Jupyter notebooks with detailed markdown explanations
- **Best Practices**: Proper error handling, logging, and documentation
- **Extensible**: Easy to add new models, factors, or cryptocurrencies
- **Interactive**: Plotly dashboards for dynamic exploration

## 🔧 Dependencies

See individual `requirements.txt` files in each project directory for specific dependencies.

## 📝 License

MIT License - feel free to use and modify for your own research. 

---

## 🧪 Testing

To run the unit tests for each project, use:

```bash
cd crypto_volatility
python -m unittest discover tests
```

or

```bash
cd factor_risk_model
python -m unittest discover tests
``` 

---

Last updated: 2025-09-23