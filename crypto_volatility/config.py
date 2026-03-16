# Configuration for the crypto_volatility pipeline
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD"]
DEFAULT_DAYS_BACK = 730          # ~2 years of history
VOLATILITY_WINDOW = 30           # rolling window for realised vol
GARCH_P = 1
GARCH_Q = 1
FORECAST_HORIZON = 10            # days ahead
ROLLING_WINDOW = 252             # rolling GARCH training window
ANNUALISATION_FACTOR = 365       # crypto trades every day
