"""
Fetch crypto OHLCV data from Yahoo Finance and compute returns/volatility.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """Fetches crypto price data via yfinance and computes basic features."""

    TICKER_MAP = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD",
    }

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self.symbols: List[str] = symbols or ["BTC-USD", "ETH-USD"]
        if not all(isinstance(s, str) and s for s in self.symbols):
            raise ValueError("All symbols must be non-empty strings")
        logger.info(f"CryptoDataCollector initialized for {self.symbols}")

    def _resolve_ticker(self, symbol: str) -> str:
        return self.TICKER_MAP.get(symbol, symbol)

    def fetch_ohlcv(self, symbol: str, start: Optional[str] = None,
                    end: Optional[str] = None, period: str = "1y") -> pd.DataFrame:
        """Fetch daily OHLCV for one symbol."""
        import yfinance as yf

        ticker = self._resolve_ticker(symbol)
        logger.info(f"Fetching {ticker} (start={start}, end={end}, period={period})")

        if start:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
        else:
            df = yf.download(ticker, period=period,
                             progress=False, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data for {ticker}")
            return df

        # flatten multi-level columns yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        logger.info(f"Got {len(df)} rows for {ticker}")
        return df

    def fetch_historical_data(self, days_back: int = 730) -> Dict[str, pd.DataFrame]:
        if days_back <= 0:
            raise ValueError("days_back must be positive")
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        data: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, start=start)
            if not df.empty:
                data[symbol] = df
        return data

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.Series:
        """Log returns from Close prices."""
        col = "Close" if "Close" in df.columns else "close"
        if col not in df.columns:
            raise KeyError("DataFrame must have a 'Close' column")
        prices = df[col]
        if (prices <= 0).any():
            raise ValueError("Prices must be positive for log returns")
        return np.log(prices / prices.shift(1))

    @staticmethod
    def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
        """Annualised rolling volatility (sqrt(365) for crypto markets)."""
        if window <= 1:
            raise ValueError("window must be > 1")
        return returns.rolling(window=window).std() * np.sqrt(365)

    def get_market_data(self, days_back: int = 730) -> Dict[str, Dict[str, pd.Series]]:
        raw = self.fetch_historical_data(days_back)
        out: Dict[str, Dict[str, pd.Series]] = {}
        for symbol, df in raw.items():
            returns = self.calculate_returns(df)
            vol = self.calculate_volatility(returns)
            col = "Close" if "Close" in df.columns else "close"
            out[symbol] = {
                "prices": df[col],
                "returns": returns,
                "volatility": vol,
                "volume": df.get("Volume", pd.Series()),
            }
        return out


def save_data_to_csv(data: Dict[str, pd.DataFrame], output_dir: str = "data") -> None:
    import os
    os.makedirs(output_dir, exist_ok=True)
    for symbol, df in data.items():
        if isinstance(df, pd.DataFrame) and df.empty:
            continue
        fname = f"{symbol.replace('/', '_').replace('-', '_')}_data.csv"
        path = os.path.join(output_dir, fname)
        df.to_csv(path)
        logger.info(f"Saved {symbol} -> {path}")
