"""Market data and fundamentals utilities for factor research."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "GS",
    "XOM",
    "CVX",
    "UNH",
    "HD",
]

SCREENING_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "NFLX",
    "JPM",
    "GS",
    "MS",
    "BAC",
    "XOM",
    "CVX",
    "COP",
    "UNH",
    "JNJ",
    "LLY",
    "HD",
    "COST",
    "WMT",
    "CAT",
    "DE",
    "GE",
]

FUNDAMENTAL_NUMERIC_FIELDS = [
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToEbitda",
    "returnOnEquity",
    "returnOnAssets",
    "profitMargins",
    "operatingMargins",
    "currentRatio",
    "debtToEquity",
]

FUNDAMENTAL_TEXT_FIELDS = [
    "shortName",
    "sector",
]


@dataclass
class StockDataCollector:
    """Download an equity price panel for factor research."""

    symbols: Optional[List[str]] = None
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None

    def __post_init__(self) -> None:
        self.symbols = self.symbols or DEFAULT_UNIVERSE
        self.end_date = self.end_date or datetime.now().strftime("%Y-%m-%d")
        if not self.symbols or not all(isinstance(symbol, str) and symbol for symbol in self.symbols):
            raise ValueError("symbols must be a non-empty list of tickers")

    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch adjusted close prices for the configured universe."""

        logger.info(
            "Downloading %d tickers from %s to %s",
            len(self.symbols),
            self.start_date,
            self.end_date,
        )
        data = yf.download(
            tickers=self.symbols,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            price_frame = data["Close"]
        else:
            price_frame = data.rename(columns={"Close": self.symbols[0]})[[self.symbols[0]]]

        price_frame = price_frame.sort_index().dropna(how="all").dropna(axis=1, how="all")
        if price_frame.empty:
            return price_frame

        logger.info(
            "Collected %d rows across %d symbols",
            len(price_frame),
            len(price_frame.columns),
        )
        return price_frame

    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Simple daily returns."""

        if prices.empty:
            raise ValueError("prices must be non-empty")
        return prices.pct_change().dropna(how="all")

    @staticmethod
    def fetch_fundamental_snapshot(symbols: List[str]) -> pd.DataFrame:
        """Fetch a current fundamental snapshot for the screener."""

        rows = []
        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).get_info()
            except Exception as exc:
                logger.warning("Failed to fetch fundamentals for %s: %s", symbol, exc)
                info = {}

            row = {"symbol": symbol}
            for field in FUNDAMENTAL_TEXT_FIELDS:
                row[field] = info.get(field)
            for field in FUNDAMENTAL_NUMERIC_FIELDS:
                row[field] = info.get(field)
            rows.append(row)

        snapshot = pd.DataFrame(rows).set_index("symbol")
        for field in FUNDAMENTAL_NUMERIC_FIELDS:
            snapshot[field] = pd.to_numeric(snapshot[field], errors="coerce")
        return snapshot.sort_index()
