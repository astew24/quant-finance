"""
Helpers for loading committed sample artifacts for offline demos.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


OUTPUT_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "output_sample"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_")


def available_sample_symbols() -> List[str]:
    symbols: List[str] = []
    for path in sorted(OUTPUT_SAMPLE_DIR.glob("*_timeseries.csv")):
        stem = path.stem.replace("_timeseries", "")
        symbols.append(stem.replace("_", "-"))
    return symbols


def sample_data_exists(symbol: str) -> bool:
    return (OUTPUT_SAMPLE_DIR / f"{_safe_symbol(symbol)}_timeseries.csv").exists()


def build_normalized_price_index(
    returns: pd.Series, base_price: float = 100.0
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float, name="Close")

    clean_returns = returns.fillna(0.0)
    growth = np.exp(clean_returns.cumsum())
    prices = base_price * growth / growth.iloc[0]
    prices.name = "Close"
    return prices


def load_sample_market_data(symbol: str, days: int | None = None) -> pd.DataFrame:
    path = OUTPUT_SAMPLE_DIR / f"{_safe_symbol(symbol)}_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"No sample timeseries found for {symbol}")

    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df["Close"] = build_normalized_price_index(df["returns"])

    ordered_cols = [
        col
        for col in ["Close", "returns", "realised_vol", "conditional_vol"]
        if col in df.columns
    ]
    sample = df[ordered_cols]
    if days is not None and days > 0:
        sample = sample.tail(days)
    return sample


def load_sample_forecast(symbol: str) -> pd.Series:
    path = OUTPUT_SAMPLE_DIR / f"{_safe_symbol(symbol)}_forecast.csv"
    if not path.exists():
        raise FileNotFoundError(f"No sample forecast found for {symbol}")

    forecast = pd.read_csv(path, index_col=0, parse_dates=True)["forecast_vol"]
    forecast.name = "forecast_vol"
    return forecast.sort_index()
