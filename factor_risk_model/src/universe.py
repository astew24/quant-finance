"""Universe loaders for equity research projects."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def load_sp500_constituents() -> pd.DataFrame:
    """Load the current S&P 500 constituents table from Wikipedia."""

    response = requests.get(
        SP500_CONSTITUENTS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError("Failed to load S&P 500 constituents table")

    constituents = tables[0].copy()
    constituents["Symbol"] = constituents["Symbol"].astype(str).str.replace(
        ".",
        "-",
        regex=False,
    )
    return constituents.rename(
        columns={
            "Symbol": "symbol",
            "Security": "security",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )


def load_sp500_symbols(limit: int | None = None) -> list[str]:
    """Return cleaned S&P 500 ticker symbols."""

    constituents = load_sp500_constituents()
    symbols = constituents["symbol"].dropna().tolist()
    return symbols[:limit] if limit is not None else symbols
