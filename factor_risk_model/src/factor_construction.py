"""Signal construction for the equity factor research project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class FactorSet:
    """Raw and standardized factor signals plus a composite score."""

    raw_signals: Dict[str, pd.DataFrame]
    standardized_signals: Dict[str, pd.DataFrame]
    composite_score: pd.DataFrame


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Demean and scale each row across the cross-section.

    Using the cross-sectional mean/std (axis=1) rather than time-series stats
    keeps the z-scores comparable across dates even when the universe changes.
    """
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def build_factor_signals(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    momentum_window: int = 252,
    skip_window: int = 21,
    reversal_window: int = 21,
    volatility_window: int = 63,
) -> FactorSet:
    """Build momentum, short-term reversal, and low-volatility signals."""

    if prices.empty or returns.empty:
        raise ValueError("prices and returns must be non-empty")

    raw_signals = {
        "momentum_12_1": prices.shift(skip_window).div(prices.shift(momentum_window)) - 1.0,
        "short_term_reversal": -(prices.div(prices.shift(reversal_window)) - 1.0),
        "low_volatility": -returns.rolling(volatility_window).std(),
    }
    standardized = {
        name: _cross_sectional_zscore(frame)
        for name, frame in raw_signals.items()
    }

    composite = (
        0.5 * standardized["momentum_12_1"]
        + 0.25 * standardized["short_term_reversal"]
        + 0.25 * standardized["low_volatility"]
    )

    return FactorSet(
        raw_signals=raw_signals,
        standardized_signals=standardized,
        composite_score=composite,
    )
