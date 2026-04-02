# Risk management utilities
from typing import Iterable
import numpy as np


def calculate_var(returns: Iterable[float], confidence_level: float = 0.05) -> float:
    """Historical Value-at-Risk at the given tail probability."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    returns_arr = np.asarray(list(returns), dtype=float)
    if returns_arr.size == 0:
        raise ValueError("returns must be non-empty")
    return float(np.percentile(returns_arr, confidence_level * 100))


def calculate_cvar(returns: Iterable[float], confidence_level: float = 0.05) -> float:
    """Expected Shortfall (CVaR): average loss beyond the VaR threshold.

    More coherent than VaR because it accounts for tail severity,
    not just tail frequency.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    returns_arr = np.asarray(list(returns), dtype=float)
    if returns_arr.size == 0:
        raise ValueError("returns must be non-empty")
    var = float(np.percentile(returns_arr, confidence_level * 100))
    tail = returns_arr[returns_arr <= var]
    if tail.size == 0:
        return var
    return float(tail.mean())


def detect_regimes(vol_series, n_regimes: int = 2):
    """Simple threshold-based regime detection on a volatility series.

    Splits into low-vol and high-vol regimes using the median as boundary.
    Returns a Series of regime labels (0 = low, 1 = high) and the threshold.

    Note: only binary regime detection is implemented right now — n_regimes
    values other than 2 are ignored. Would need quantile-based splitting for
    more than two regimes.
    """
    import pandas as pd
    vol_clean = vol_series.dropna()
    if len(vol_clean) < 10:
        raise ValueError("Need at least 10 observations for regime detection")
    threshold = float(vol_clean.median())
    labels = (vol_clean > threshold).astype(int)
    return labels, threshold
