# Portfolio optimization utilities
from typing import Iterable
import numpy as np


def calculate_sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.02) -> float:
    returns_arr = np.asarray(list(returns), dtype=float)
    if returns_arr.size == 0:
        raise ValueError("returns must be non-empty")
    excess_returns = returns_arr - risk_free_rate / 252
    std = excess_returns.std()
    if std == 0:
        raise ValueError("returns must have non-zero variance")
    return float(excess_returns.mean() / std)
