# Risk management utilities
from typing import Iterable
import numpy as np


def calculate_var(returns: Iterable[float], confidence_level: float = 0.05) -> float:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    returns_arr = np.asarray(list(returns), dtype=float)
    if returns_arr.size == 0:
        raise ValueError("returns must be non-empty")
    return float(np.percentile(returns_arr, confidence_level * 100))
