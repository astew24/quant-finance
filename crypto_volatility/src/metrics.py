# Performance metrics and evaluation
from typing import Iterable
import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_arr.size == 0:
        raise ValueError("Inputs must be non-empty")
    return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
