"""
Walk-forward backtesting with baseline comparison and statistical tests.

Compares GARCH forecasts against naive baselines and tests whether
the improvement is statistically significant (Diebold-Mariano).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from crypto_volatility.src.garch_model import GARCHModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Holds walk-forward backtest outputs for one model."""
    name: str
    forecasts: pd.Series = field(default_factory=pd.Series)
    actuals: pd.Series = field(default_factory=pd.Series)
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    direction_accuracy: float = 0.0
    n_windows: int = 0


def walk_forward_garch(
    returns_pct: pd.Series,
    window: int = 252,
    step: int = 1,
) -> BacktestResult:
    """Walk-forward 1-step-ahead GARCH volatility forecast.

    At each step t, fit GARCH on [t-window : t], forecast vol at t,
    compare to |r_t| as a realised vol proxy.
    """
    n = len(returns_pct)
    if n < window + 10:
        raise ValueError("Not enough data for walk-forward evaluation")

    fc_vals, actual_vals, dates = [], [], []

    for i in range(window, n, step):
        train = returns_pct.iloc[i - window:i]
        try:
            g = GARCHModel()
            g.fit(train)
            fc = g.forecast(horizon=1)
            fc_vals.append(fc.values[0])
            actual_vals.append(abs(returns_pct.iloc[i]))
            dates.append(returns_pct.index[i])
        except Exception:
            continue

    if len(fc_vals) < 5:
        raise ValueError("Too few successful forecast windows")

    fc_s = pd.Series(fc_vals, index=dates, name="garch_forecast")
    actual_s = pd.Series(actual_vals, index=dates, name="actual_vol")

    errors = (fc_s - actual_s).values
    dir_correct = (np.sign(fc_s.diff().dropna()) == np.sign(actual_s.diff().dropna()))

    return BacktestResult(
        name="GARCH(1,1)",
        forecasts=fc_s,
        actuals=actual_s,
        mse=float(np.mean(errors ** 2)),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        direction_accuracy=float(dir_correct.mean()) if len(dir_correct) > 0 else 0.0,
        n_windows=len(fc_vals),
    )


def naive_baselines(returns_pct: pd.Series, window: int = 252, step: int = 1) -> Dict[str, BacktestResult]:
    """Compute naive volatility forecast baselines for comparison.

    Baselines:
        - Historical: rolling std of past `window` returns
        - EWMA: exponentially weighted std (lambda=0.94, RiskMetrics)
        - Random Walk: yesterday's |return| as tomorrow's vol forecast
    """
    n = len(returns_pct)
    results = {}

    # pre-compute for efficiency
    abs_returns = returns_pct.abs()

    for name, forecast_fn in [
        ("Historical Vol", lambda i: returns_pct.iloc[i - window:i].std()),
        ("EWMA (lambda=0.94)", lambda i: returns_pct.iloc[:i].ewm(span=int(1 / 0.06)).std().iloc[-1]),
        ("Random Walk", lambda i: abs_returns.iloc[i - 1]),
    ]:
        fc_vals, actual_vals, dates = [], [], []
        for i in range(window, n, step):
            try:
                fc_vals.append(float(forecast_fn(i)))
                actual_vals.append(float(abs_returns.iloc[i]))
                dates.append(returns_pct.index[i])
            except Exception:
                continue

        if len(fc_vals) < 5:
            continue

        fc_s = pd.Series(fc_vals, index=dates)
        actual_s = pd.Series(actual_vals, index=dates)
        errors = (fc_s - actual_s).values
        dir_correct = (np.sign(fc_s.diff().dropna()) == np.sign(actual_s.diff().dropna()))

        results[name] = BacktestResult(
            name=name,
            forecasts=fc_s,
            actuals=actual_s,
            mse=float(np.mean(errors ** 2)),
            mae=float(np.mean(np.abs(errors))),
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            direction_accuracy=float(dir_correct.mean()) if len(dir_correct) > 0 else 0.0,
            n_windows=len(fc_vals),
        )

    return results


def diebold_mariano_test(
    e1: np.ndarray, e2: np.ndarray, horizon: int = 1
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0 where d_t = e1_t^2 - e2_t^2.
    Negative test stat means model 1 has smaller loss (better).

    Returns (test_statistic, p_value).
    """
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    d_mean = d.mean()
    # Newey-West style variance with bandwidth = horizon - 1
    gamma_0 = np.var(d, ddof=1)
    cov_sum = 0.0
    for k in range(1, horizon):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d[k:]) > 1 else 0.0
        cov_sum += 2 * gamma_k

    var_d = (gamma_0 + cov_sum) / n
    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)

    return float(dm_stat), float(p_value)


def run_full_backtest(
    returns_pct: pd.Series,
    window: int = 252,
    step: int = 5,
) -> Tuple[BacktestResult, Dict[str, BacktestResult], Dict[str, Tuple[float, float]]]:
    """Run GARCH + baselines + DM tests. Returns (garch_result, baselines, dm_tests)."""

    garch_result = walk_forward_garch(returns_pct, window=window, step=step)
    baselines = naive_baselines(returns_pct, window=window, step=step)

    # align and run DM tests: GARCH vs each baseline
    dm_tests = {}
    garch_errors = (garch_result.forecasts - garch_result.actuals).values

    for name, baseline in baselines.items():
        # align on common dates
        common = garch_result.forecasts.index.intersection(baseline.forecasts.index)
        if len(common) < 10:
            continue
        e_garch = (garch_result.forecasts.loc[common] - garch_result.actuals.loc[common]).values
        e_base = (baseline.forecasts.loc[common] - baseline.actuals.loc[common]).values
        dm_stat, p_val = diebold_mariano_test(e_garch, e_base)
        dm_tests[name] = (dm_stat, p_val)

    return garch_result, baselines, dm_tests
