"""Portfolio and strategy performance analytics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _to_series(values: Iterable[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype(float).dropna()
    else:
        series = pd.Series(list(values), dtype=float).dropna()
    if series.empty:
        raise ValueError("returns must be non-empty")
    return series


def cumulative_returns(returns: Iterable[float]) -> pd.Series:
    series = _to_series(returns)
    return (1.0 + series).cumprod() - 1.0


def annualized_return(
    returns: Iterable[float],
    periods_per_year: int = 365,
) -> float:
    series = _to_series(returns)
    compounded = float((1.0 + series).prod())
    return compounded ** (periods_per_year / len(series)) - 1.0


def annualized_volatility(
    returns: Iterable[float],
    periods_per_year: int = 365,
) -> float:
    series = _to_series(returns)
    return float(series.std(ddof=0) * np.sqrt(periods_per_year))


def downside_volatility(
    returns: Iterable[float],
    periods_per_year: int = 365,
) -> float:
    series = _to_series(returns)
    downside = np.minimum(series.values, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    series = _to_series(returns)
    excess = series - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year=periods_per_year)
    if vol == 0:
        return 0.0
    return float(annualized_return(excess, periods_per_year=periods_per_year) / vol)


def sortino_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    series = _to_series(returns)
    excess = series - risk_free_rate / periods_per_year
    down_vol = downside_volatility(excess, periods_per_year=periods_per_year)
    if down_vol == 0:
        return 0.0
    return float(annualized_return(excess, periods_per_year=periods_per_year) / down_vol)


def max_drawdown(returns: Iterable[float]) -> float:
    curve = 1.0 + cumulative_returns(returns)
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def calmar_ratio(
    returns: Iterable[float],
    periods_per_year: int = 365,
) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annualized_return(returns, periods_per_year=periods_per_year) / mdd)


def hit_rate(returns: Iterable[float]) -> float:
    series = _to_series(returns)
    return float((series > 0).mean())


def summarize_strategy(
    returns: Iterable[float],
    benchmark_returns: Optional[Iterable[float]] = None,
    turnover: Optional[Iterable[float]] = None,
    periods_per_year: int = 365,
) -> Dict[str, float]:
    series = _to_series(returns)
    summary = {
        "observations": int(len(series)),
        "total_return": float((1.0 + series).prod() - 1.0),
        "annual_return": annualized_return(series, periods_per_year=periods_per_year),
        "annual_volatility": annualized_volatility(
            series, periods_per_year=periods_per_year
        ),
        "sharpe_ratio": sharpe_ratio(series, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(series, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(series),
        "calmar_ratio": calmar_ratio(series, periods_per_year=periods_per_year),
        "hit_rate": hit_rate(series),
    }

    if turnover is not None:
        turnover_series = _to_series(turnover)
        summary["average_turnover"] = float(turnover_series.mean())

    if benchmark_returns is not None:
        benchmark_series = _to_series(benchmark_returns)
        common_index = series.index.intersection(benchmark_series.index)
        if len(common_index) > 0:
            aligned = series.loc[common_index]
            benchmark_aligned = benchmark_series.loc[common_index]
            excess = aligned - benchmark_aligned
            tracking_error = annualized_volatility(
                excess, periods_per_year=periods_per_year
            )
            summary["benchmark_total_return"] = float(
                (1.0 + benchmark_aligned).prod() - 1.0
            )
            summary["tracking_error"] = tracking_error
            summary["information_ratio"] = (
                annualized_return(excess, periods_per_year=periods_per_year)
                / tracking_error
                if tracking_error > 0
                else 0.0
            )

    return summary
