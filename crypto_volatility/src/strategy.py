"""Trading overlays derived from volatility forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from crypto_volatility.src.performance import (
    annualized_volatility,
    cumulative_returns,
    summarize_strategy,
)


@dataclass
class StrategyResult:
    """Container for a volatility-managed strategy backtest."""

    positions: pd.Series = field(default_factory=pd.Series)
    gross_returns: pd.Series = field(default_factory=pd.Series)
    net_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    turnover: pd.Series = field(default_factory=pd.Series)
    cumulative_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)
    summary: Dict[str, float] = field(default_factory=dict)
    benchmark_summary: Dict[str, float] = field(default_factory=dict)


def run_volatility_target_strategy(
    asset_returns: pd.Series,
    forecast_volatility: pd.Series,
    target_annual_vol: float = 0.35,
    max_leverage: float = 2.0,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 365,
) -> StrategyResult:
    """Scale exposure inversely with forecast volatility."""

    common_index = asset_returns.dropna().index.intersection(
        forecast_volatility.dropna().index
    )
    if len(common_index) < 20:
        raise ValueError("Need at least 20 overlapping observations")

    returns = asset_returns.loc[common_index].astype(float)
    forecast = forecast_volatility.loc[common_index].abs().astype(float)
    if (forecast <= 0).all():
        raise ValueError("forecast_volatility must contain positive values")

    target_daily_vol = target_annual_vol / np.sqrt(periods_per_year)
    positions = (
        target_daily_vol / forecast.replace(0.0, np.nan)
    ).clip(lower=0.0, upper=max_leverage).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())

    gross_returns = positions * returns
    costs = turnover * (transaction_cost_bps / 10_000.0)
    net_returns = gross_returns - costs

    summary = summarize_strategy(
        net_returns,
        benchmark_returns=returns,
        turnover=turnover,
        periods_per_year=periods_per_year,
    )
    benchmark_summary = summarize_strategy(returns, periods_per_year=periods_per_year)
    summary["realized_annual_volatility"] = annualized_volatility(
        net_returns, periods_per_year=periods_per_year
    )
    summary["target_annual_volatility"] = target_annual_vol
    summary["volatility_target_error"] = (
        summary["realized_annual_volatility"] - target_annual_vol
    )
    summary["average_leverage"] = float(positions.mean())
    summary["max_leverage"] = float(positions.max())
    summary["transaction_cost_bps"] = float(transaction_cost_bps)

    return StrategyResult(
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        benchmark_returns=returns,
        turnover=turnover,
        cumulative_curve=cumulative_returns(net_returns),
        benchmark_curve=cumulative_returns(returns),
        summary=summary,
        benchmark_summary=benchmark_summary,
    )
