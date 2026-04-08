"""Cross-sectional factor backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_risk_model.src.analytics import summarize_returns


@dataclass
class FactorBacktestResult:
    """Results for a long-short factor strategy."""

    portfolio_returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=pd.Series)
    information_coefficient: pd.Series = field(default_factory=pd.Series)
    summary: Dict[str, float] = field(default_factory=dict)


def _make_long_short_weights(
    scores: pd.Series,
    selection_quantile: float,
) -> pd.Series:
    valid = scores.dropna().sort_values()
    weights = pd.Series(0.0, index=scores.index)
    if len(valid) < 4:
        return weights

    bucket_size = max(1, int(np.floor(len(valid) * selection_quantile)))
    shorts = valid.index[:bucket_size]
    longs = valid.index[-bucket_size:]

    weights.loc[longs] = 0.5 / len(longs)
    weights.loc[shorts] = -0.5 / len(shorts)
    return weights


def _holding_period_return(returns: pd.DataFrame) -> pd.Series:
    return (1.0 + returns).prod() - 1.0


def run_factor_backtest(
    returns: pd.DataFrame,
    composite_score: pd.DataFrame,
    standardized_signals: Dict[str, pd.DataFrame],
    rebalance_frequency: int = 63,
    selection_quantile: float = 0.2,
    transaction_cost_bps: float = 10.0,
    periods_per_year: int = 252,
) -> FactorBacktestResult:
    """Backtest a dollar-neutral factor strategy."""

    if returns.empty or composite_score.empty:
        raise ValueError("returns and composite_score must be non-empty")
    if not 0 < selection_quantile < 0.5:
        raise ValueError("selection_quantile must be between 0 and 0.5")

    common_index = returns.index.intersection(composite_score.index)
    returns = returns.loc[common_index].sort_index()
    composite_score = composite_score.loc[common_index].sort_index()
    standardized_signals = {
        name: signal.loc[common_index].sort_index()
        for name, signal in standardized_signals.items()
    }

    valid_rebalance_dates = composite_score.index[
        composite_score.notna().sum(axis=1) >= 4
    ]
    if len(valid_rebalance_dates) < 3:
        raise ValueError("not enough valid factor observations to backtest")

    rebalance_dates = valid_rebalance_dates[::rebalance_frequency]
    portfolio_returns = pd.Series(dtype=float)
    benchmark_returns = pd.Series(dtype=float)
    turnover = pd.Series(dtype=float)
    information_coefficient = pd.Series(dtype=float)
    weights_history = pd.DataFrame(0.0, index=rebalance_dates, columns=returns.columns)
    factor_returns = pd.DataFrame(dtype=float)

    previous_weights = pd.Series(0.0, index=returns.columns)

    for i, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[i + 1]
        start_loc = returns.index.get_loc(rebalance_date) + 1
        end_loc = returns.index.get_loc(next_rebalance_date) + 1
        if start_loc >= end_loc:
            continue

        holding_returns = returns.iloc[start_loc:end_loc].fillna(0.0)
        if holding_returns.empty:
            continue

        current_scores = composite_score.loc[rebalance_date]
        current_weights = _make_long_short_weights(
            current_scores,
            selection_quantile=selection_quantile,
        )
        weights_history.loc[rebalance_date] = current_weights
        current_turnover = float((current_weights - previous_weights).abs().sum())
        turnover.loc[rebalance_date] = current_turnover

        period_portfolio = holding_returns.dot(current_weights)
        if not period_portfolio.empty:
            period_portfolio.iloc[0] -= current_turnover * transaction_cost_bps / 10_000.0
        portfolio_returns = pd.concat([portfolio_returns, period_portfolio])

        period_benchmark = holding_returns.mean(axis=1)
        benchmark_returns = pd.concat([benchmark_returns, period_benchmark])

        forward_returns = _holding_period_return(holding_returns)
        aligned_scores = current_scores.dropna()
        aligned_forward = forward_returns.reindex(aligned_scores.index).dropna()
        aligned_scores = aligned_scores.loc[aligned_forward.index]
        if len(aligned_scores) >= 3:
            ic = spearmanr(aligned_scores, aligned_forward).correlation
            information_coefficient.loc[rebalance_date] = (
                0.0 if np.isnan(ic) else float(ic)
            )

        period_factor_returns = {}
        for factor_name, signal_frame in standardized_signals.items():
            factor_weights = _make_long_short_weights(
                signal_frame.loc[rebalance_date],
                selection_quantile=selection_quantile,
            )
            period_factor_returns[factor_name] = holding_returns.dot(factor_weights)
        factor_returns = pd.concat(
            [factor_returns, pd.DataFrame(period_factor_returns)],
            axis=0,
        )

        previous_weights = current_weights

    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="last")]
    benchmark_returns = benchmark_returns.reindex(portfolio_returns.index)
    factor_returns = factor_returns.loc[portfolio_returns.index]

    summary = summarize_returns(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        turnover=turnover,
        periods_per_year=periods_per_year,
    )
    summary["mean_information_coefficient"] = float(information_coefficient.mean())
    ic_std = float(information_coefficient.std(ddof=0))
    summary["information_coefficient_ir"] = (
        float(information_coefficient.mean() / ic_std) if ic_std > 0 else 0.0
    )
    summary["selection_quantile"] = float(selection_quantile)
    summary["transaction_cost_bps"] = float(transaction_cost_bps)

    return FactorBacktestResult(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        weights=weights_history,
        factor_returns=factor_returns,
        turnover=turnover,
        information_coefficient=information_coefficient,
        summary=summary,
    )
