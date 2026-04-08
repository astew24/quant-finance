"""End-to-end pipeline for the equity factor research project."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from factor_risk_model.src.backtesting import FactorBacktestResult, run_factor_backtest
from factor_risk_model.src.data_collector import DEFAULT_UNIVERSE, StockDataCollector
from factor_risk_model.src.factor_construction import FactorSet, build_factor_signals
from factor_risk_model.src.factor_regression import FactorRegression

logger = logging.getLogger(__name__)


@dataclass
class FactorResearchResult:
    """Outputs for the factor research pipeline."""

    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_set: Optional[FactorSet] = None
    backtest: Optional[FactorBacktestResult] = None
    exposures: pd.Series = field(default_factory=pd.Series)
    exposure_summary: Dict = field(default_factory=dict)


def run_factor_research(
    symbols=None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "factor_risk_model/output",
    rebalance_frequency: int = 63,
    selection_quantile: float = 0.2,
    transaction_cost_bps: float = 10.0,
) -> FactorResearchResult:
    """Execute the full factor research workflow."""

    collector = StockDataCollector(
        symbols=symbols or DEFAULT_UNIVERSE,
        start_date=start_date,
        end_date=end_date,
    )
    prices = collector.fetch_price_data()
    if prices.empty:
        raise ValueError("No prices downloaded for the requested universe")

    returns = collector.calculate_returns(prices).dropna(how="all")
    factor_set = build_factor_signals(prices, returns)
    backtest = run_factor_backtest(
        returns=returns,
        composite_score=factor_set.composite_score,
        standardized_signals=factor_set.standardized_signals,
        rebalance_frequency=rebalance_frequency,
        selection_quantile=selection_quantile,
        transaction_cost_bps=transaction_cost_bps,
    )

    factor_frame = backtest.factor_returns.copy()
    factor_frame["market"] = backtest.benchmark_returns
    regression = FactorRegression(method="ols", scale_factors=False)
    regression.fit(backtest.portfolio_returns, factor_frame)

    result = FactorResearchResult(
        prices=prices,
        returns=returns,
        factor_set=factor_set,
        backtest=backtest,
        exposures=regression.exposures,
        exposure_summary=regression.get_model_summary(),
    )
    _save_outputs(result, output_dir)
    return result


def _save_outputs(result: FactorResearchResult, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    result.prices.to_csv(os.path.join(output_dir, "prices.csv"))
    result.returns.to_csv(os.path.join(output_dir, "returns.csv"))
    result.factor_set.composite_score.to_csv(os.path.join(output_dir, "composite_score.csv"))
    result.backtest.weights.to_csv(os.path.join(output_dir, "weights.csv"))
    pd.DataFrame(
        {
            "strategy_returns": result.backtest.portfolio_returns,
            "benchmark_returns": result.backtest.benchmark_returns,
        }
    ).to_csv(os.path.join(output_dir, "strategy_returns.csv"))
    result.backtest.factor_returns.to_csv(os.path.join(output_dir, "factor_returns.csv"))
    result.backtest.information_coefficient.to_csv(
        os.path.join(output_dir, "information_coefficient.csv"),
        header=["ic"],
    )
    result.exposures.to_csv(os.path.join(output_dir, "factor_exposures.csv"), header=["beta"])
    pd.DataFrame([result.backtest.summary]).to_csv(
        os.path.join(output_dir, "summary.csv"),
        index=False,
    )


def generate_report(result: FactorResearchResult) -> str:
    """Generate a concise text report for CLI use."""

    lines = [
        "=" * 60,
        "Cross-Sectional Factor Research Report",
        "=" * 60,
        "",
        f"Universe size : {len(result.prices.columns)}",
        f"Date range    : {result.returns.index[0].date()} to {result.returns.index[-1].date()}",
        f"Observations  : {len(result.backtest.portfolio_returns)} daily returns",
        "",
        "Strategy metrics:",
        f"  Annual return   : {result.backtest.summary['annual_return']:.2%}",
        f"  Annual volatility: {result.backtest.summary['annual_volatility']:.2%}",
        f"  Sharpe ratio    : {result.backtest.summary['sharpe_ratio']:.2f}",
        f"  Max drawdown    : {result.backtest.summary['max_drawdown']:.2%}",
        f"  Mean IC         : {result.backtest.summary['mean_information_coefficient']:.3f}",
        "",
        "Factor exposures:",
    ]

    for factor_name, beta in result.exposures.items():
        lines.append(f"  {factor_name:<20} {beta: .3f}")

    return "\n".join(lines)
