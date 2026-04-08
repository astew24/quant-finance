"""
End-to-end crypto volatility forecasting pipeline.

Fetch data -> compute returns/vol -> fit GARCH -> forecast -> evaluate -> save.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crypto_volatility.config import (
    ANNUALISATION_FACTOR,
    DEFAULT_DAYS_BACK,
    DEFAULT_SYMBOLS,
    FORECAST_HORIZON,
    GARCH_P,
    GARCH_Q,
    ROLLING_WINDOW,
    VOLATILITY_WINDOW,
)
from crypto_volatility.src.backtesting import run_full_backtest
from crypto_volatility.src.data_collector import CryptoDataCollector
from crypto_volatility.src.garch_model import GARCHModel
from crypto_volatility.src.metrics import calculate_rmse
from crypto_volatility.src.performance import summarize_strategy
from crypto_volatility.src.risk_utils import calculate_var
from crypto_volatility.src.strategy import run_volatility_target_strategy

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    symbol: str
    prices: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    realised_vol: pd.Series = field(default_factory=pd.Series)
    conditional_vol: pd.Series = field(default_factory=pd.Series)
    forecast: pd.Series = field(default_factory=pd.Series)
    model_summary: Dict = field(default_factory=dict)
    eval_metrics: Dict = field(default_factory=dict)
    backtest_metrics: Dict = field(default_factory=dict)
    baseline_metrics: Dict[str, Dict] = field(default_factory=dict)
    dm_test_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strategy_summary: Dict = field(default_factory=dict)
    benchmark_summary: Dict = field(default_factory=dict)
    strategy_returns: pd.Series = field(default_factory=pd.Series)
    strategy_positions: pd.Series = field(default_factory=pd.Series)
    strategy_turnover: pd.Series = field(default_factory=pd.Series)
    var_95: float = 0.0
    var_99: float = 0.0


def run_pipeline(
    symbols: Optional[List[str]] = None,
    days_back: int = DEFAULT_DAYS_BACK,
    forecast_horizon: int = FORECAST_HORIZON,
    output_dir: str = "output",
) -> Dict[str, PipelineResult]:
    """Run the full pipeline for each symbol."""

    symbols = symbols or DEFAULT_SYMBOLS
    collector = CryptoDataCollector(symbols=symbols)
    results: Dict[str, PipelineResult] = {}

    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        logger.info("=" * 60)
        logger.info(f"Processing {symbol}")
        logger.info("=" * 60)

        result = PipelineResult(symbol=symbol)

        # 1) fetch data
        df = collector.fetch_ohlcv(symbol, start=_start_date(days_back))
        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        col = "Close" if "Close" in df.columns else "close"
        result.prices = df[col]

        # 2) returns & realised vol
        result.returns = collector.calculate_returns(df).dropna()
        result.realised_vol = collector.calculate_volatility(
            result.returns, window=VOLATILITY_WINDOW
        )

        logger.info(
            f"  Data: {result.returns.index[0].date()} to "
            f"{result.returns.index[-1].date()} ({len(result.returns)} obs)"
        )

        # 3) fit GARCH (scale to pct for numerical stability)
        returns_pct = result.returns * 100
        garch = GARCHModel(p=GARCH_P, q=GARCH_Q)
        garch.fit(returns_pct)

        summary = garch.get_model_summary()
        result.model_summary = {
            "aic": summary["aic"],
            "bic": summary["bic"],
            "log_likelihood": summary["log_likelihood"],
            "params": summary["params"],
        }
        result.conditional_vol = summary["conditional_volatility"] / 100

        logger.info(f"  GARCH AIC={summary['aic']:.2f}  BIC={summary['bic']:.2f}")

        # 4) forecast
        forecast_pct = garch.forecast(horizon=forecast_horizon)
        result.forecast = forecast_pct / 100
        logger.info(f"  {forecast_horizon}-day forecast:")
        for date, val in result.forecast.items():
            logger.info(f"    {date.date()}: {val:.6f}")

        # 5) in-sample evaluation
        common = result.realised_vol.dropna().index.intersection(
            result.conditional_vol.index
        )
        if len(common) > 0:
            actual = result.realised_vol.loc[common]
            fitted = result.conditional_vol.loc[common]
            rmse = calculate_rmse(actual.values, fitted.values)
            corr = float(np.corrcoef(actual.values, fitted.values)[0, 1])
            result.eval_metrics = {
                "in_sample_rmse": rmse,
                "correlation": corr,
                "n_obs": len(common),
            }
            logger.info(f"  RMSE: {rmse:.6f}  Corr: {corr:.4f}")

        # 5b) walk-forward backtest + volatility-managed overlay
        backtest_window = min(ROLLING_WINDOW, max(120, len(returns_pct) // 2))
        if len(returns_pct) >= backtest_window + 10:
            garch_backtest, baselines, dm_tests = run_full_backtest(
                returns_pct,
                window=backtest_window,
                step=1,
            )
            result.backtest_metrics = {
                "rmse": garch_backtest.rmse,
                "mae": garch_backtest.mae,
                "mse": garch_backtest.mse,
                "direction_accuracy": garch_backtest.direction_accuracy,
                "n_windows": garch_backtest.n_windows,
            }
            result.baseline_metrics = {
                name: {
                    "rmse": baseline.rmse,
                    "mae": baseline.mae,
                    "direction_accuracy": baseline.direction_accuracy,
                    "n_windows": baseline.n_windows,
                }
                for name, baseline in baselines.items()
            }
            result.dm_test_results = {
                name: {"stat": stat, "p_value": p_value}
                for name, (stat, p_value) in dm_tests.items()
            }

            strategy = run_volatility_target_strategy(
                result.returns,
                garch_backtest.forecasts / 100.0,
                target_annual_vol=0.35,
                max_leverage=2.0,
                transaction_cost_bps=10.0,
                periods_per_year=ANNUALISATION_FACTOR,
            )
            result.strategy_returns = strategy.net_returns
            result.strategy_positions = strategy.positions
            result.strategy_turnover = strategy.turnover
            result.strategy_summary = strategy.summary
            result.benchmark_summary = strategy.benchmark_summary
            logger.info(
                "  Strategy Sharpe: %.3f  MaxDD: %.2f%%  Avg leverage: %.2f",
                strategy.summary["sharpe_ratio"],
                strategy.summary["max_drawdown"] * 100,
                strategy.summary["average_leverage"],
            )
        else:
            result.benchmark_summary = summarize_strategy(
                result.returns, periods_per_year=ANNUALISATION_FACTOR
            )

        # 6) risk metrics
        result.var_95 = calculate_var(result.returns.values, 0.05)
        result.var_99 = calculate_var(result.returns.values, 0.01)
        logger.info(f"  VaR(95%): {result.var_95:.4%}  VaR(99%): {result.var_99:.4%}")

        results[symbol] = result

    _save_outputs(results, output_dir)
    return results


def _start_date(days_back: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")


def _save_outputs(results: Dict[str, PipelineResult], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for sym, r in results.items():
        safe = sym.replace("/", "_").replace("-", "_")

        # timeseries csv
        ts = pd.DataFrame({"returns": r.returns, "realised_vol": r.realised_vol})
        if not r.conditional_vol.empty:
            ts["conditional_vol"] = r.conditional_vol
        ts.to_csv(os.path.join(output_dir, f"{safe}_timeseries.csv"))

        # forecast csv
        r.forecast.to_csv(
            os.path.join(output_dir, f"{safe}_forecast.csv"),
            header=["forecast_vol"],
        )
        if not r.strategy_returns.empty:
            strategy = pd.DataFrame(
                {
                    "strategy_returns": r.strategy_returns,
                    "position": r.strategy_positions,
                    "turnover": r.strategy_turnover,
                }
            )
            strategy.to_csv(os.path.join(output_dir, f"{safe}_strategy.csv"))

        rows.append({
            "symbol": sym,
            "n_obs": len(r.returns),
            "mean_return": r.returns.mean(),
            "ann_vol": r.returns.std() * np.sqrt(365),
            "VaR_95": r.var_95,
            "VaR_99": r.var_99,
            "GARCH_AIC": r.model_summary.get("aic"),
            "GARCH_BIC": r.model_summary.get("bic"),
            **r.eval_metrics,
            **{f"garch_backtest_{k}": v for k, v in r.backtest_metrics.items()},
            **{f"strategy_{k}": v for k, v in r.strategy_summary.items()},
            **{f"buy_hold_{k}": v for k, v in r.benchmark_summary.items()},
        })

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    logger.info(f"Outputs saved to {output_dir}/")


def generate_report(results: Dict[str, PipelineResult]) -> str:
    """Plain-text summary report."""
    lines = ["=" * 60, "Crypto Volatility Forecasting Report", "=" * 60, ""]

    for sym, r in results.items():
        lines.append(f"--- {sym} ---")
        lines.append(f"  Observations : {len(r.returns)}")
        lines.append(
            f"  Date range   : {r.returns.index[0].date()} to "
            f"{r.returns.index[-1].date()}"
        )
        lines.append(f"  Mean return  : {r.returns.mean():.6f}")
        lines.append(f"  Ann. vol     : {r.returns.std() * np.sqrt(365):.4%}")
        lines.append(f"  VaR (95%)    : {r.var_95:.4%}")
        lines.append(f"  VaR (99%)    : {r.var_99:.4%}")

        aic = r.model_summary.get("aic")
        bic = r.model_summary.get("bic")
        if aic is not None:
            lines.append(f"  GARCH AIC    : {aic:.2f}")
            lines.append(f"  GARCH BIC    : {bic:.2f}")

        if r.eval_metrics:
            lines.append(f"  RMSE         : {r.eval_metrics['in_sample_rmse']:.6f}")
            lines.append(f"  Correlation  : {r.eval_metrics['correlation']:.4f}")

        if r.backtest_metrics:
            lines.append("  Walk-forward backtest:")
            lines.append(f"    RMSE       : {r.backtest_metrics['rmse']:.6f}")
            lines.append(
                f"    Direction  : {r.backtest_metrics['direction_accuracy']:.2%}"
            )
            for name, baseline in r.baseline_metrics.items():
                dm = r.dm_test_results.get(name, {})
                p_val = dm.get("p_value")
                lines.append(
                    f"    vs {name:<14} RMSE={baseline['rmse']:.6f}  "
                    f"DM p-value={p_val:.4f}" if p_val is not None else
                    f"    vs {name:<14} RMSE={baseline['rmse']:.6f}"
                )

        if r.strategy_summary:
            lines.append("  Vol-managed overlay:")
            lines.append(
                f"    Sharpe     : {r.strategy_summary['sharpe_ratio']:.3f}"
            )
            lines.append(
                f"    Max drawdown: {r.strategy_summary['max_drawdown']:.2%}"
            )
            lines.append(
                f"    Avg leverage: {r.strategy_summary['average_leverage']:.2f}"
            )

        lines.append(f"  Forecast ({len(r.forecast)}d):")
        for date, val in r.forecast.items():
            lines.append(f"    {date.date()} : {val:.6f}")
        lines.append("")

    return "\n".join(lines)
