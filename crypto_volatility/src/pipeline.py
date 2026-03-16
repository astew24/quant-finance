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
    DEFAULT_DAYS_BACK,
    DEFAULT_SYMBOLS,
    FORECAST_HORIZON,
    GARCH_P,
    GARCH_Q,
    VOLATILITY_WINDOW,
)
from crypto_volatility.src.data_collector import CryptoDataCollector
from crypto_volatility.src.garch_model import GARCHModel
from crypto_volatility.src.metrics import calculate_rmse
from crypto_volatility.src.risk_utils import calculate_var

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

        lines.append(f"  Forecast ({len(r.forecast)}d):")
        for date, val in r.forecast.items():
            lines.append(f"    {date.date()} : {val:.6f}")
        lines.append("")

    return "\n".join(lines)
