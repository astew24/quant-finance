"""Quantamental valuation and thesis generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ThesisResult:
    """Structured outputs for top screened ideas."""

    ideas_table: pd.DataFrame
    markdown_report: str


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def estimate_cost_of_equity(
    beta: float | None,
    risk_free_rate: float = 0.043,
    equity_risk_premium: float = 0.050,
) -> float:
    """Simple CAPM-based cost of equity."""

    beta = 1.0 if beta is None or np.isnan(beta) else float(beta)
    return risk_free_rate + beta * equity_risk_premium


def simple_dcf_per_share(
    free_cash_flow: float | None,
    shares_outstanding: float | None,
    total_cash: float | None,
    total_debt: float | None,
    growth_rate: float | None,
    cost_of_equity: float,
    terminal_growth: float = 0.025,
    forecast_years: int = 5,
) -> float | None:
    """Estimate intrinsic value per share from a simple FCF DCF."""

    required = [free_cash_flow, shares_outstanding]
    if any(value is None or np.isnan(value) or value <= 0 for value in required):
        return None

    fcf = float(free_cash_flow)
    shares = float(shares_outstanding)
    cash = 0.0 if total_cash is None or np.isnan(total_cash) else float(total_cash)
    debt = 0.0 if total_debt is None or np.isnan(total_debt) else float(total_debt)
    growth = 0.04 if growth_rate is None or np.isnan(growth_rate) else float(growth_rate)
    growth = _clip(growth, -0.02, 0.18)
    discount_rate = _clip(cost_of_equity, 0.06, 0.18)
    terminal_growth = min(terminal_growth, discount_rate - 0.01)

    present_value = 0.0
    current_fcf = fcf
    for year in range(1, forecast_years + 1):
        current_fcf *= 1.0 + growth
        present_value += current_fcf / ((1.0 + discount_rate) ** year)

    terminal_value = (
        current_fcf * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    )
    terminal_pv = terminal_value / ((1.0 + discount_rate) ** forecast_years)
    equity_value = present_value + terminal_pv + cash - debt
    return equity_value / shares if shares > 0 else None


def build_top_idea_table(
    screen: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Enrich top-ranked ideas with valuation context."""

    top = screen.head(top_n).copy()
    sector_pe = screen.groupby("sector")["forward_pe"].median()
    sector_pb = screen.groupby("sector")["price_to_book"].median()
    sector_factor = screen.groupby("sector")["factor_score"].median()

    top["sector_forward_pe_median"] = top["sector"].map(sector_pe)
    top["sector_price_to_book_median"] = top["sector"].map(sector_pb)
    top["forward_pe_discount_to_sector"] = (
        top["sector_forward_pe_median"] - top["forward_pe"]
    ) / top["sector_forward_pe_median"]
    top["price_to_book_discount_to_sector"] = (
        top["sector_price_to_book_median"] - top["price_to_book"]
    ) / top["sector_price_to_book_median"]
    top["factor_score_vs_sector"] = top["factor_score"] - top["sector"].map(sector_factor)
    current_price = top.get("currentPrice", pd.Series(np.nan, index=top.index))
    target_mean_price = top.get("targetMeanPrice", pd.Series(np.nan, index=top.index))
    top["street_target_upside"] = (target_mean_price - current_price) / current_price
    top["street_target_upside"] = top["street_target_upside"].replace([np.inf, -np.inf], np.nan)

    return top


def build_quantamental_theses(
    screen: pd.DataFrame,
    prices: pd.DataFrame,
    top_n: int = 5,
) -> ThesisResult:
    """Generate concise, quantitative thesis writeups for top-ranked names."""

    top = build_top_idea_table(screen, top_n=top_n)
    price_frame = prices.ffill().reindex(columns=top.index)
    trailing_returns = price_frame.pct_change()
    one_year_return = price_frame.iloc[-1] / price_frame.iloc[-253] - 1.0
    annual_vol = trailing_returns.iloc[-252:].std() * np.sqrt(252)
    sharpe_1y = (
        trailing_returns.iloc[-252:].mean() / trailing_returns.iloc[-252:].std()
    ) * np.sqrt(252)

    top["trailing_1y_return"] = one_year_return.reindex(top.index)
    top["trailing_1y_volatility"] = annual_vol.reindex(top.index)
    top["trailing_1y_sharpe"] = sharpe_1y.reindex(top.index)

    dcf_values: List[float | None] = []
    dcf_upside: List[float | None] = []
    for _, row in top.iterrows():
        cost_of_equity = estimate_cost_of_equity(row.get("beta"))
        dcf_value = simple_dcf_per_share(
            row.get("free_cashflow"),
            row.get("shares_outstanding"),
            row.get("total_cash"),
            row.get("total_debt"),
            row.get("earnings_growth"),
            cost_of_equity,
        )
        dcf_values.append(dcf_value)
        current_price = row.get("currentPrice")
        if dcf_value is None or current_price is None or np.isnan(current_price) or current_price <= 0:
            dcf_upside.append(np.nan)
        else:
            upside = dcf_value / current_price - 1.0
            if upside < -0.50 or upside > 2.00:
                dcf_values[-1] = None
                dcf_upside.append(np.nan)
            else:
                dcf_upside.append(upside)

    top["dcf_fair_value"] = dcf_values
    top["dcf_upside"] = dcf_upside

    lines = [
        "# Quantamental Equity Research Brief",
        "",
        "This brief summarizes the highest-ranked names from the value-momentum-quality screener.",
        "",
    ]

    for symbol, row in top.iterrows():
        valuation_signal = row["forward_pe_discount_to_sector"]
        quality_signal = row["quality_score"]
        momentum_signal = row["momentum_score"]
        risk_flag = []
        if pd.notna(row.get("debt_to_equity")) and row["debt_to_equity"] > 150:
            risk_flag.append("leverage is elevated")
        if pd.notna(row.get("trailing_1y_volatility")) and row["trailing_1y_volatility"] > 0.40:
            risk_flag.append("price volatility is above a typical large-cap profile")
        if pd.notna(row.get("dcf_upside")) and row["dcf_upside"] < -0.10:
            risk_flag.append("DCF support is limited at the current price")
        risk_text = "; ".join(risk_flag) if risk_flag else "primary risks are factor crowding and earnings revisions"

        lines.extend(
            [
                f"## {symbol}",
                f"- Sector: {row.get('sector', 'N/A')}",
                f"- Screen rank: {int(row['screen_rank'])} with composite screen score `{row['screen_score']:.2f}` and predicted outperformance probability `{row['predicted_outperformance_probability']:.2%}`",
                f"- Factor profile: value `{row['value_score']:.2f}`, momentum `{row['momentum_score']:.2f}`, quality `{row['quality_score']:.2f}`",
                (
                    f"- Relative valuation: forward P/E `{row['forward_pe']:.1f}x` versus sector median `{row['sector_forward_pe_median']:.1f}x` "
                    f"({row['forward_pe_discount_to_sector']:+.1%}); street target implies `{row['street_target_upside']:+.1%}` upside"
                    if pd.notna(row.get("street_target_upside"))
                    else f"- Relative valuation: forward P/E `{row['forward_pe']:.1f}x` versus sector median `{row['sector_forward_pe_median']:.1f}x` ({row['forward_pe_discount_to_sector']:+.1%})"
                ),
                f"- Risk-adjusted profile: trailing 1Y return `{row['trailing_1y_return']:.1%}`, annualized volatility `{row['trailing_1y_volatility']:.1%}`, trailing 1Y Sharpe `{row['trailing_1y_sharpe']:.2f}`",
                (
                    f"- DCF: estimated fair value `${row['dcf_fair_value']:.2f}` with implied upside `{row['dcf_upside']:+.1%}`"
                    if pd.notna(row.get("dcf_fair_value"))
                    else "- DCF: insufficient fundamental coverage for a stable fair-value estimate"
                ),
                (
                    "- Investment thesis: the name screens well because it combines positive price leadership with above-median operating quality and valuation support."
                    if momentum_signal > 0 and quality_signal > 0 and valuation_signal > 0
                    else "- Investment thesis: the signal is more tactical than deep value, with strength driven by the current factor mix rather than one single cheapness metric."
                ),
                f"- Key risks: {risk_text}.",
                "",
            ]
        )

    return ThesisResult(ideas_table=top, markdown_report="\n".join(lines).strip() + "\n")
