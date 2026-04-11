"""
Quant finance portfolio showcase built on top of committed project artifacts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_volatility.src.backtesting import run_full_backtest
from crypto_volatility.src.data_collector import CryptoDataCollector
from crypto_volatility.src.demo_data import (
    available_sample_symbols,
    load_sample_forecast,
    load_sample_market_data,
    sample_data_exists,
)
from crypto_volatility.src.garch_model import GARCHModel
from crypto_volatility.src.risk_utils import calculate_cvar, calculate_var, detect_regimes
from options_pricing.src.black_scholes import black_scholes_greeks, black_scholes_price
from options_pricing.src.numerical_methods import american_option_binomial, monte_carlo_price

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
CRYPTO_SAMPLE_DIR = ROOT / "crypto_volatility" / "output_sample"
FACTOR_SAMPLE_DIR = ROOT / "factor_risk_model" / "output_sample"
CRYPTO_LIVE_AVAILABLE = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]
CRYPTO_SAMPLE_AVAILABLE = available_sample_symbols()


st.set_page_config(
    page_title="Quant Finance Portfolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


COLORS = {
    "primary": "#0f766e",
    "secondary": "#0f172a",
    "accent": "#d97706",
    "success": "#15803d",
    "warning": "#b45309",
    "danger": "#b91c1c",
    "muted": "#64748b",
    "bg_card": "#f8fafc",
    "text": "#0f172a",
}

CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=COLORS["text"]),
    margin=dict(l=8, r=8, t=52, b=8),
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --card-border: #dbe4ee;
    --card-bg: rgba(248, 250, 252, 0.88);
    --page-accent: #0f766e;
    --page-secondary: #0f172a;
    --page-warm: #d97706;
}

.stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 35%),
        radial-gradient(circle at top right, rgba(217, 119, 6, 0.10), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #edf2f7 100%);
}

h1 {
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: var(--page-secondary) !important;
}

h2, h3 {
    font-weight: 650 !important;
    letter-spacing: -0.02em !important;
    color: var(--page-secondary) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(248,250,252,0.96));
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b !important;
}

[data-testid="stMetricValue"] {
    color: var(--page-secondary) !important;
    font-weight: 700 !important;
}

.portfolio-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    min-height: 180px;
}

.portfolio-card h3 {
    margin: 0 0 8px 0;
    font-size: 1.1rem;
}

.portfolio-card p {
    margin: 0;
    color: #334155;
    line-height: 1.5;
}

.source-note {
    color: #475569;
    font-size: 0.9rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 600;
    color: #475569;
}

.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: var(--page-secondary) !important;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.92);
}

.stAlert {
    border-radius: 12px !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def chart(height: int = 340, **layout_overrides):
    layout = {**CHART_LAYOUT, "height": height}
    layout.update(layout_overrides)
    return layout


def render_chart(fig: go.Figure, caption: str) -> None:
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)


def cumulative_curve(returns: pd.Series) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    curve = (1.0 + returns.fillna(0.0)).cumprod() - 1.0
    curve.name = "cumulative_return"
    return curve


def format_factor_name(name: str) -> str:
    return name.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def load_crypto_summary() -> pd.DataFrame:
    return pd.read_csv(CRYPTO_SAMPLE_DIR / "summary.csv").set_index("symbol")


@st.cache_data(show_spinner=False)
def load_crypto_strategy(symbol: str) -> pd.DataFrame:
    path = CRYPTO_SAMPLE_DIR / f"{symbol.replace('-', '_')}_strategy.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


@st.cache_data(show_spinner=False)
def load_crypto_sample_asset(symbol: str, days: int, vol_window: int, forecast_horizon: int) -> dict:
    market = load_sample_market_data(symbol, days=days)
    returns = market["returns"].dropna()
    prices = market["Close"]
    realised_vol = CryptoDataCollector.calculate_volatility(returns, window=vol_window)
    conditional_vol = market.get("conditional_vol", pd.Series(dtype=float)).dropna()
    forecast = load_sample_forecast(symbol).head(forecast_horizon)
    summary = {}
    crypto_summary = load_crypto_summary()
    if symbol in crypto_summary.index:
        summary = crypto_summary.loc[symbol].to_dict()

    strategy = load_crypto_strategy(symbol)
    benchmark_curve = pd.Series(dtype=float)
    strategy_curve = pd.Series(dtype=float)
    if not strategy.empty:
        aligned_benchmark = returns.loc[returns.index.intersection(strategy.index)]
        if not aligned_benchmark.empty:
            benchmark_curve = cumulative_curve(aligned_benchmark)
        strategy_curve = cumulative_curve(strategy["strategy_returns"])

    return {
        "symbol": symbol,
        "source": "Precomputed sample output",
        "notice": "",
        "prices": prices,
        "returns": returns,
        "realised_vol": realised_vol,
        "conditional_vol": conditional_vol,
        "forecast": forecast,
        "summary": summary,
        "strategy": strategy,
        "strategy_curve": strategy_curve,
        "benchmark_curve": benchmark_curve,
        "price_axis": "Normalized Price Index (base = 100)",
        "price_title": f"{symbol} Demo Price Path",
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_live_crypto_asset(symbol: str, days: int, vol_window: int, forecast_horizon: int) -> dict:
    collector = CryptoDataCollector(symbols=[symbol])
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = collector.fetch_ohlcv(symbol, start=start)
    if df.empty:
        raise ValueError("live data provider returned no rows")

    returns = collector.calculate_returns(df).dropna()
    if returns.empty:
        raise ValueError("no return series available")

    price_column = "Close" if "Close" in df.columns else "close"
    prices = df[price_column]
    realised_vol = CryptoDataCollector.calculate_volatility(returns, window=vol_window)

    conditional_vol = pd.Series(dtype=float)
    forecast = pd.Series(dtype=float)
    diagnostics = {}
    params_table = pd.DataFrame()
    validation_table = pd.DataFrame()

    if len(returns) >= 90:
        scaled_returns = returns * 100.0
        garch = GARCHModel()
        garch.fit(scaled_returns)
        summary = garch.get_model_summary()
        conditional_vol = summary["conditional_volatility"] / 100.0
        forecast = garch.forecast(horizon=forecast_horizon) / 100.0
        diagnostics = {
            "aic": summary["aic"],
            "bic": summary["bic"],
            "log_likelihood": summary["log_likelihood"],
        }
        params_table = pd.DataFrame(
            {
                "Parameter": list(summary["params"].keys()),
                "Estimate": list(summary["params"].values()),
            }
        )

        if len(scaled_returns) >= 272:
            backtest_window = min(252, len(scaled_returns) // 3)
            if len(scaled_returns) >= backtest_window + 20:
                garch_bt, baselines, _ = run_full_backtest(
                    scaled_returns, window=backtest_window, step=5
                )
                rows = [
                    {
                        "Model": "GARCH(1,1)",
                        "RMSE": garch_bt.rmse,
                        "MAE": garch_bt.mae,
                        "Direction Accuracy": garch_bt.direction_accuracy,
                        "Windows": garch_bt.n_windows,
                    }
                ]
                for name, result in baselines.items():
                    rows.append(
                        {
                            "Model": name,
                            "RMSE": result.rmse,
                            "MAE": result.mae,
                            "Direction Accuracy": result.direction_accuracy,
                            "Windows": result.n_windows,
                        }
                    )
                validation_table = pd.DataFrame(rows)

    return {
        "symbol": symbol,
        "source": "Live Yahoo Finance",
        "notice": "",
        "prices": prices,
        "returns": returns,
        "realised_vol": realised_vol,
        "conditional_vol": conditional_vol,
        "forecast": forecast,
        "summary": diagnostics,
        "params_table": params_table,
        "validation_table": validation_table,
        "strategy": pd.DataFrame(),
        "strategy_curve": pd.Series(dtype=float),
        "benchmark_curve": pd.Series(dtype=float),
        "price_axis": "Spot Price (USD)",
        "price_title": f"{symbol} Spot Price",
    }


def get_crypto_asset(symbol: str, use_live: bool, days: int, vol_window: int, forecast_horizon: int) -> dict:
    if use_live:
        try:
            return load_live_crypto_asset(symbol, days, vol_window, forecast_horizon)
        except Exception:
            if sample_data_exists(symbol):
                asset = load_crypto_sample_asset(symbol, days, vol_window, min(forecast_horizon, 10))
                asset["source"] = "Precomputed sample output (live fallback)"
                asset["notice"] = (
                    f"Live Yahoo Finance data was unavailable for {symbol}, so this view fell back to the committed sample output."
                )
                return asset
            return {
                "symbol": symbol,
                "error": "Live data was unavailable and no committed sample output exists for this asset.",
            }
    if not sample_data_exists(symbol):
        return {
            "symbol": symbol,
            "error": "No committed sample output exists for this asset.",
        }
    return load_crypto_sample_asset(symbol, days, vol_window, min(forecast_horizon, 10))


@st.cache_data(show_spinner=False)
def load_factor_outputs() -> dict:
    summary = pd.read_csv(FACTOR_SAMPLE_DIR / "summary.csv").iloc[0]
    metadata = pd.read_csv(FACTOR_SAMPLE_DIR / "run_metadata.csv").iloc[0]
    latest_screen = pd.read_csv(FACTOR_SAMPLE_DIR / "latest_screen.csv").sort_values("screen_rank")
    exposures = pd.read_csv(FACTOR_SAMPLE_DIR / "factor_exposures.csv", index_col=0)["beta"]
    strategy_returns = pd.read_csv(
        FACTOR_SAMPLE_DIR / "strategy_returns.csv", index_col=0, parse_dates=True
    ).sort_index()
    screen_metrics = pd.read_csv(FACTOR_SAMPLE_DIR / "screening_model_metrics.csv").iloc[0]
    model_coefficients = pd.read_csv(
        FACTOR_SAMPLE_DIR / "screening_model_coefficients.csv", index_col=0
    )["coefficient"].sort_values(ascending=False)
    top_ideas_md = (FACTOR_SAMPLE_DIR / "top_quantamental_ideas.md").read_text()
    return {
        "summary": summary,
        "metadata": metadata,
        "latest_screen": latest_screen,
        "exposures": exposures,
        "strategy_returns": strategy_returns,
        "screen_metrics": screen_metrics,
        "model_coefficients": model_coefficients,
        "top_ideas_md": top_ideas_md,
    }


def render_overview_tab() -> None:
    st.markdown(
        """
This repository brings together three applied quant projects: crypto volatility forecasting and risk management, cross-sectional equity factor research, and derivatives pricing. Each project is runnable on its own, and the committed sample artifacts make the full portfolio reviewable without setup.

Taken together, the repo demonstrates end-to-end quantitative engineering: data collection, feature construction, statistical modeling, validation, risk measurement, and presentation in a testable Python codebase.
"""
    )

    cards = st.columns(3)
    cards[0].markdown(
        """
<div class="portfolio-card">
  <h3>Crypto Volatility Risk Engine</h3>
  <p>GARCH-based volatility forecasting, walk-forward validation, VaR/CVaR analytics, and volatility-managed overlays for BTC and ETH research workflows.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        """
<div class="portfolio-card">
  <h3>Quantamental Equity Research Platform</h3>
  <p>Value, momentum, and quality ranking, logistic screening, factor attribution, and thesis generation for a systematic equity research stack.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        """
<div class="portfolio-card">
  <h3>Options Pricing Toolkit</h3>
  <p>Black-Scholes pricing, Greeks, Monte Carlo simulation, and American option trees for derivatives valuation and scenario analysis.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
<div class="source-note">
Use the tabs above to inspect each project. The crypto tab defaults to committed sample data for instant loading, the factor tab uses precomputed sample research artifacts, and the options tab is fully interactive and stateless.
</div>
""",
        unsafe_allow_html=True,
    )


def render_crypto_validation(asset: dict) -> None:
    st.markdown("#### Validation Snapshot")
    summary = asset.get("summary", {})
    validation_table = asset.get("validation_table", pd.DataFrame())

    if not validation_table.empty:
        display_validation = validation_table.copy()
        display_validation["Direction Accuracy"] = display_validation["Direction Accuracy"] * 100.0
        st.dataframe(
            display_validation,
            use_container_width=True,
            hide_index=True,
            column_config={
                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "Direction Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    elif summary:
        snapshot = pd.DataFrame(
            [
                {"Metric": "GARCH OOS RMSE", "Value": f"{summary.get('garch_backtest_rmse', np.nan):.3f}"},
                {
                    "Metric": "Direction Accuracy",
                    "Value": f"{summary.get('garch_backtest_direction_accuracy', np.nan):.1%}",
                },
                {"Metric": "Vol-Managed Sharpe", "Value": f"{summary.get('strategy_sharpe_ratio', np.nan):.2f}"},
                {
                    "Metric": "Vol-Managed Max Drawdown",
                    "Value": f"{summary.get('strategy_max_drawdown', np.nan):.1%}",
                },
                {
                    "Metric": "Average Leverage",
                    "Value": f"{summary.get('strategy_average_leverage', np.nan):.2f}x",
                },
            ]
        )
        st.table(snapshot)
    else:
        st.caption("Validation metrics are unavailable for this live selection.")

    strategy_curve = asset.get("strategy_curve", pd.Series(dtype=float))
    benchmark_curve = asset.get("benchmark_curve", pd.Series(dtype=float))
    if not strategy_curve.empty and not benchmark_curve.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=strategy_curve.index,
                y=strategy_curve.values,
                mode="lines",
                name="Vol-managed overlay",
                line=dict(color=COLORS["primary"], width=2.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                mode="lines",
                name="Buy and hold",
                line=dict(color=COLORS["muted"], width=2.0, dash="dot"),
            )
        )
        fig.update_layout(
            **chart(
                title="Volatility-Managed Overlay vs Buy-and-Hold",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        render_chart(
            fig,
            "Compares the precomputed volatility-targeted overlay with the underlying asset over the same period, showing how forecast-driven sizing changed the return path.",
        )

    params_table = asset.get("params_table", pd.DataFrame())
    if not params_table.empty:
        with st.expander("Live GARCH Diagnostics", expanded=False):
            diag_cols = st.columns(3)
            diag_cols[0].metric("AIC", f"{summary.get('aic', np.nan):.1f}")
            diag_cols[1].metric("BIC", f"{summary.get('bic', np.nan):.1f}")
            diag_cols[2].metric("Log-Likelihood", f"{summary.get('log_likelihood', np.nan):.1f}")
            st.dataframe(
                params_table,
                use_container_width=True,
                hide_index=True,
                column_config={"Estimate": st.column_config.NumberColumn(format="%.6f")},
            )


def render_crypto_asset(asset: dict, vol_window: int, forecast_horizon: int) -> None:
    if asset.get("error"):
        st.warning(asset["error"])
        return

    symbol = asset["symbol"]
    prices = asset["prices"]
    returns = asset["returns"]
    realised_vol = asset["realised_vol"].dropna()
    conditional_vol = asset["conditional_vol"].dropna()
    forecast = asset["forecast"].dropna()
    summary = asset.get("summary", {})

    st.subheader(symbol)
    st.caption(f"Source: {asset['source']}")
    if asset.get("notice"):
        st.caption(asset["notice"])

    annual_vol = returns.std() * np.sqrt(365.0)
    var_95 = calculate_var(returns.values, 0.05)
    current_vol = realised_vol.iloc[-1] if not realised_vol.empty else np.nan
    forecast_last = forecast.iloc[-1] if not forecast.empty else np.nan

    metric_cols = st.columns(5)
    metric_cols[0].metric("Observations", f"{len(returns):,}")
    metric_cols[1].metric("Annualized Vol", f"{annual_vol:.1%}")
    metric_cols[2].metric("VaR (95%)", f"{var_95:.2%}")
    if asset["source"].startswith("Precomputed"):
        metric_cols[3].metric("GARCH OOS RMSE", f"{summary.get('garch_backtest_rmse', np.nan):.3f}")
        metric_cols[4].metric("Forecast End", f"{forecast_last:.2%}")
    else:
        metric_cols[3].metric("Current Realized Vol", f"{current_vol:.2%}")
        metric_cols[4].metric("Forecast End", f"{forecast_last:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                mode="lines",
                line=dict(color=COLORS["primary"], width=2.0),
                fill="tozeroy",
                fillcolor="rgba(15, 118, 110, 0.10)",
            )
        )
        fig.update_layout(
            **chart(
                title=asset["price_title"],
                xaxis_title="Date",
                yaxis_title=asset["price_axis"],
            )
        )
        render_chart(
            fig,
            "Shows the asset price path used in the volatility analysis. In sample mode the price is reconstructed as a normalized index so the dashboard remains read-only and instant to load.",
        )

    with c2:
        fig = go.Figure()
        bar_colors = [COLORS["danger"] if value < 0 else COLORS["success"] for value in returns.values]
        fig.add_trace(
            go.Bar(
                x=returns.index,
                y=returns.values,
                marker_color=bar_colors,
                opacity=0.75,
            )
        )
        fig.update_layout(
            **chart(
                title="Daily Log Returns",
                xaxis_title="Date",
                yaxis_title="Log Return",
            )
        )
        render_chart(
            fig,
            "Highlights the distribution and clustering of positive and negative daily moves, which is the raw input for the volatility and risk models.",
        )

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=realised_vol.index,
                y=realised_vol.values,
                mode="lines",
                name=f"Realized ({vol_window}d)",
                line=dict(color=COLORS["accent"], width=2.0),
            )
        )
        if not conditional_vol.empty:
            fig.add_trace(
                go.Scatter(
                    x=conditional_vol.index,
                    y=conditional_vol.values,
                    mode="lines",
                    name="Conditional",
                    line=dict(color=COLORS["warning"], width=2.0),
                )
            )
        fig.update_layout(
            **chart(
                title="Realized vs Conditional Volatility",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        render_chart(
            fig,
            "Compares observed rolling volatility with the model-implied conditional volatility, showing whether the process captures volatility clustering.",
        )

    with c4:
        fig = go.Figure()
        recent_conditional = conditional_vol.tail(60)
        if not recent_conditional.empty:
            fig.add_trace(
                go.Scatter(
                    x=recent_conditional.index,
                    y=recent_conditional.values,
                    mode="lines",
                    name="Recent conditional vol",
                    line=dict(color=COLORS["muted"], width=1.8),
                )
            )
            bridge_index = [recent_conditional.index[-1]] + list(forecast.index)
            bridge_values = [recent_conditional.iloc[-1]] + list(forecast.values)
            fig.add_trace(
                go.Scatter(
                    x=bridge_index,
                    y=bridge_values,
                    mode="lines+markers",
                    name="Forward forecast",
                    line=dict(color=COLORS["primary"], width=2.6, dash="dot"),
                    marker=dict(size=7),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode="lines+markers",
                    name="Forward forecast",
                    line=dict(color=COLORS["primary"], width=2.6, dash="dot"),
                    marker=dict(size=7),
                )
            )
        fig.update_layout(
            **chart(
                title=f"{forecast_horizon}-Day Volatility Forecast",
                xaxis_title="Date",
                yaxis_title="Forecast Daily Volatility",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        fig.update_yaxes(tickformat=".1%")
        render_chart(
            fig,
            "Shows the near-term volatility forecast used for risk planning and position sizing, bridging the most recent model state into the forward horizon.",
        )

    c5, c6 = st.columns(2)
    cvar_95 = calculate_cvar(returns.values, 0.05)
    with c5:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=70,
                marker_color=COLORS["secondary"],
                opacity=0.72,
            )
        )
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color=COLORS["danger"],
            annotation_text=f"VaR 95%: {var_95:.2%}",
        )
        fig.add_vline(
            x=cvar_95,
            line_dash="dot",
            line_color=COLORS["warning"],
            annotation_text=f"CVaR 95%: {cvar_95:.2%}",
        )
        fig.update_layout(
            **chart(
                title="Return Distribution with VaR and CVaR",
                xaxis_title="Daily Log Return",
                yaxis_title="Observation Count",
                showlegend=False,
            )
        )
        render_chart(
            fig,
            "Plots the empirical return distribution and tail-risk thresholds, showing both the expected cutoff loss and the average loss beyond that cutoff.",
        )

    with c6:
        fig = go.Figure()
        if not realised_vol.empty:
            labels, threshold = detect_regimes(realised_vol)
            low_mask = labels == 0
            high_mask = labels == 1
            fig.add_trace(
                go.Scatter(
                    x=realised_vol.index[low_mask],
                    y=realised_vol.values[low_mask],
                    mode="markers",
                    name="Low-vol regime",
                    marker=dict(color=COLORS["success"], size=4, opacity=0.7),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=realised_vol.index[high_mask],
                    y=realised_vol.values[high_mask],
                    mode="markers",
                    name="High-vol regime",
                    marker=dict(color=COLORS["danger"], size=4, opacity=0.7),
                )
            )
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=COLORS["muted"],
                annotation_text=f"Threshold: {threshold:.1%}",
            )
        fig.update_layout(
            **chart(
                title="Volatility Regime Detection",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        render_chart(
            fig,
            "Classifies the realized volatility history into calmer and stressed regimes, which is useful for explaining risk state changes instead of just reporting a single average.",
        )

    render_crypto_validation(asset)


def render_crypto_tab() -> None:
    st.markdown("### Crypto Volatility Risk Engine")

    controls_left, controls_right = st.columns([1.3, 2.2])
    with controls_left:
        use_live = st.toggle(
            "Use live Yahoo Finance data",
            value=False,
            help="Default sample mode uses committed research artifacts so the portfolio opens instantly.",
        )
        days_back = st.slider("History (days)", 365, 1825, 730, step=30)
        vol_window = st.slider("Volatility window", 7, 90, 30)
        horizon_cap = 30 if use_live else 10
        forecast_horizon = st.slider(
            "Forecast horizon (days)",
            1,
            horizon_cap,
            min(10, horizon_cap),
        )

    with controls_right:
        asset_choices = CRYPTO_LIVE_AVAILABLE if use_live else CRYPTO_SAMPLE_AVAILABLE
        default_symbols = [sym for sym in ["BTC-USD", "ETH-USD"] if sym in asset_choices] or asset_choices[:1]
        symbols = st.multiselect(
            "Assets",
            asset_choices,
            default=default_symbols,
            max_selections=2,
        )
        st.caption(
            "Sample mode is the default so the crypto tab renders from committed artifacts without any network dependency. Live mode keeps the same UI and quietly falls back to sample output if Yahoo Finance is unavailable."
        )

    if not symbols:
        st.info("Select at least one asset to render the crypto research view.")
        return

    with st.spinner("Loading crypto analytics..."):
        assets = [get_crypto_asset(symbol, use_live, days_back, vol_window, forecast_horizon) for symbol in symbols]

    valid_assets = [asset for asset in assets if not asset.get("error")]
    if not valid_assets:
        st.warning("No crypto data could be loaded for the current selection.")
        return

    symbol_tabs = st.tabs([asset["symbol"] for asset in valid_assets])
    for tab, asset in zip(symbol_tabs, valid_assets):
        with tab:
            render_crypto_asset(asset, vol_window=vol_window, forecast_horizon=forecast_horizon)


def render_factor_tab() -> None:
    artifacts = load_factor_outputs()
    summary = artifacts["summary"]
    metadata = artifacts["metadata"]
    screen_metrics = artifacts["screen_metrics"]

    st.markdown("### Quantamental Equity Research Platform")
    st.caption(
        "This tab uses precomputed sample output from `factor_risk_model/output_sample/` so the portfolio remains fully stateless, read-only, and quick to review."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Return", f"{summary['total_return']:.1%}")
    metric_cols[1].metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
    metric_cols[2].metric("Mean IC", f"{summary['mean_information_coefficient']:.3f}")
    metric_cols[3].metric("Holdout Accuracy", f"{screen_metrics['holdout_accuracy']:.1%}")
    metric_cols[4].metric("Holdout ROC AUC", f"{screen_metrics['holdout_roc_auc']:.3f}")

    st.caption(
        f"Backtest window: {metadata['start_date']} to {metadata['end_date']}. "
        f"Backtest universe: {metadata['backtest_universe']} ({int(metadata['backtest_universe_size'])} names). "
        f"Screening universe: {metadata['screening_universe']} ({int(metadata['screening_universe_size'])} names)."
    )

    strategy_returns = artifacts["strategy_returns"].copy()
    strategy_curve = cumulative_curve(strategy_returns["strategy_returns"])
    benchmark_curve = cumulative_curve(strategy_returns["benchmark_returns"])

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=strategy_curve.index,
                y=strategy_curve.values,
                mode="lines",
                name="Long-short strategy",
                line=dict(color=COLORS["primary"], width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                mode="lines",
                name="Equal-weight benchmark",
                line=dict(color=COLORS["muted"], width=2.0, dash="dot"),
            )
        )
        fig.update_layout(
            **chart(
                title="Strategy vs Benchmark Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        render_chart(
            fig,
            "Shows how the long-short factor strategy compounded against the benchmark, which matters because the backtest is the bridge from factor ideas to investable portfolio behavior.",
        )

    with c2:
        exposures = artifacts["exposures"].rename(index=format_factor_name).sort_values()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=exposures.values,
                y=exposures.index,
                orientation="h",
                marker_color=[
                    COLORS["success"] if value >= 0 else COLORS["danger"] for value in exposures.values
                ],
            )
        )
        fig.add_vline(x=0.0, line_color=COLORS["muted"], line_width=1)
        fig.update_layout(
            **chart(
                title="Estimated Factor Exposures",
                xaxis_title="Beta (unitless)",
                yaxis_title="Factor Portfolio",
                showlegend=False,
            )
        )
        render_chart(
            fig,
            "Displays the regression betas of the strategy to the factor-mimicking portfolios, showing whether the realized return stream loaded on the intended signals instead of pure market beta.",
        )

    st.markdown("#### Latest Factor Screen")
    st.caption(
        "Sortable table from the latest screening snapshot. The ranks combine factor scores with the classifier’s outperformance probability."
    )
    latest_screen = artifacts["latest_screen"].copy()
    display_screen = latest_screen[
        [
            "Ticker",
            "name",
            "sector",
            "value_score",
            "momentum_score",
            "quality_score",
            "factor_score",
            "predicted_outperformance_probability",
            "screen_score",
            "screen_rank",
            "currentPrice",
            "targetMeanPrice",
        ]
    ].rename(
        columns={
            "name": "Company",
            "sector": "Sector",
            "value_score": "Value Score",
            "momentum_score": "Momentum Score",
            "quality_score": "Quality Score",
            "factor_score": "Factor Score",
            "predicted_outperformance_probability": "ML Outperf. Prob.",
            "screen_score": "Composite Screen Score",
            "screen_rank": "Screen Rank",
            "currentPrice": "Price (USD)",
            "targetMeanPrice": "Street Target (USD)",
        }
    )
    display_screen["ML Outperf. Prob."] = display_screen["ML Outperf. Prob."] * 100.0
    st.dataframe(
        display_screen,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Value Score": st.column_config.NumberColumn(format="%.2f"),
            "Momentum Score": st.column_config.NumberColumn(format="%.2f"),
            "Quality Score": st.column_config.NumberColumn(format="%.2f"),
            "Factor Score": st.column_config.NumberColumn(format="%.2f"),
            "ML Outperf. Prob.": st.column_config.NumberColumn(format="%.1f%%"),
            "Composite Screen Score": st.column_config.NumberColumn(format="%.2f"),
            "Screen Rank": st.column_config.NumberColumn(format="%d"),
            "Price (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Street Target (USD)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    with st.expander("Classifier Coefficients", expanded=False):
        coeffs = artifacts["model_coefficients"].rename(index=format_factor_name)
        coeff_frame = coeffs.reset_index()
        coeff_frame.columns = ["Feature", "Coefficient"]
        st.dataframe(
            coeff_frame,
            use_container_width=True,
            hide_index=True,
            column_config={"Coefficient": st.column_config.NumberColumn(format="%.3f")},
        )

    st.markdown("#### Top Quantamental Ideas")
    st.caption(
        "Formatted markdown generated by the thesis layer, combining screening rank, valuation context, and a short narrative for the highest-ranked names."
    )
    st.markdown(artifacts["top_ideas_md"])


def render_options_tab() -> None:
    st.markdown("### Options Pricing Toolkit")
    st.caption(
        "This tab is fully interactive and stateless. It calls the repo’s analytical and numerical pricing functions directly, with no external data or API keys."
    )

    controls_col, summary_col = st.columns([1.1, 1.4])
    with controls_col:
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
        spot = st.slider("Spot Price S (USD)", 50.0, 150.0, 100.0, 1.0)
        strike = st.slider("Strike K (USD)", 50.0, 150.0, 100.0, 1.0)
        maturity = st.slider("Maturity T (years)", 0.10, 2.00, 1.00, 0.05)
        rate = st.slider("Risk-Free Rate r (%)", 0.0, 10.0, 3.0, 0.25) / 100.0
        sigma = st.slider("Volatility sigma (%)", 5.0, 80.0, 20.0, 1.0) / 100.0
        with st.expander("Numerical Settings", expanded=False):
            n_paths = st.slider("Monte Carlo Paths", 10_000, 100_000, 50_000, 5_000)
            n_steps = st.slider("Binomial Steps", 50, 400, 200, 25)

    with summary_col:
        try:
            bs_price = black_scholes_price(spot, strike, rate, sigma, maturity, option_type=option_type)
            greeks = black_scholes_greeks(spot, strike, rate, sigma, maturity, option_type=option_type)
            mc = monte_carlo_price(
                spot,
                strike,
                rate,
                sigma,
                maturity,
                option_type=option_type,
                n_paths=n_paths,
                seed=42,
            )
            american = american_option_binomial(
                spot,
                strike,
                rate,
                sigma,
                maturity,
                option_type=option_type,
                n_steps=n_steps,
            )
        except Exception:
            st.warning("The pricing inputs could not be evaluated for this scenario.")
            return

        summary_table = pd.DataFrame(
            [
                {"Metric": "Black-Scholes Price", "Value": bs_price},
                {"Metric": "Delta", "Value": greeks["delta"]},
                {"Metric": "Gamma", "Value": greeks["gamma"]},
                {"Metric": "Vega", "Value": greeks["vega"]},
                {"Metric": "Theta", "Value": greeks["theta"]},
                {"Metric": "Rho", "Value": greeks["rho"]},
            ]
        )
        st.table(summary_table.style.format({"Value": "{:.4f}"}))

        comparison_table = pd.DataFrame(
            [
                {"Method": "Black-Scholes", "Price (USD)": bs_price, "Notes": "Closed-form European price"},
                {
                    "Method": "Monte Carlo",
                    "Price (USD)": mc.price,
                    "Notes": f"95% CI [{mc.confidence_interval_low:.4f}, {mc.confidence_interval_high:.4f}]",
                },
                {
                    "Method": "American Tree",
                    "Price (USD)": american,
                    "Notes": "Cox-Ross-Rubinstein binomial tree",
                },
            ]
        )
        st.markdown("#### Pricing Method Comparison")
        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True,
            column_config={"Price (USD)": st.column_config.NumberColumn(format="$%.4f")},
        )

    c1, c2 = st.columns(2)
    with c1:
        expiry_spots = np.linspace(0.4 * spot, 1.6 * spot, 120)
        if option_type == "call":
            intrinsic = np.maximum(expiry_spots - strike, 0.0)
            breakeven = strike + bs_price
        else:
            intrinsic = np.maximum(strike - expiry_spots, 0.0)
            breakeven = strike - bs_price
        pnl = intrinsic - bs_price

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=expiry_spots,
                y=intrinsic,
                mode="lines",
                name="Intrinsic payoff",
                line=dict(color=COLORS["primary"], width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=expiry_spots,
                y=pnl,
                mode="lines",
                name="Net P/L after premium",
                line=dict(color=COLORS["warning"], width=2.0, dash="dash"),
            )
        )
        fig.add_hline(y=0.0, line_color=COLORS["muted"], line_width=1)
        fig.add_vline(
            x=breakeven,
            line_color=COLORS["danger"],
            line_dash="dot",
            annotation_text=f"Breakeven: {breakeven:.2f}",
        )
        fig.update_layout(
            **chart(
                title="Expiry Payoff and Net P/L",
                xaxis_title="Underlying Price at Expiry (USD)",
                yaxis_title="Payoff / P&L (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        )
        render_chart(
            fig,
            "Shows the convex payoff profile and breakeven of the contract at expiry, which is the clearest way to explain option asymmetry to a reviewer.",
        )

    with c2:
        vol_grid = np.linspace(0.05, 0.80, 60)
        price_vs_vol = [
            black_scholes_price(spot, strike, rate, float(vol), maturity, option_type=option_type)
            for vol in vol_grid
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=vol_grid * 100.0,
                y=price_vs_vol,
                mode="lines",
                line=dict(color=COLORS["secondary"], width=2.4),
            )
        )
        fig.add_vline(
            x=sigma * 100.0,
            line_color=COLORS["primary"],
            line_dash="dot",
            annotation_text=f"Selected sigma: {sigma:.0%}",
        )
        fig.update_layout(
            **chart(
                title="Option Value vs Implied Volatility",
                xaxis_title="Implied Volatility (%)",
                yaxis_title="Option Value (USD)",
            )
        )
        render_chart(
            fig,
            "Shows the vega relationship for the current contract: as implied volatility rises, option value increases because the distribution of possible terminal prices widens.",
        )


st.title("Quant Finance Portfolio")
st.markdown(
    """
<p style="color: #334155; font-size: 1.05rem; margin-top: -10px;">
Three applied projects across market risk, equity factor research, and derivatives pricing, assembled as a portfolio showcase with committed sample artifacts and a stateless Streamlit front end.
</p>
""",
    unsafe_allow_html=True,
)

overview_tab, crypto_tab, factor_tab, options_tab = st.tabs(
    ["Overview", "Crypto Volatility", "Factor Risk Model", "Options Pricing"]
)

with overview_tab:
    render_overview_tab()

with crypto_tab:
    render_crypto_tab()

with factor_tab:
    render_factor_tab()

with options_tab:
    render_options_tab()
