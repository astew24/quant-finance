"""
Crypto Market Analytics Platform
Interactive dashboard for volatility forecasting, risk monitoring, and market insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from crypto_volatility.src.data_collector import CryptoDataCollector
from crypto_volatility.src.garch_model import GARCHModel, compare_garch_models
from crypto_volatility.src.metrics import calculate_rmse
from crypto_volatility.src.risk_utils import calculate_var, calculate_cvar, detect_regimes
from crypto_volatility.src.backtesting import run_full_backtest
from crypto_volatility.src.demo_data import available_sample_symbols, load_sample_market_data, sample_data_exists

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Market Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#6366f1",    # indigo
    "secondary": "#8b5cf6",  # violet
    "accent": "#06b6d4",     # cyan
    "success": "#10b981",    # emerald
    "warning": "#f59e0b",    # amber
    "danger": "#ef4444",     # red
    "muted": "#94a3b8",      # slate
    "bg_card": "#1e293b",    # slate-800
    "text": "#e2e8f0",       # slate-200
}

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

LIVE_AVAILABLE = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]
SAMPLE_AVAILABLE = available_sample_symbols()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Header */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: #e2e8f0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0f172a;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1e293b;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 0.85rem;
    color: #94a3b8;
}

.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #e2e8f0 !important;
    background: #1e293b !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 10px 24px !important;
}

/* Alert boxes */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Settings")

data_mode = st.sidebar.radio(
    "Data source",
    ["Auto", "Live", "Demo sample"],
    index=0,
    help="Auto tries live Yahoo Finance first and falls back to committed sample data when available.",
)

asset_choices = SAMPLE_AVAILABLE if data_mode == "Demo sample" and SAMPLE_AVAILABLE else LIVE_AVAILABLE
default_symbols = [sym for sym in ["BTC-USD", "ETH-USD"] if sym in asset_choices] or asset_choices[:1]
symbols = st.sidebar.multiselect("Assets", asset_choices, default=default_symbols)
if not symbols:
    symbols = default_symbols[:1]

days_back = st.sidebar.slider("History (days)", 365, 1825, 730, step=30)
vol_window = st.sidebar.slider("Volatility window", 7, 90, 30)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 10)
alert_mult = st.sidebar.slider("Alert threshold (x avg vol)", 1.0, 3.0, 1.5, 0.1)

run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
if data_mode == "Demo sample":
    if SAMPLE_AVAILABLE:
        st.sidebar.caption(
            "Uses committed sample outputs for an offline, presentation-safe run. "
            "BTC and ETH demo data are bundled in the repo."
        )
    else:
        st.sidebar.error("No committed sample data was found in the repository.")
elif data_mode == "Live":
    st.sidebar.caption(
        "Fetches live data from Yahoo Finance. "
        "No API key required."
    )
else:
    st.sidebar.caption(
        "Tries live Yahoo Finance first, then falls back to committed sample data when available."
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch(sym, days, mode):
    def load_sample():
        sample = load_sample_market_data(sym, days=days)
        return sample, sample["returns"].dropna()

    if mode == "Demo sample":
        if not sample_data_exists(sym):
            return pd.DataFrame(), pd.Series(dtype=float), "Unavailable", f"No committed sample data exists for {sym}."
        df, returns = load_sample()
        return df, returns, "Demo sample", "Loaded committed sample data from crypto_volatility/output_sample."

    collector = CryptoDataCollector(symbols=[sym])
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = collector.fetch_ohlcv(sym, start=start)
        if not df.empty:
            return df, collector.calculate_returns(df).dropna(), "Live Yahoo Finance", ""
        live_error = "Yahoo Finance returned no rows."
    except Exception as exc:
        live_error = str(exc)

    if mode == "Auto" and sample_data_exists(sym):
        df, returns = load_sample()
        return (
            df,
            returns,
            "Demo sample fallback",
            f"Live fetch failed for {sym}; using committed sample data instead ({live_error}).",
        )

    return pd.DataFrame(), pd.Series(dtype=float), "Unavailable", f"Unable to load {sym}: {live_error}"


def chart(height=320, **kw):
    layout = {**CHART_LAYOUT, "height": height}
    layout.update(kw)
    return layout


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Crypto Market Analytics Platform")
st.markdown(
    '<p style="color: #94a3b8; font-size: 1.05rem; margin-top: -10px;">'
    'Volatility forecasting, risk monitoring, and market insights for crypto assets'
    '</p>', unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not run:
    # Landing page
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("""
    #### Data Pipeline
    Processes thousands of daily OHLCV observations across BTC, ETH,
    and altcoin markets. Computes log returns and rolling realised volatility.
    """)
    c2.markdown("""
    #### Forecasting
    GARCH(1,1) captures volatility clustering with walk-forward
    backtesting against naive baselines. Statistical significance
    verified via Diebold-Mariano test.
    """)
    c3.markdown("""
    #### Risk Monitoring
    Value-at-Risk and Expected Shortfall (CVaR) at configurable
    confidence levels. Real-time volatility alerting flags regime
    changes for position sizing decisions.
    """)
    c4.markdown("""
    #### Model Validation
    Out-of-sample direction accuracy, RMSE comparison across
    models, and regime-aware performance analysis. No fake
    claims -- every metric is computed live.
    """)

    st.markdown("---")
    st.info("Select assets in the sidebar and click **Run Analysis** to start the pipeline.")
    st.caption(
        "Built with Python, arch, yfinance, scikit-learn, and Streamlit. "
        "Use `Demo sample` in the sidebar for an offline-safe walkthrough."
    )
    st.stop()

# Fetch data for all symbols
all_data = {}
total_rows = 0
load_messages = []
progress = st.progress(0, text="Fetching data...")
for i, sym in enumerate(symbols):
    df, rets, source, message = fetch(sym, days_back, data_mode)
    if not df.empty:
        all_data[sym] = {"df": df, "returns": rets, "source": source}
        total_rows += len(df)
    if message:
        load_messages.append(message)
    progress.progress((i + 1) / len(symbols), text=f"Loaded {sym}")
progress.empty()

if not all_data:
    error_message = "Could not load any data."
    if load_messages:
        error_message = f"{error_message} {' '.join(load_messages)}"
    st.error(error_message)
    st.stop()

total_pts = sum(len(d["df"]) * len(d["df"].columns) for d in all_data.values())
date_min = min(d["df"].index.min() for d in all_data.values())
date_max = max(d["df"].index.max() for d in all_data.values())
sample_symbols = [sym for sym, sd in all_data.items() if sd["source"] != "Live Yahoo Finance"]

if sample_symbols:
    joined = ", ".join(sample_symbols)
    st.warning(
        f"Using committed sample data for {joined}. "
        "Sample-backed price charts are normalized to a base value of 100 for offline demos."
    )
elif load_messages:
    st.info(" ".join(load_messages))

# ===== TABS =====
tab_overview, tab_vol, tab_risk, tab_backtest, tab_method = st.tabs([
    "Market Overview",
    "Volatility Analysis",
    "Risk Monitoring",
    "Model Validation",
    "Methodology",
])

# =========================================================================
# TAB 1: MARKET OVERVIEW
# =========================================================================
with tab_overview:
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets", len(all_data))
    k2.metric("Daily Observations", f"{total_rows:,}")
    k3.metric("Data Points", f"{total_pts:,}")
    k4.metric("Date Range", f"{date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}")

    # Price charts
    for sym, sd in all_data.items():
        df = sd["df"]
        source = sd["source"]
        col = "Close" if "Close" in df.columns else "close"
        prices = df[col]
        rets = sd["returns"]
        price_axis = "USD" if source == "Live Yahoo Finance" else "Normalized Index (base = 100)"
        price_title = f"{sym} Price" if source == "Live Yahoo Finance" else f"{sym} Demo Price Path"

        st.subheader(sym)
        st.caption(f"Source: {source}")
        pc, rc = st.columns([3, 2])

        with pc:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices.values,
                mode="lines", line=dict(color=COLORS["primary"], width=1.5),
                fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.08)",
            ))
            fig.update_layout(**chart(yaxis_title=price_axis, title=price_title))
            st.plotly_chart(fig, use_container_width=True)

        with rc:
            fig = go.Figure()
            colors_bar = [COLORS["danger"] if r < 0 else COLORS["success"]
                          for r in rets.values]
            fig.add_trace(go.Bar(
                x=rets.index, y=rets.values,
                marker_color=colors_bar, opacity=0.7,
            ))
            fig.update_layout(**chart(yaxis_title="Return", title="Daily Log Returns"))
            st.plotly_chart(fig, use_container_width=True)

        # quick stats
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Mean Return", f"{rets.mean():.4%}")
        s2.metric("Annualised Vol", f"{rets.std() * np.sqrt(365):.1%}")
        s3.metric("Sharpe (ann.)", f"{rets.mean() / rets.std() * np.sqrt(365):.2f}"
                  if rets.std() > 0 else "N/A")
        s4.metric("Max Drawdown", f"{(prices / prices.cummax() - 1).min():.1%}")
        s5.metric("Skewness", f"{float(rets.skew()):.2f}")


# =========================================================================
# TAB 2: VOLATILITY ANALYSIS
# =========================================================================
with tab_vol:
    st.markdown("---")

    for sym, sd in all_data.items():
        st.subheader(sym)
        rets = sd["returns"]
        vol = CryptoDataCollector.calculate_volatility(rets, window=vol_window)

        returns_pct = rets * 100
        with st.spinner(f"Fitting GARCH(1,1) on {sym}..."):
            garch = GARCHModel()
            garch.fit(returns_pct)
            summary = garch.get_model_summary()
            cond_vol = summary["conditional_volatility"] / 100
            fc_pct = garch.forecast(horizon=forecast_horizon)
            forecast = fc_pct / 100

        # Vol overlay
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            rv = vol.dropna()
            fig.add_trace(go.Scatter(
                x=rv.index, y=rv.values,
                mode="lines", name=f"Realised ({vol_window}d)",
                line=dict(color=COLORS["secondary"], width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=cond_vol.index, y=cond_vol.values,
                mode="lines", name="GARCH conditional",
                line=dict(color=COLORS["warning"], width=1.2),
            ))
            fig.update_layout(**chart(
                yaxis_title="Annualised vol",
                title="Realised vs Conditional Volatility",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Forecast
            bridge = pd.Series([cond_vol.iloc[-1]], index=[cond_vol.index[-1]])
            fc_line = pd.concat([bridge, forecast])
            tail = cond_vol.iloc[-60:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail.values,
                mode="lines", name="Recent",
                line=dict(color=COLORS["warning"], width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fc_line.index, y=fc_line.values,
                mode="lines+markers", name="Forecast",
                line=dict(color=COLORS["success"], width=2.5, dash="dot"),
                marker=dict(size=7),
            ))
            fig.update_layout(**chart(
                yaxis_title="Daily vol",
                title=f"{forecast_horizon}-Day Forecast",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Regime detection
        labels, threshold = detect_regimes(vol.dropna())
        regime_pct_high = labels.mean()

        # Model info
        with st.expander(f"GARCH Parameters & Diagnostics ({sym})", expanded=False):
            p1, p2, p3 = st.columns(3)
            p1.metric("AIC", f"{summary['aic']:.1f}")
            p2.metric("BIC", f"{summary['bic']:.1f}")
            p3.metric("Log-Likelihood", f"{summary['log_likelihood']:.1f}")

            params = summary["params"]
            st.dataframe(
                pd.DataFrame({"Parameter": params.keys(), "Estimate": [f"{v:.6f}" for v in params.values()]}),
                use_container_width=True, hide_index=True,
            )

            # persistence
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)
            persistence = alpha + beta
            st.metric("Volatility Persistence (alpha + beta)", f"{persistence:.4f}",
                      help="Values close to 1.0 indicate highly persistent volatility (slow mean-reversion)")

            st.dataframe(
                pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in forecast.index],
                               "Forecast Vol": [f"{v:.6f}" for v in forecast.values]}),
                use_container_width=True, hide_index=True,
            )

    # GARCH spec comparison
    st.markdown("---")
    st.subheader("GARCH Specification Comparison")
    first_sym = list(all_data.keys())[0]
    r0 = all_data[first_sym]["returns"] * 100
    with st.spinner("Comparing models..."):
        comp = compare_garch_models(r0, {
            "GARCH(1,1)": (1, 1), "GARCH(1,2)": (1, 2),
            "GARCH(2,1)": (2, 1), "GARCH(2,2)": (2, 2),
        })
    rows = []
    for name, res in comp.items():
        if "error" not in res:
            rows.append({"Model": name, "AIC": res["aic"], "BIC": res["bic"],
                          "Log-Likelihood": res["log_likelihood"]})
    if rows:
        cdf = pd.DataFrame(rows)
        best = cdf.loc[cdf["AIC"].idxmin(), "Model"]
        st.dataframe(cdf.style.format({"AIC": "{:.1f}", "BIC": "{:.1f}", "Log-Likelihood": "{:.1f}"}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Best model by AIC: **{best}** (lower = better)")


# =========================================================================
# TAB 3: RISK MONITORING
# =========================================================================
with tab_risk:
    st.markdown("---")

    for sym, sd in all_data.items():
        st.subheader(f"{sym} Risk Dashboard")
        rets = sd["returns"]
        vol = CryptoDataCollector.calculate_volatility(rets, window=vol_window)

        var_95 = calculate_var(rets.values, 0.05)
        var_99 = calculate_var(rets.values, 0.01)
        cvar_95 = calculate_cvar(rets.values, 0.05)
        cvar_99 = calculate_cvar(rets.values, 0.01)

        # KPIs
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR (95%)", f"{var_95:.2%}")
        r2.metric("CVaR (95%)", f"{cvar_95:.2%}",
                  help="Expected Shortfall: average loss on days beyond VaR")
        r3.metric("VaR (99%)", f"{var_99:.2%}")
        r4.metric("CVaR (99%)", f"{cvar_99:.2%}")

        c1, c2 = st.columns(2)

        # Return distribution with VaR/CVaR
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=rets.values, nbinsx=80,
                marker_color=COLORS["primary"], opacity=0.7,
            ))
            fig.add_vline(x=var_95, line_dash="dash", line_color=COLORS["danger"],
                          annotation_text=f"VaR 95%: {var_95:.2%}")
            fig.add_vline(x=cvar_95, line_dash="dot", line_color=COLORS["warning"],
                          annotation_text=f"CVaR 95%: {cvar_95:.2%}")
            fig.update_layout(**chart(
                xaxis_title="Daily return", yaxis_title="Count",
                title="Return Distribution with Risk Thresholds",
                showlegend=False,
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Volatility regime detection
        with c2:
            rv = vol.dropna()
            labels, thresh = detect_regimes(rv)

            fig = go.Figure()
            low_mask = labels == 0
            high_mask = labels == 1

            fig.add_trace(go.Scatter(
                x=rv.index[low_mask], y=rv.values[low_mask],
                mode="markers", name="Low vol regime",
                marker=dict(color=COLORS["success"], size=3, opacity=0.6),
            ))
            fig.add_trace(go.Scatter(
                x=rv.index[high_mask], y=rv.values[high_mask],
                mode="markers", name="High vol regime",
                marker=dict(color=COLORS["danger"], size=3, opacity=0.6),
            ))
            fig.add_hline(y=thresh, line_dash="dash", line_color=COLORS["muted"],
                          annotation_text=f"Regime threshold: {thresh:.1%}")

            avg_vol = rv.mean()
            fig.add_hline(y=avg_vol * alert_mult, line_dash="dot",
                          line_color=COLORS["warning"],
                          annotation_text=f"Alert ({alert_mult}x avg)")

            fig.update_layout(**chart(
                yaxis_title="Vol", title="Volatility Regime Detection",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ))
            st.plotly_chart(fig, use_container_width=True)

        # Alert status
        current_vol = rv.iloc[-1] if len(rv) > 0 else 0
        avg_vol = rv.mean() if len(rv) > 0 else 1
        ratio = current_vol / avg_vol if avg_vol > 0 else 0
        current_regime = "High" if labels.iloc[-1] == 1 else "Low"

        a1, a2, a3 = st.columns(3)
        a1.metric("Current Vol", f"{current_vol:.1%}")
        a2.metric("Vol Regime", current_regime)
        a3.metric("vs Average", f"{ratio:.1f}x")

        if ratio >= alert_mult:
            st.warning(
                f"**ELEVATED VOLATILITY** -- {sym} current vol ({current_vol:.1%}) "
                f"is {ratio:.1f}x historical average. Consider reducing exposure or "
                f"tightening risk limits."
            )
        else:
            st.success(
                f"**Normal conditions** -- {sym} vol at {ratio:.1f}x average. "
                f"Within acceptable range."
            )

        # Tail analysis
        with st.expander(f"Tail Risk Analysis ({sym})"):
            # worst days
            worst = rets.nsmallest(10)
            st.markdown("**10 Worst Daily Returns**")
            st.dataframe(
                pd.DataFrame({"Date": worst.index.strftime("%Y-%m-%d"),
                               "Return": [f"{v:.2%}" for v in worst.values]}),
                use_container_width=True, hide_index=True,
            )

            # VaR exceedances
            exceedances = (rets < var_95).sum()
            expected = len(rets) * 0.05
            st.markdown(
                f"**VaR Backtest:** {exceedances} days exceeded 95% VaR "
                f"(expected ~{expected:.0f} out of {len(rets)} days). "
                f"{'Model is well-calibrated.' if abs(exceedances - expected) / expected < 0.3 else 'Model may need recalibration.'}"
            )


# =========================================================================
# TAB 4: MODEL VALIDATION
# =========================================================================
with tab_backtest:
    st.markdown("---")

    for sym, sd in all_data.items():
        st.subheader(f"{sym} Walk-Forward Backtest")

        rets = sd["returns"]
        returns_pct = rets * 100
        bt_window = min(252, len(returns_pct) // 3)

        if len(returns_pct) < bt_window + 20:
            st.warning(f"Not enough data for {sym} backtest. Need more history.")
            continue

        with st.spinner(f"Running walk-forward backtest for {sym} (this takes a moment)..."):
            try:
                garch_bt, baselines, dm_tests = run_full_backtest(
                    returns_pct, window=bt_window, step=5
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                continue

        # Performance comparison table
        all_models = {"GARCH(1,1)": garch_bt, **baselines}
        perf_rows = []
        for name, res in all_models.items():
            perf_rows.append({
                "Model": name,
                "RMSE": res.rmse,
                "MAE": res.mae,
                "Direction Acc.": res.direction_accuracy,
                "Windows": res.n_windows,
            })

        perf_df = pd.DataFrame(perf_rows)
        best_rmse_model = perf_df.loc[perf_df["RMSE"].idxmin(), "Model"]

        st.dataframe(
            perf_df.style.format({
                "RMSE": "{:.4f}", "MAE": "{:.4f}",
                "Direction Acc.": "{:.1%}",
            }).highlight_min(subset=["RMSE", "MAE"], color="#065f46")
             .highlight_max(subset=["Direction Acc."], color="#065f46"),
            use_container_width=True, hide_index=True,
        )

        # Key metrics callout
        b1, b2, b3 = st.columns(3)
        b1.metric("GARCH Direction Accuracy", f"{garch_bt.direction_accuracy:.0%}")
        b2.metric("Best Model (RMSE)", best_rmse_model)
        b3.metric("OOS Eval Windows", garch_bt.n_windows)

        # Diebold-Mariano tests
        if dm_tests:
            st.markdown("**Diebold-Mariano Test (GARCH vs Baselines)**")
            dm_rows = []
            for name, (stat, pval) in dm_tests.items():
                if np.isnan(stat):
                    continue
                sig = "Yes" if pval < 0.05 else "No"
                better = "GARCH" if stat < 0 else name
                dm_rows.append({
                    "Comparison": f"GARCH vs {name}",
                    "DM Statistic": stat,
                    "p-value": pval,
                    "Significant (5%)": sig,
                    "Better Model": better,
                })
            if dm_rows:
                st.dataframe(
                    pd.DataFrame(dm_rows).style.format(
                        {"DM Statistic": "{:.3f}", "p-value": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Negative DM statistic = GARCH has smaller loss. "
                    "p < 0.05 means the difference is statistically significant."
                )

        # OOS charts
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=garch_bt.actuals.values, y=garch_bt.forecasts.values,
                mode="markers", marker=dict(color=COLORS["primary"], size=4, opacity=0.5),
            ))
            mn = min(garch_bt.actuals.min(), garch_bt.forecasts.min())
            mx = max(garch_bt.actuals.max(), garch_bt.forecasts.max())
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color=COLORS["danger"], dash="dash", width=1),
                showlegend=False,
            ))
            fig.update_layout(**chart(
                title="Forecast vs Actual (OOS)",
                xaxis_title="Actual |return| (%)",
                yaxis_title="GARCH forecast (%)",
            ))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=garch_bt.actuals.index, y=garch_bt.actuals.values,
                mode="lines", name="Actual |return|",
                line=dict(color=COLORS["secondary"], width=1),
            ))
            fig.add_trace(go.Scatter(
                x=garch_bt.forecasts.index, y=garch_bt.forecasts.values,
                mode="lines", name="GARCH forecast",
                line=dict(color=COLORS["warning"], width=1),
            ))
            fig.update_layout(**chart(
                title="Rolling OOS Timeseries",
                yaxis_title="Volatility (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ))
            st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# TAB 5: METHODOLOGY
# =========================================================================
with tab_method:
    st.markdown("---")

    st.markdown("""
    ### Approach

    This platform implements a complete volatility forecasting pipeline for cryptocurrency
    markets. The core methodology uses **GARCH(1,1)** (Generalized Autoregressive Conditional
    Heteroskedasticity), a standard model in financial econometrics for capturing
    **volatility clustering** -- the empirical tendency for large price moves to be
    followed by more large moves.

    ### Data

    Daily OHLCV data sourced from Yahoo Finance. Log returns are computed as
    r_t = ln(P_t / P_{t-1}). Annualised volatility uses a sqrt(365) scaling factor
    since crypto markets trade 24/7/365, unlike equities (sqrt(252)).

    For demos, the app can also load committed sample outputs from `crypto_volatility/output_sample`.
    In that mode, the price chart is reconstructed as a normalized index so the dashboard remains
    usable without a live network dependency.

    ### GARCH(1,1) Model

    The conditional variance follows:

    > sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Where:
    - **omega**: long-run variance floor
    - **alpha**: reaction to recent shocks (higher = more responsive)
    - **beta**: persistence of past variance (higher = slower decay)
    - **alpha + beta**: persistence parameter (close to 1.0 = highly persistent)

    Returns are scaled to percentage for numerical stability during optimisation.
    Model selection across GARCH(p,q) specifications uses AIC/BIC.

    ### Walk-Forward Backtesting

    At each step t, the model is fit on a rolling window of past returns and produces
    a 1-step-ahead volatility forecast. This forecast is compared to the realised
    volatility proxy (|r_t|). Three naive baselines provide context:

    - **Historical volatility**: rolling standard deviation
    - **EWMA** (lambda=0.94): exponentially weighted moving average (RiskMetrics approach)
    - **Random walk**: yesterday's absolute return

    The **Diebold-Mariano test** checks whether GARCH's forecast improvement
    over each baseline is statistically significant (H0: equal predictive accuracy).

    ### Risk Metrics

    - **Value-at-Risk (VaR)**: the q-th percentile of the return distribution.
      At 95%, it answers "what is the worst daily loss we expect to see 19 out of 20 days?"
    - **Expected Shortfall (CVaR)**: the average loss conditional on exceeding VaR.
      More informative than VaR because it captures tail severity, not just frequency.
      CVaR is a **coherent risk measure** (sub-additive), while VaR is not.
    - **VaR backtest**: counts actual exceedances vs expected to check model calibration.

    ### Regime Detection

    A simple threshold-based approach splits the volatility series into low and high
    regimes using the median as boundary. This helps identify market regime shifts
    relevant to settlement risk and position sizing.

    ### Limitations

    - GARCH assumes normally distributed innovations; crypto returns have fat tails.
      Extensions like GJR-GARCH or GARCH-t would better capture asymmetric tail risk.
    - The random walk baseline is hard to beat consistently in short samples.
    - Regime detection is threshold-based, not a proper Markov-switching model.
    - Yahoo Finance data may have gaps or lag behind real-time exchange data.
    - No intraday or orderbook-level analysis.

    ### Tech Stack

    Python, arch (GARCH), yfinance, scikit-learn, SciPy (statistical tests),
    NumPy/Pandas, Plotly, Streamlit.
    """)

    st.markdown("---")
    st.caption("Source: github.com/astew24/quant-finance")
