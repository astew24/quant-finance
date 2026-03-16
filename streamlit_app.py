"""
Crypto Volatility Forecasting Dashboard

Wraps the existing GARCH pipeline into an interactive Streamlit app.
Run with: streamlit run streamlit_app.py
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
from crypto_volatility.src.risk_utils import calculate_var

CHART_TEMPLATE = "plotly_dark"
CHART_BG = "rgba(0,0,0,0)"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Volatility Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline Settings")

AVAILABLE_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD"]
symbols = st.sidebar.multiselect(
    "Assets", AVAILABLE_SYMBOLS, default=["BTC-USD", "ETH-USD"],
)
if not symbols:
    symbols = ["BTC-USD"]

days_back = st.sidebar.slider("History (days)", 365, 1825, 730, step=30,
                               help="More history = more observations for training")
vol_window = st.sidebar.slider("Volatility window", 7, 90, 30)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Monitoring**")
alert_threshold = st.sidebar.slider(
    "Vol alert threshold (x avg)", 1.0, 3.0, 1.5, 0.1,
    help="Trigger alert when current vol exceeds this multiple of historical average"
)

run_btn = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Fetches live market data from Yahoo Finance. "
    "Fits GARCH(1,1) to model volatility clustering and "
    "generates forward-looking forecasts."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Crypto Volatility Forecasting")
st.markdown(
    "End-to-end analytics pipeline for BTC/ETH volatility. "
    "Fetches live market data, fits GARCH models, runs out-of-sample evaluation, "
    "and monitors risk with VaR and volatility alerting."
)

# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(sym: str, days: int):
    collector = CryptoDataCollector(symbols=[sym])
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = collector.fetch_ohlcv(sym, start=start)
    if df.empty:
        return df, pd.Series(dtype=float)
    returns = collector.calculate_returns(df).dropna()
    return df, returns


def run_rolling_oos(returns_pct, window=252, step=5):
    """Rolling out-of-sample 1-step GARCH evaluation.
    Steps by `step` days to keep runtime reasonable on Streamlit Cloud."""
    n = len(returns_pct)
    if n < window + 1:
        return None, None, {}

    fc_vals, actual_vals, fc_dates = [], [], []
    indices = range(window, n - 1, step)
    for i in indices:
        train = returns_pct.iloc[i - window:i]
        try:
            g = GARCHModel()
            g.fit(train)
            fc = g.forecast(horizon=1)
            fc_vals.append(fc.values[0])
            # actual next-day squared return as realised vol proxy
            actual_vals.append(abs(returns_pct.iloc[i]))
            fc_dates.append(returns_pct.index[i])
        except Exception:
            continue

    if len(fc_vals) < 10:
        return None, None, {}

    fc_s = pd.Series(fc_vals, index=fc_dates)
    actual_s = pd.Series(actual_vals, index=fc_dates)

    # direction accuracy: did forecast direction match actual move direction?
    if len(fc_s) > 1:
        dir_correct = (np.sign(fc_s.diff().dropna()) == np.sign(actual_s.diff().dropna()))
        dir_acc = dir_correct.mean()
    else:
        dir_acc = np.nan

    rmse = float(np.sqrt(np.mean((fc_s.values - actual_s.values) ** 2)))
    corr = float(np.corrcoef(fc_s.values, actual_s.values)[0, 1]) if len(fc_s) > 2 else np.nan

    metrics = {
        "direction_accuracy": dir_acc,
        "oos_rmse": rmse,
        "oos_correlation": corr,
        "n_windows": len(fc_vals),
    }
    return fc_s, actual_s, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if run_btn:
    all_data = {}
    total_obs = 0

    # fetch all symbols
    progress = st.progress(0, text="Fetching market data...")
    for idx, sym in enumerate(symbols):
        df, returns = fetch_data(sym, days_back)
        if not df.empty:
            all_data[sym] = {"df": df, "returns": returns}
            total_obs += len(df)  # each row is a daily observation (OHLCV)
        progress.progress((idx + 1) / len(symbols), text=f"Fetched {sym}")
    progress.empty()

    if not all_data:
        st.error("Could not fetch data for any symbol. Check your connection.")
        st.stop()

    # -----------------------------------------------------------------------
    # Top-level KPIs
    # -----------------------------------------------------------------------
    st.markdown("---")
    total_data_points = sum(len(d["df"]) * len(d["df"].columns) for d in all_data.values())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets Analyzed", len(all_data))
    k2.metric("Daily Observations", f"{total_obs:,}")
    k3.metric("Total Data Points", f"{total_data_points:,}")
    k4.metric("Forecast Horizon", f"{forecast_horizon}d")

    # -----------------------------------------------------------------------
    # Per-symbol analysis
    # -----------------------------------------------------------------------
    for sym, sdata in all_data.items():
        st.markdown("---")
        st.header(sym)

        df = sdata["df"]
        returns = sdata["returns"]
        col_name = "Close" if "Close" in df.columns else "close"
        prices = df[col_name]
        vol = CryptoDataCollector.calculate_volatility(returns, window=vol_window)

        # fit GARCH
        returns_pct = returns * 100
        with st.spinner(f"Fitting GARCH(1,1) on {sym}..."):
            garch = GARCHModel(p=1, q=1)
            garch.fit(returns_pct)
            summary = garch.get_model_summary()
            cond_vol = summary["conditional_volatility"] / 100
            forecast_pct = garch.forecast(horizon=forecast_horizon)
            forecast = forecast_pct / 100

        ann_vol = returns.std() * np.sqrt(365)
        var_95 = calculate_var(returns.values, 0.05)
        var_99 = calculate_var(returns.values, 0.01)

        # --- KPI row ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observations", f"{len(returns):,}")
        m2.metric("Annualised Vol", f"{ann_vol:.1%}")
        m3.metric("VaR (95%)", f"{var_95:.2%}")
        m4.metric("VaR (99%)", f"{var_99:.2%}")

        # --- Volatility Alerting ---
        rv_recent = vol.dropna()
        if len(rv_recent) > 60:
            current_vol = rv_recent.iloc[-1]
            avg_vol = rv_recent.mean()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

            if vol_ratio >= alert_threshold:
                st.warning(
                    f"**HIGH VOLATILITY ALERT** -- Current vol ({current_vol:.1%}) "
                    f"is {vol_ratio:.1f}x the historical average ({avg_vol:.1%}). "
                    f"Consider reducing position size or tightening stops."
                )
            else:
                st.success(
                    f"Volatility normal -- Current: {current_vol:.1%}, "
                    f"Avg: {avg_vol:.1%} ({vol_ratio:.1f}x)"
                )

        # --- Charts row 1: Price + Returns ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Price")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices.values,
                mode="lines", line=dict(color="#3b82f6", width=1.5),
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="USD", template=CHART_TEMPLATE,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Return Distribution + VaR")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns.values, nbinsx=80,
                marker_color="#3b82f6", opacity=0.7,
            ))
            fig.add_vline(x=var_95, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"VaR 95%: {var_95:.2%}")
            fig.add_vline(x=var_99, line_dash="dash", line_color="#f97316",
                          annotation_text=f"VaR 99%: {var_99:.2%}")
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Daily log return", yaxis_title="Count",
                template=CHART_TEMPLATE, showlegend=False,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Charts row 2: Volatility + Forecast ---
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Realised vs GARCH Conditional Vol")
            fig = go.Figure()
            rv = vol.dropna()
            fig.add_trace(go.Scatter(
                x=rv.index, y=rv.values,
                mode="lines", name=f"Realised ({vol_window}d)",
                line=dict(color="#8b5cf6", width=1.2),
            ))
            fig.add_trace(go.Scatter(
                x=cond_vol.index, y=cond_vol.values,
                mode="lines", name="GARCH conditional",
                line=dict(color="#f59e0b", width=1.2),
            ))
            # alert threshold line
            if len(rv) > 0:
                avg_v = rv.mean()
                fig.add_hline(
                    y=avg_v * alert_threshold, line_dash="dot",
                    line_color="#ef4444", opacity=0.5,
                    annotation_text=f"Alert ({alert_threshold}x avg)",
                )
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Annualised vol", template=CHART_TEMPLATE,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            st.subheader(f"{forecast_horizon}-Day Forecast")
            # bridge from last conditional vol to forecast
            bridge = pd.Series([cond_vol.iloc[-1]], index=[cond_vol.index[-1]])
            fc_ext = pd.concat([bridge, forecast])

            fig = go.Figure()
            tail = cond_vol.iloc[-60:]
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail.values,
                mode="lines", name="Recent conditional",
                line=dict(color="#f59e0b", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fc_ext.index, y=fc_ext.values,
                mode="lines+markers", name="Forecast",
                line=dict(color="#10b981", width=2.5, dash="dot"),
                marker=dict(size=7),
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Daily vol", template=CHART_TEMPLATE,
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Out-of-sample evaluation ---
        with st.expander(f"Out-of-Sample Evaluation ({sym})", expanded=True):
            with st.spinner("Running rolling out-of-sample GARCH..."):
                fc_oos, actual_oos, oos_metrics = run_rolling_oos(
                    returns_pct, window=min(252, len(returns_pct) // 2), step=5
                )

            if oos_metrics:
                o1, o2, o3, o4 = st.columns(4)
                dir_acc = oos_metrics["direction_accuracy"]
                o1.metric("Direction Accuracy", f"{dir_acc:.0%}")
                o2.metric("OOS RMSE", f"{oos_metrics['oos_rmse']:.4f}")
                o3.metric("OOS Correlation", f"{oos_metrics['oos_correlation']:.3f}")
                o4.metric("Eval Windows", f"{oos_metrics['n_windows']}")

                # forecast vs actual scatter
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Forecast vs Actual (OOS)",
                                                    "Rolling Forecast Timeseries"])

                fig.add_trace(go.Scatter(
                    x=actual_oos.values, y=fc_oos.values,
                    mode="markers", marker=dict(color="#3b82f6", size=4, opacity=0.6),
                    name="OOS points",
                ), row=1, col=1)
                # 45-degree line
                mn = min(actual_oos.min(), fc_oos.min())
                mx = max(actual_oos.max(), fc_oos.max())
                fig.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="#ef4444", dash="dash"), name="Perfect fit",
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=actual_oos.index, y=actual_oos.values,
                    mode="lines", name="Actual |return|",
                    line=dict(color="#8b5cf6", width=1),
                ), row=1, col=2)
                fig.add_trace(go.Scatter(
                    x=fc_oos.index, y=fc_oos.values,
                    mode="lines", name="GARCH forecast",
                    line=dict(color="#f59e0b", width=1),
                ), row=1, col=2)

                fig.update_layout(
                    height=320, margin=dict(l=0, r=0, t=30, b=0),
                    template=CHART_TEMPLATE,
                    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.05),
                )
                fig.update_xaxes(title_text="Actual", row=1, col=1)
                fig.update_yaxes(title_text="Forecast", row=1, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for out-of-sample evaluation. Try increasing history.")

        # --- Model details ---
        with st.expander(f"Model Details ({sym})"):
            d1, d2, d3 = st.columns(3)
            d1.metric("AIC", f"{summary['aic']:.1f}")
            d2.metric("BIC", f"{summary['bic']:.1f}")
            d3.metric("Log-Likelihood", f"{summary['log_likelihood']:.1f}")

            # in-sample eval
            common = vol.dropna().index.intersection(cond_vol.index)
            if len(common) > 0:
                actual = vol.loc[common]
                fitted = cond_vol.loc[common]
                rmse_is = calculate_rmse(actual.values, fitted.values)
                corr_is = float(np.corrcoef(actual.values, fitted.values)[0, 1])
                e1, e2 = st.columns(2)
                e1.metric("In-Sample RMSE", f"{rmse_is:.6f}")
                e2.metric("In-Sample Correlation", f"{corr_is:.4f}")

            params = summary["params"]
            st.markdown("**GARCH(1,1) Parameters**")
            st.dataframe(
                pd.DataFrame({"Parameter": list(params.keys()),
                               "Value": [f"{v:.6f}" for v in params.values()]}),
                use_container_width=True, hide_index=True,
            )

            st.markdown("**Forecast Values**")
            st.dataframe(
                pd.DataFrame({"Date": [d.strftime("%Y-%m-%d") for d in forecast.index],
                               "Forecast Vol": [f"{v:.6f}" for v in forecast.values]}),
                use_container_width=True, hide_index=True,
            )

    # -----------------------------------------------------------------------
    # Model comparison across GARCH specs (using first symbol)
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.header("Model Comparison")

    first_sym = list(all_data.keys())[0]
    returns_first = all_data[first_sym]["returns"] * 100

    with st.spinner("Comparing GARCH specifications..."):
        specs = {
            "GARCH(1,1)": (1, 1),
            "GARCH(1,2)": (1, 2),
            "GARCH(2,1)": (2, 1),
            "GARCH(2,2)": (2, 2),
        }
        comp = compare_garch_models(returns_first, specs)

    comp_rows = []
    for name, res in comp.items():
        if "error" in res:
            continue
        comp_rows.append({
            "Model": name,
            "AIC": res["aic"],
            "BIC": res["bic"],
            "Log-Likelihood": res["log_likelihood"],
        })

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        best_aic = comp_df.loc[comp_df["AIC"].idxmin(), "Model"]

        st.dataframe(comp_df.style.format({
            "AIC": "{:.1f}", "BIC": "{:.1f}", "Log-Likelihood": "{:.1f}"
        }), use_container_width=True, hide_index=True)
        st.caption(f"Best model by AIC: **{best_aic}** (lower is better)")

    # multi-symbol forecast comparison
    if len(all_data) > 1:
        st.subheader("Cross-Asset Forecast Comparison")
        fig = go.Figure()
        colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]
        for idx, (sym, sdata) in enumerate(all_data.items()):
            returns_s = sdata["returns"] * 100
            try:
                g = GARCHModel()
                g.fit(returns_s)
                fc = g.forecast(horizon=forecast_horizon)
                fig.add_trace(go.Scatter(
                    x=fc.index, y=(fc / 100).values,
                    mode="lines+markers", name=sym,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=6),
                ))
            except Exception:
                pass
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Forecasted daily vol", xaxis_title="Date",
            template=CHART_TEMPLATE,
            paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # --- Landing page ---
    st.info("Configure settings in the sidebar and click **Run Pipeline** to start.")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        ### Data Pipeline
        Processes thousands of daily OHLCV observations
        from BTC and ETH markets via Yahoo Finance.
        Computes log returns and rolling realised volatility
        across configurable time windows.
        """)

    with c2:
        st.markdown("""
        ### Forecasting Models
        **GARCH(1,1)** captures volatility clustering --
        the tendency for large moves to follow large moves.
        Multiple specifications compared via AIC/BIC.
        Out-of-sample rolling evaluation with direction accuracy.

        LSTM model available in codebase for deep learning comparison.
        """)

    with c3:
        st.markdown("""
        ### Risk Monitoring
        **Value-at-Risk** at 95% and 99% confidence levels.
        Real-time **volatility alerting** flags when current
        vol exceeds a configurable threshold of historical average,
        supporting position sizing and stop-loss decisions.
        """)

    st.markdown("---")
    st.caption(
        "Built with Python, arch, yfinance, scikit-learn, and Streamlit. "
        "Source: github.com/astew24/quant-finance"
    )
