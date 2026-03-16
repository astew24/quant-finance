"""
Crypto Volatility Forecasting -- Interactive Dashboard

Streamlit app that wraps the existing GARCH pipeline.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from crypto_volatility.src.data_collector import CryptoDataCollector
from crypto_volatility.src.garch_model import GARCHModel, compare_garch_models
from crypto_volatility.src.metrics import calculate_rmse
from crypto_volatility.src.risk_utils import calculate_var

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
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

symbol = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD"], index=0)
days_back = st.sidebar.slider("History (days)", 180, 1095, 730, step=30)
vol_window = st.sidebar.slider("Volatility window (days)", 7, 90, 30)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 10)

run_btn = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** fetches real market data from Yahoo Finance, "
    "fits a GARCH(1,1) model to capture volatility clustering, "
    "and generates forward-looking volatility forecasts."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Crypto Volatility Forecasting")
st.markdown(
    "GARCH-based volatility modeling for cryptocurrency markets. "
    "Select an asset and click **Run Pipeline** to fetch live data, "
    "fit the model, and generate forecasts."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(sym: str, days: int):
    collector = CryptoDataCollector(symbols=[sym])
    from datetime import datetime, timedelta
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = collector.fetch_ohlcv(sym, start=start)
    if df.empty:
        return df, pd.Series(dtype=float), pd.Series(dtype=float)
    returns = collector.calculate_returns(df).dropna()
    return df, returns, returns


def fit_garch(returns_pct: pd.Series, horizon: int):
    garch = GARCHModel(p=1, q=1)
    garch.fit(returns_pct)
    summary = garch.get_model_summary()
    forecast = garch.forecast(horizon=horizon)
    return garch, summary, forecast


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner(f"Fetching {symbol} data..."):
        df, returns, _ = fetch_data(symbol, days_back)

    if df.empty:
        st.error("Could not fetch data. Check your network connection and try again.")
        st.stop()

    col_name = "Close" if "Close" in df.columns else "close"
    prices = df[col_name]

    vol = CryptoDataCollector.calculate_volatility(returns, window=vol_window)

    # fit GARCH
    with st.spinner("Fitting GARCH(1,1) model..."):
        returns_pct = returns * 100
        garch, summary, forecast_pct = fit_garch(returns_pct, forecast_horizon)
        cond_vol = summary["conditional_volatility"] / 100
        forecast = forecast_pct / 100

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)

    ann_vol = returns.std() * np.sqrt(365)
    var_95 = calculate_var(returns.values, 0.05)
    var_99 = calculate_var(returns.values, 0.01)

    k1.metric("Observations", f"{len(returns):,}")
    k2.metric("Annualised Volatility", f"{ann_vol:.1%}")
    k3.metric("VaR (95%)", f"{var_95:.2%}")
    k4.metric("VaR (99%)", f"{var_99:.2%}")

    # -----------------------------------------------------------------------
    # Price chart
    # -----------------------------------------------------------------------
    st.subheader(f"{symbol} Price History")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="Close",
        line=dict(color="#3b82f6", width=1.5),
    ))
    fig_price.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="USD", xaxis_title="",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # -----------------------------------------------------------------------
    # Returns distribution
    # -----------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Return Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=returns.values, nbinsx=80,
            marker_color="#3b82f6", opacity=0.7,
            name="Daily returns",
        ))
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="#ef4444",
                           annotation_text=f"VaR 95%: {var_95:.2%}")
        fig_hist.add_vline(x=var_99, line_dash="dash", line_color="#f97316",
                           annotation_text=f"VaR 99%: {var_99:.2%}")
        fig_hist.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Daily log return", yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------------------------------
    # Volatility chart
    # -----------------------------------------------------------------------
    with col_right:
        st.subheader("Volatility: Realised vs GARCH")
        fig_vol = go.Figure()
        rv = vol.dropna()
        fig_vol.add_trace(go.Scatter(
            x=rv.index, y=rv.values,
            mode="lines", name=f"Realised ({vol_window}d)",
            line=dict(color="#8b5cf6", width=1.2),
        ))
        fig_vol.add_trace(go.Scatter(
            x=cond_vol.index, y=cond_vol.values,
            mode="lines", name="GARCH conditional",
            line=dict(color="#f59e0b", width=1.2),
        ))
        fig_vol.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Annualised vol", xaxis_title="",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # -----------------------------------------------------------------------
    # Forecast chart
    # -----------------------------------------------------------------------
    st.subheader(f"{forecast_horizon}-Day Volatility Forecast")

    # connect forecast to last conditional vol point
    last_cond_date = cond_vol.index[-1]
    last_cond_val = cond_vol.iloc[-1]
    bridge = pd.Series([last_cond_val], index=[last_cond_date])
    forecast_extended = pd.concat([bridge, forecast])

    fig_fc = go.Figure()
    # trailing conditional vol for context (last 60 days)
    tail = cond_vol.iloc[-60:]
    fig_fc.add_trace(go.Scatter(
        x=tail.index, y=tail.values,
        mode="lines", name="GARCH conditional (recent)",
        line=dict(color="#f59e0b", width=1.5),
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_extended.index, y=forecast_extended.values,
        mode="lines+markers", name="Forecast",
        line=dict(color="#10b981", width=2.5, dash="dot"),
        marker=dict(size=7),
    ))
    fig_fc.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Daily vol (annualised)", xaxis_title="",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # -----------------------------------------------------------------------
    # Model details
    # -----------------------------------------------------------------------
    st.subheader("Model Details")

    m1, m2, m3 = st.columns(3)
    m1.metric("AIC", f"{summary['aic']:.1f}")
    m2.metric("BIC", f"{summary['bic']:.1f}")
    m3.metric("Log-Likelihood", f"{summary['log_likelihood']:.1f}")

    # evaluation
    common = vol.dropna().index.intersection(cond_vol.index)
    if len(common) > 0:
        actual = vol.loc[common]
        fitted = cond_vol.loc[common]
        rmse = calculate_rmse(actual.values, fitted.values)
        corr = float(np.corrcoef(actual.values, fitted.values)[0, 1])

        e1, e2 = st.columns(2)
        e1.metric("In-Sample RMSE", f"{rmse:.6f}")
        e2.metric("Correlation (realised vs conditional)", f"{corr:.4f}")

    # GARCH parameters table
    params = summary["params"]
    st.markdown("**Estimated GARCH(1,1) Parameters**")
    param_df = pd.DataFrame({
        "Parameter": list(params.keys()),
        "Value": [f"{v:.6f}" for v in params.values()],
    })
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    # forecast table
    st.markdown("**Forecast Values**")
    fc_df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in forecast.index],
        "Forecasted Vol": [f"{v:.6f}" for v in forecast.values],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Model comparison (GARCH specs)
    # -----------------------------------------------------------------------
    with st.expander("GARCH Specification Comparison"):
        specs = {
            "GARCH(1,1)": (1, 1),
            "GARCH(1,2)": (1, 2),
            "GARCH(2,1)": (2, 1),
            "GARCH(2,2)": (2, 2),
        }
        with st.spinner("Comparing GARCH specifications..."):
            comp = compare_garch_models(returns_pct, specs)

        comp_rows = []
        for name, res in comp.items():
            if "error" in res:
                continue
            comp_rows.append({
                "Model": name,
                "AIC": f"{res['aic']:.1f}",
                "BIC": f"{res['bic']:.1f}",
                "Log-Likelihood": f"{res['log_likelihood']:.1f}",
            })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                         hide_index=True)

else:
    # landing state
    st.info("Configure settings in the sidebar and click **Run Pipeline** to start.")

    st.markdown("---")
    st.markdown("### About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What it does**

        Fetches real cryptocurrency market data and fits a GARCH(1,1) model
        to capture volatility clustering -- the tendency for large price
        moves to be followed by more large moves. The model generates
        forward-looking volatility forecasts used in risk management.

        **Data**

        Daily OHLCV from Yahoo Finance (no API key needed). Supports
        BTC-USD and ETH-USD with configurable history length.
        """)
    with col2:
        st.markdown("""
        **Models**

        - **GARCH(1,1)** -- Generalized Autoregressive Conditional
          Heteroskedasticity. Captures time-varying volatility dynamics.
        - Multiple GARCH specifications compared via AIC/BIC.
        - LSTM model available in the codebase (optional, requires TensorFlow).

        **Risk Outputs**

        - Annualised volatility
        - Value-at-Risk at 95% and 99% confidence
        - Multi-day ahead volatility forecasts
        - In-sample model evaluation (RMSE, correlation)
        """)

    st.markdown("---")
    st.markdown(
        "*Built with Python, arch, yfinance, and Streamlit. "
        "Source on [GitHub](https://github.com/astew24/quant-finance).*"
    )
