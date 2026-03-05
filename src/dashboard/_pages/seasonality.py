"""Seasonality Patterns page for the Streamlit dashboard.

Displays seasonal decomposition (trend, seasonal, residual) and
ACF/PACF bar charts for a selected store-family combination.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from src.dashboard.utils import get_store_families, load_sales_data


def render_seasonality_patterns() -> None:
    """Render the Seasonality Patterns page."""
    st.title("Seasonality Patterns")

    df = load_sales_data()
    if df.empty:
        st.warning("No sales data found. Run the data pipeline first.")
        return

    stores, families = get_store_families(df)
    if not stores or not families:
        st.warning("Sales data missing store_nbr or family columns.")
        return

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store", options=stores)
    with col2:
        family = st.selectbox("Product Family", options=families)

    ts = df[(df["store_nbr"] == store) & (df["family"] == family)].copy()
    if ts.empty:
        st.info("No data for the selected combination.")
        return

    ts = ts.set_index("date")["sales"].sort_index()
    ts = ts.asfreq("D")
    ts = ts.ffill()

    if len(ts) < 14:
        st.info("Not enough data points for seasonal decomposition.")
        return

    # --- Seasonal decomposition ---
    period = min(7, len(ts) // 2)
    decomposition = seasonal_decompose(ts, model="additive", period=period)

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    fig.add_trace(
        go.Scatter(x=ts.index, y=ts.values, name="Original"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend.values,
            name="Trend",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal.values,
            name="Seasonal",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid.values,
            name="Residual",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(height=800, title="Seasonal Decomposition")
    st.plotly_chart(fig, use_container_width=True)

    # --- ACF / PACF ---
    st.subheader("Autocorrelation Analysis")
    col_a, col_b = st.columns(2)

    max_lags = min(50, len(ts) // 2 - 1)
    if max_lags < 2:
        st.info("Not enough data for ACF / PACF.")
        return

    acf_vals = acf(ts.dropna(), nlags=max_lags)
    pacf_lags = min(max_lags, len(ts) // 4)
    pacf_vals = pacf(ts.dropna(), nlags=max(1, pacf_lags))

    ci = 1.96 / np.sqrt(len(ts))

    with col_a:
        acf_fig = go.Figure()
        acf_fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"))
        acf_fig.add_hline(y=ci, line_dash="dash", line_color="red")
        acf_fig.add_hline(y=-ci, line_dash="dash", line_color="red")
        acf_fig.update_layout(title="ACF")
        st.plotly_chart(acf_fig, use_container_width=True)

    with col_b:
        pacf_fig = go.Figure()
        pacf_fig.add_trace(
            go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF")
        )
        pacf_fig.add_hline(y=ci, line_dash="dash", line_color="red")
        pacf_fig.add_hline(y=-ci, line_dash="dash", line_color="red")
        pacf_fig.update_layout(title="PACF")
        st.plotly_chart(pacf_fig, use_container_width=True)
