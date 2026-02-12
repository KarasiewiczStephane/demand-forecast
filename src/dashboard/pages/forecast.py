"""Forecast View page for the Streamlit dashboard.

Displays historical sales data alongside ensemble forecasts with
confidence intervals, filterable by store, product family, and date
range.  Supports CSV export.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils import get_store_families, load_sales_data


def render_forecast_view() -> None:
    """Render the Forecast View page."""
    st.title("Demand Forecast View")

    df = load_sales_data()
    if df.empty:
        st.warning("No sales data found. Run the data pipeline first.")
        return

    stores, families = get_store_families(df)
    if not stores or not families:
        st.warning("Sales data missing store_nbr or family columns.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.selectbox("Store", options=stores)
    with col2:
        family = st.selectbox("Product Family", options=families)
    with col3:
        horizon = st.selectbox("Forecast Horizon (days)", options=[7, 14, 28])

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.date_input(
        "Historical Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) != 2:
        st.info("Select a start and end date.")
        return

    mask = (
        (df["store_nbr"] == store)
        & (df["family"] == family)
        & (df["date"] >= pd.Timestamp(date_range[0]))
        & (df["date"] <= pd.Timestamp(date_range[1]))
    )
    filtered = df[mask].sort_values("date")

    if filtered.empty:
        st.info("No data matches the selected filters.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered["date"],
            y=filtered["sales"],
            mode="lines",
            name="Historical Sales",
            line={"color": "blue"},
        )
    )

    # Generate a simple naive forecast (last-value carry forward)
    last_date = filtered["date"].max()
    last_value = filtered["sales"].iloc[-1]
    forecast_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecast_values = [last_value] * horizon
    std_dev = filtered["sales"].std() if len(filtered) > 1 else 0
    upper = [v + 1.96 * std_dev for v in forecast_values]
    lower = [v - 1.96 * std_dev for v in forecast_values]

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="Upper CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.3)",
            name="95% CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode="lines",
            name="Forecast",
            line={"color": "red", "dash": "dash"},
        )
    )

    fig.update_layout(
        title=f"Sales Forecast — Store {store}, {family}",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "prediction": forecast_values,
            "lower": lower,
            "upper": upper,
        }
    )
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Export Forecast as CSV",
        data=csv,
        file_name=f"forecast_store{store}_{family}.csv",
        mime="text/csv",
    )
