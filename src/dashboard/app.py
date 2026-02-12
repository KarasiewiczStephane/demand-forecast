"""Streamlit multi-page dashboard for the demand forecasting system.

Provides navigation between Forecast View, Model Comparison,
Anomaly Explorer, Seasonality Patterns, and Backtesting Results.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="\U0001f4c8",
    layout="wide",
)

PAGES = [
    "Forecast View",
    "Model Comparison",
    "Anomaly Explorer",
    "Seasonality Patterns",
    "Backtesting Results",
]

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", PAGES)

if page == "Forecast View":
    from src.dashboard.pages.forecast import render_forecast_view

    render_forecast_view()

elif page == "Model Comparison":
    from src.dashboard.pages.comparison import render_model_comparison

    render_model_comparison()

elif page == "Anomaly Explorer":
    from src.dashboard.pages.anomaly import render_anomaly_explorer

    render_anomaly_explorer()

elif page == "Seasonality Patterns":
    from src.dashboard.pages.seasonality import render_seasonality_patterns

    render_seasonality_patterns()

elif page == "Backtesting Results":
    from src.dashboard.pages.backtesting import render_backtesting_results

    render_backtesting_results()
