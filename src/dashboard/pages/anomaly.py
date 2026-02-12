"""Anomaly Explorer page for the Streamlit dashboard.

Displays a timeline of detected anomalies with filters for store,
product family, and detection method.  Includes summary metrics and
a detailed table.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils import load_anomaly_report


def render_anomaly_explorer() -> None:
    """Render the Anomaly Explorer page."""
    st.title("Anomaly Explorer")

    anomaly_df = load_anomaly_report()
    if anomaly_df.empty:
        st.warning("No anomaly report found. Run anomaly detection first.")
        return

    # --- Filters ---
    col1, col2, col3 = st.columns(3)

    stores_available = (
        ["All"] + sorted(anomaly_df["store_nbr"].unique().tolist())
        if "store_nbr" in anomaly_df.columns
        else ["All"]
    )
    families_available = (
        ["All"] + sorted(anomaly_df["family"].unique().tolist())
        if "family" in anomaly_df.columns
        else ["All"]
    )
    methods_available = (
        ["All"] + sorted(anomaly_df["method"].unique().tolist())
        if "method" in anomaly_df.columns
        else ["All"]
    )

    with col1:
        store = st.selectbox("Store", options=stores_available)
    with col2:
        family = st.selectbox("Product Family", options=families_available)
    with col3:
        method = st.selectbox("Detection Method", options=methods_available)

    filtered = anomaly_df.copy()
    if store != "All" and "store_nbr" in filtered.columns:
        filtered = filtered[filtered["store_nbr"] == store]
    if family != "All" and "family" in filtered.columns:
        filtered = filtered[filtered["family"] == family]
    if method != "All" and "method" in filtered.columns:
        filtered = filtered[filtered["method"] == method]

    # --- Summary metrics ---
    st.subheader("Anomaly Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Anomalies", len(filtered))
    if "method" in filtered.columns:
        m2.metric("IQR Detected", int((filtered["method"] == "iqr").sum()))
        m3.metric(
            "Isolation Forest",
            int((filtered["method"] == "isolation_forest").sum()),
        )

    # --- Timeline plot ---
    if "date" in filtered.columns and "original_value" in filtered.columns:
        st.subheader("Anomaly Timeline")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["original_value"],
                mode="markers",
                name="Anomalies",
                marker={"size": 10, "color": "red", "symbol": "x"},
            )
        )
        fig.update_layout(
            title="Detected Anomalies",
            xaxis_title="Date",
            yaxis_title="Original Value",
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Detail table ---
    st.subheader("Anomaly Details")
    display_cols = [
        c
        for c in [
            "date",
            "store_nbr",
            "family",
            "original_value",
            "method",
            "action",
            "new_value",
        ]
        if c in filtered.columns
    ]
    if display_cols:
        st.dataframe(
            filtered[display_cols].sort_values("date", ascending=False)
            if "date" in filtered.columns
            else filtered[display_cols],
            use_container_width=True,
        )
    else:
        st.dataframe(filtered, use_container_width=True)
