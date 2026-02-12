"""Model Comparison page for the Streamlit dashboard.

Shows a side-by-side metrics table across Prophet, LSTM, XGBoost, and
Ensemble, plus error distribution histograms and box plots.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils import load_backtest_report


def render_model_comparison() -> None:
    """Render the Model Comparison page."""
    st.title("Model Comparison")

    report_df = load_backtest_report()
    if report_df.empty:
        st.warning(
            "No backtest report found. Run backtesting first to populate results."
        )
        return

    # --- Aggregate metrics table ---
    st.subheader("Performance Metrics (mean across windows)")

    metric_cols = [
        c for c in report_df.columns if c in ("mape", "rmse", "mae", "smape")
    ]
    if metric_cols and "model" in report_df.columns:
        agg = report_df.groupby("model")[metric_cols].mean().reset_index()
        st.dataframe(
            agg.style.highlight_min(subset=metric_cols, color="lightgreen"),
            use_container_width=True,
        )
    else:
        st.info("Report does not contain expected metric columns.")

    # --- Error distribution ---
    if "rmse" in report_df.columns and "model" in report_df.columns:
        st.subheader("Error Distributions")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                report_df,
                x="rmse",
                color="model",
                nbins=30,
                title="RMSE Distribution by Model",
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = px.box(
                report_df,
                x="model",
                y="rmse",
                title="RMSE Box Plot by Model",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # --- Rolling MAPE over windows ---
    if "mape" in report_df.columns and "test_start" in report_df.columns:
        st.subheader("MAPE Over Backtest Windows")
        fig = go.Figure()
        for model in report_df["model"].unique():
            model_data = report_df[report_df["model"] == model].sort_values(
                "test_start"
            )
            fig.add_trace(
                go.Scatter(
                    x=model_data["test_start"],
                    y=model_data["mape"],
                    mode="lines+markers",
                    name=model,
                )
            )
        fig.update_layout(
            xaxis_title="Test Period Start",
            yaxis_title="MAPE (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
