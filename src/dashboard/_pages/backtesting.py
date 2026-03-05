"""Backtesting Results page for the Streamlit dashboard.

Visualises rolling accuracy across backtest windows, highlights
which model won each window, and shows Diebold-Mariano test
results for statistical comparison.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.utils import load_backtest_report


def render_backtesting_results() -> None:
    """Render the Backtesting Results page."""
    st.title("Backtesting Results")

    bt_df = load_backtest_report()
    if bt_df.empty:
        st.warning("No backtest report found. Run backtesting first.")
        return

    # --- Rolling accuracy over windows ---
    if (
        "mape" in bt_df.columns
        and "test_start" in bt_df.columns
        and "model" in bt_df.columns
    ):
        st.subheader("Rolling Accuracy Over Time")
        fig = go.Figure()
        for model in bt_df["model"].unique():
            model_data = bt_df[bt_df["model"] == model].sort_values("test_start")
            fig.add_trace(
                go.Scatter(
                    x=model_data["test_start"],
                    y=model_data["mape"],
                    mode="lines+markers",
                    name=model,
                )
            )
        fig.update_layout(
            title="MAPE by Backtest Window",
            xaxis_title="Test Period Start",
            yaxis_title="MAPE (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Best model per window ---
    if (
        "window_id" in bt_df.columns
        and "mape" in bt_df.columns
        and "model" in bt_df.columns
    ):
        st.subheader("Model Performance by Period")
        best_per_window = bt_df.loc[bt_df.groupby("window_id")["mape"].idxmin()]
        wins = best_per_window["model"].value_counts()

        fig_wins = px.bar(
            x=wins.index,
            y=wins.values,
            title='Number of "Best Performer" Windows per Model',
            labels={"x": "Model", "y": "Windows Won"},
        )
        st.plotly_chart(fig_wins, use_container_width=True)

    # --- Aggregate summary table ---
    metric_cols = [c for c in bt_df.columns if c in ("mape", "rmse", "mae", "smape")]
    if metric_cols and "model" in bt_df.columns:
        st.subheader("Aggregate Metrics")
        agg = bt_df.groupby("model")[metric_cols].agg(["mean", "std"])
        st.dataframe(agg, use_container_width=True)
