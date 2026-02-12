"""Shared data loading utilities for the Streamlit dashboard.

Provides cached loaders that read from DuckDB, CSV reports, and
model checkpoints so each page can access data without redundant I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import load_config

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def load_sales_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load historical sales data.

    Tries loading from the processed CSV first, then falls back to
    the raw training data.

    Args:
        csv_path: Explicit path to a sales CSV. When *None* the path
            is inferred from the project config.

    Returns:
        DataFrame with at least ``date``, ``store_nbr``, ``family``,
        and ``sales`` columns.
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        config = load_config()
        raw_path = Path(config.data["raw_path"])
        train_path = raw_path / "train.csv"
        if train_path.exists():
            df = pd.read_csv(str(train_path))
        else:
            df = pd.DataFrame(columns=["date", "store_nbr", "family", "sales"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def load_backtest_report(report_path: str | None = None) -> pd.DataFrame:
    """Load a backtest CSV report.

    Args:
        report_path: Path to the report CSV. Defaults to
            ``data/reports/backtest_report.csv``.

    Returns:
        DataFrame of per-window, per-model metrics.
    """
    path = report_path or "data/reports/backtest_report.csv"
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_anomaly_report(report_path: str | None = None) -> pd.DataFrame:
    """Load an anomaly detection report.

    Args:
        report_path: Path to the anomaly CSV. Defaults to
            ``data/reports/anomaly_report.csv``.

    Returns:
        DataFrame of detected anomalies.
    """
    path = report_path or "data/reports/anomaly_report.csv"
    if Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_store_families(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    """Extract unique stores and families from a sales DataFrame.

    Args:
        df: Sales DataFrame with ``store_nbr`` and ``family`` columns.

    Returns:
        Tuple of (sorted store numbers, sorted families).
    """
    stores = (
        sorted(df["store_nbr"].unique().tolist()) if "store_nbr" in df.columns else []
    )
    families = sorted(df["family"].unique().tolist()) if "family" in df.columns else []
    return stores, families
