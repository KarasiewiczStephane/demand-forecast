"""Shared test fixtures for the demand forecasting test suite."""

from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Create a temporary config YAML file.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the temporary config file.
    """
    config = {
        "data": {
            "raw_path": str(tmp_path / "data" / "raw"),
            "processed_path": str(tmp_path / "data" / "processed"),
            "duckdb_path": str(tmp_path / "data" / "forecast.duckdb"),
            "kaggle_competition": "store-sales-time-series-forecasting",
        },
        "models": {
            "forecast_horizons": [7, 14, 28],
            "lstm_lookback": 28,
            "checkpoint_dir": str(tmp_path / "models" / "checkpoints"),
        },
        "backtesting": {
            "train_window_months": 12,
            "test_window_days": 28,
            "step_days": 7,
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture()
def sample_sales_df() -> pd.DataFrame:
    """Create a sample sales DataFrame for testing.

    Returns:
        DataFrame with columns matching the Store Sales dataset.
    """
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    rows = []
    for date in dates:
        for store in [1, 2]:
            for family in ["GROCERY I", "BEVERAGES"]:
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": max(0, 100 + (date.dayofweek * 10) + store * 5),
                        "onpromotion": 1 if date.day % 7 == 0 else 0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def tmp_duckdb_path(tmp_path: Path) -> str:
    """Provide a temporary DuckDB file path.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        String path for a temporary DuckDB database.
    """
    return str(tmp_path / "test.duckdb")
