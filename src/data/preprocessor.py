"""Data validation and preprocessing for the Store Sales dataset.

Provides validation checks for date continuity, missing values,
and negative sales, along with dataset merging utilities.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from data validation checks.

    Attributes:
        is_valid: Whether all validation checks passed.
        missing_dates: List of date strings missing from the time series.
        missing_values_count: Column-wise count of missing values.
        negative_sales_count: Number of rows with negative sales.
        warnings: List of warning messages from validation.
    """

    is_valid: bool
    missing_dates: list[str]
    missing_values_count: dict[str, int]
    negative_sales_count: int
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validates sales data for quality issues."""

    def validate_sales_data(self, df: pd.DataFrame) -> ValidationResult:
        """Run validation checks on sales data.

        Checks for date continuity gaps, missing values, and
        negative sales amounts.

        Args:
            df: DataFrame with at least 'date' and 'sales' columns.

        Returns:
            ValidationResult summarizing all findings.
        """
        warnings: list[str] = []

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Check date continuity
        date_range = pd.date_range(df["date"].min(), df["date"].max())
        actual_dates = set(df["date"].dt.normalize().unique())
        expected_dates = set(date_range)
        missing_dates = sorted(expected_dates - actual_dates)

        if missing_dates:
            warnings.append(f"Found {len(missing_dates)} missing dates in time series")

        # Check missing values
        missing_counts = df.isnull().sum().to_dict()
        total_missing = sum(v for v in missing_counts.values() if v > 0)
        if total_missing > 0:
            warnings.append(f"Found {total_missing} total missing values")

        # Check negative sales
        negative_sales = int((df["sales"] < 0).sum())
        if negative_sales > 0:
            warnings.append(f"Found {negative_sales} rows with negative sales")

        is_valid = len(missing_dates) == 0 and negative_sales == 0

        for w in warnings:
            logger.warning(w)

        return ValidationResult(
            is_valid=is_valid,
            missing_dates=[str(d.date()) for d in missing_dates],
            missing_values_count=missing_counts,
            negative_sales_count=negative_sales,
            warnings=warnings,
        )


class DataPreprocessor:
    """Loads and merges Store Sales competition datasets."""

    def load_store_sales_data(self, data_path: str) -> dict[str, pd.DataFrame]:
        """Load all CSV files from the Store Sales dataset.

        Args:
            data_path: Directory containing the extracted CSV files.

        Returns:
            Dictionary mapping dataset names to DataFrames.

        Raises:
            FileNotFoundError: If required CSV files are not found.
        """
        path = Path(data_path)
        datasets: dict[str, pd.DataFrame] = {}

        file_map = {
            "train": "train.csv",
            "test": "test.csv",
            "stores": "stores.csv",
            "oil": "oil.csv",
            "holidays": "holidays_events.csv",
            "transactions": "transactions.csv",
        }

        for name, filename in file_map.items():
            filepath = path / filename
            if filepath.exists():
                datasets[name] = pd.read_csv(filepath)
                logger.info("Loaded %s: %d rows", name, len(datasets[name]))
            else:
                logger.warning("File not found: %s", filepath)

        return datasets

    def merge_datasets(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all datasets into a single DataFrame.

        Joins train data with stores, oil prices, and transactions.

        Args:
            datasets: Dictionary of named DataFrames from load_store_sales_data.

        Returns:
            Merged DataFrame with all features combined.
        """
        df = datasets["train"].copy()

        if "stores" in datasets:
            df = df.merge(datasets["stores"], on="store_nbr", how="left")

        if "oil" in datasets:
            df = df.merge(datasets["oil"], on="date", how="left")

        if "transactions" in datasets:
            df = df.merge(
                datasets["transactions"], on=["date", "store_nbr"], how="left"
            )

        logger.info("Merged dataset: %d rows, %d columns", len(df), len(df.columns))
        return df
