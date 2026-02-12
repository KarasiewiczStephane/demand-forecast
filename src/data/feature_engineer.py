"""Feature engineering pipeline for time series forecasting.

Builds comprehensive features including lag values, rolling statistics,
temporal indicators, Ecuadorian holiday effects, and promotion/oil
price features for the Store Sales dataset.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates ML features from raw sales and auxiliary data.

    Attributes:
        lag_days: List of lag periods to generate.
        rolling_windows: List of rolling window sizes.
    """

    def __init__(
        self,
        lag_days: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        """Initialize the feature engineer.

        Args:
            lag_days: Lag periods in days for lag features.
            rolling_windows: Window sizes for rolling statistics.
        """
        self.lag_days = lag_days or [7, 14, 28, 365]
        self.rolling_windows = rolling_windows or [7, 28]

    def add_lag_features(
        self, df: pd.DataFrame, target_col: str = "sales"
    ) -> pd.DataFrame:
        """Add lagged values of the target column.

        Args:
            df: Input DataFrame with target column.
            target_col: Column to create lags for.

        Returns:
            DataFrame with lag feature columns appended.
        """
        result = df.copy()
        for lag in self.lag_days:
            result[f"{target_col}_lag_{lag}"] = result.groupby(["store_nbr", "family"])[
                target_col
            ].shift(lag)
        return result

    def add_rolling_features(
        self, df: pd.DataFrame, target_col: str = "sales"
    ) -> pd.DataFrame:
        """Add rolling window statistics (mean, std, min, max).

        Args:
            df: Input DataFrame with target column.
            target_col: Column to compute rolling stats for.

        Returns:
            DataFrame with rolling feature columns appended.
        """
        result = df.copy()
        for window in self.rolling_windows:
            grouped = result.groupby(["store_nbr", "family"])[target_col]
            result[f"{target_col}_rolling_mean_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            result[f"{target_col}_rolling_std_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
            result[f"{target_col}_rolling_min_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).min()
            )
            result[f"{target_col}_rolling_max_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).max()
            )
        return result

    def add_temporal_features(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        """Add date-based calendar features.

        Args:
            df: Input DataFrame with a date column.
            date_col: Name of the date column.

        Returns:
            DataFrame with temporal feature columns appended.
        """
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])

        result["day_of_week"] = result[date_col].dt.dayofweek
        result["day_of_month"] = result[date_col].dt.day
        result["month"] = result[date_col].dt.month
        result["quarter"] = result[date_col].dt.quarter
        result["year"] = result[date_col].dt.year
        result["week_of_year"] = result[date_col].dt.isocalendar().week.astype(int)
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        result["is_month_start"] = result[date_col].dt.is_month_start.astype(int)
        result["is_month_end"] = result[date_col].dt.is_month_end.astype(int)

        return result

    def add_holiday_features(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        holidays_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Add holiday indicator features from the dataset holidays table.

        Args:
            df: Input DataFrame with a date column.
            date_col: Name of the date column.
            holidays_df: DataFrame with 'date' and 'type' columns.

        Returns:
            DataFrame with holiday feature columns appended.
        """
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])

        if holidays_df is not None:
            holidays_df = holidays_df.copy()
            holidays_df["date"] = pd.to_datetime(holidays_df["date"])
            for htype in holidays_df["type"].unique():
                holiday_dates = holidays_df[holidays_df["type"] == htype]["date"]
                col_name = f"is_holiday_{str(htype).lower().replace(' ', '_')}"
                result[col_name] = result[date_col].isin(holiday_dates).astype(int)

        result["days_to_payday"] = result[date_col].apply(
            lambda x: min(abs(x.day - 15), abs(x.day - x.days_in_month))
        )

        return result

    def add_promotion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the promotion flag from the dataset.

        Args:
            df: Input DataFrame, optionally containing 'onpromotion'.

        Returns:
            DataFrame with cleaned promotion column.
        """
        result = df.copy()
        if "onpromotion" in result.columns:
            result["onpromotion"] = result["onpromotion"].fillna(0).astype(int)
        return result

    def add_oil_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add oil price features with lag and percent change.

        Args:
            df: Input DataFrame, optionally containing 'dcoilwtico'.

        Returns:
            DataFrame with oil price feature columns appended.
        """
        result = df.copy()
        if "dcoilwtico" in result.columns:
            result["oil_price"] = result["dcoilwtico"].ffill()
            result["oil_price_lag_7"] = result["oil_price"].shift(7)
            result["oil_price_change"] = result["oil_price"].pct_change(7)
        return result

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        holidays_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run the complete feature engineering pipeline.

        Args:
            df: Raw DataFrame with sales data.
            holidays_df: Optional holidays DataFrame for holiday features.

        Returns:
            DataFrame with all engineered features.
        """
        result = df.copy()
        result = self.add_temporal_features(result)
        result = self.add_lag_features(result)
        result = self.add_rolling_features(result)
        result = self.add_holiday_features(result, holidays_df=holidays_df)
        result = self.add_promotion_features(result)
        result = self.add_oil_features(result)

        logger.info("Feature engineering complete: %d columns", len(result.columns))
        return result
