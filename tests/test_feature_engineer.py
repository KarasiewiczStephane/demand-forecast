"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd

from src.data.feature_engineer import FeatureEngineer


def _make_df(n: int = 60) -> pd.DataFrame:
    """Create a sample DataFrame for feature engineering tests."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rows = []
    for date in dates:
        for store in [1, 2]:
            for family in ["GROCERY I", "BEVERAGES"]:
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": float(100 + date.dayofweek * 10 + store * 5),
                        "onpromotion": 1 if date.day % 7 == 0 else 0,
                    }
                )
    return pd.DataFrame(rows)


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_add_lag_features(self) -> None:
        """Lag features should create new columns with correct shifts."""
        df = _make_df(30)
        fe = FeatureEngineer(lag_days=[7, 14])
        result = fe.add_lag_features(df)
        assert "sales_lag_7" in result.columns
        assert "sales_lag_14" in result.columns

    def test_lag_values_correct(self) -> None:
        """Lag values should match the shifted target."""
        df = _make_df(30)
        fe = FeatureEngineer(lag_days=[1])
        result = fe.add_lag_features(df)
        store1_grocery = result[
            (result["store_nbr"] == 1) & (result["family"] == "GROCERY I")
        ].sort_values("date")
        lag_vals = store1_grocery["sales_lag_1"].dropna().values
        original = store1_grocery["sales"].values[:-1]
        np.testing.assert_array_equal(lag_vals, original)

    def test_add_rolling_features(self) -> None:
        """Rolling features should create mean/std/min/max columns."""
        df = _make_df(30)
        fe = FeatureEngineer(rolling_windows=[7])
        result = fe.add_rolling_features(df)
        assert "sales_rolling_mean_7" in result.columns
        assert "sales_rolling_std_7" in result.columns
        assert "sales_rolling_min_7" in result.columns
        assert "sales_rolling_max_7" in result.columns

    def test_add_temporal_features(self) -> None:
        """Temporal features should include all expected columns."""
        df = _make_df(10)
        fe = FeatureEngineer()
        result = fe.add_temporal_features(df)
        expected = [
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "year",
            "week_of_year",
            "is_weekend",
            "is_month_start",
            "is_month_end",
        ]
        for col in expected:
            assert col in result.columns

    def test_day_of_week_correct(self) -> None:
        """Day of week should be 0 for Monday."""
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-01-02")],  # Monday
                "store_nbr": [1],
                "family": ["GROCERY I"],
                "sales": [100],
            }
        )
        fe = FeatureEngineer()
        result = fe.add_temporal_features(df)
        assert result["day_of_week"].iloc[0] == 0

    def test_is_weekend(self) -> None:
        """Weekend flag should be 1 for Saturday/Sunday."""
        df = pd.DataFrame(
            {
                "date": [
                    pd.Timestamp("2023-01-07"),  # Saturday
                    pd.Timestamp("2023-01-09"),  # Monday
                ],
                "store_nbr": [1, 1],
                "family": ["GROCERY I", "GROCERY I"],
                "sales": [100, 100],
            }
        )
        fe = FeatureEngineer()
        result = fe.add_temporal_features(df)
        assert result["is_weekend"].iloc[0] == 1
        assert result["is_weekend"].iloc[1] == 0

    def test_add_holiday_features(self) -> None:
        """Holiday features should create indicator columns."""
        df = _make_df(10)
        holidays = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02"],
                "type": ["Holiday", "Event"],
            }
        )
        fe = FeatureEngineer()
        result = fe.add_holiday_features(df, holidays_df=holidays)
        assert "is_holiday_holiday" in result.columns
        assert "is_holiday_event" in result.columns
        assert "days_to_payday" in result.columns

    def test_holiday_without_df(self) -> None:
        """Holiday features should work without holidays DataFrame."""
        df = _make_df(10)
        fe = FeatureEngineer()
        result = fe.add_holiday_features(df)
        assert "days_to_payday" in result.columns

    def test_add_promotion_features(self) -> None:
        """Promotion feature should fill NaN and convert to int."""
        df = pd.DataFrame({"onpromotion": [1, 0, None, 1]})
        fe = FeatureEngineer()
        result = fe.add_promotion_features(df)
        assert result["onpromotion"].isna().sum() == 0
        assert result["onpromotion"].dtype == int

    def test_add_oil_features(self) -> None:
        """Oil features should create price, lag, and change columns."""
        df = pd.DataFrame(
            {
                "dcoilwtico": [70.0, None, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0],
            }
        )
        fe = FeatureEngineer()
        result = fe.add_oil_features(df)
        assert "oil_price" in result.columns
        assert "oil_price_lag_7" in result.columns
        assert "oil_price_change" in result.columns

    def test_oil_without_column(self) -> None:
        """Oil features should be skipped if column is missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        fe = FeatureEngineer()
        result = fe.add_oil_features(df)
        assert "oil_price" not in result.columns

    def test_engineer_all_features(self) -> None:
        """Full pipeline should produce expected feature columns."""
        df = _make_df(60)
        fe = FeatureEngineer()
        result = fe.engineer_all_features(df)
        assert "day_of_week" in result.columns
        assert "sales_lag_7" in result.columns
        assert "sales_rolling_mean_7" in result.columns
        assert "days_to_payday" in result.columns

    def test_full_pipeline_row_count(self) -> None:
        """Full pipeline should preserve row count."""
        df = _make_df(30)
        fe = FeatureEngineer()
        result = fe.engineer_all_features(df)
        assert len(result) == len(df)
