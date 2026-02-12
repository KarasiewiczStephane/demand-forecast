"""Tests for the Prophet forecasting model."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.prophet_model import (
    ProphetForecast,
    ProphetForecaster,
    ProphetMultiStore,
)


def _make_prophet_df(n: int = 90) -> pd.DataFrame:
    """Create sample data suitable for Prophet training."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "sales": [100 + i % 7 * 10 + i * 0.5 for i in range(n)],
            "store_nbr": 1,
            "family": "GROCERY I",
        }
    )


def _make_multi_store_df(n: int = 90) -> pd.DataFrame:
    """Create sample data with multiple stores."""
    rows = []
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    for date in dates:
        for store in [1, 2]:
            rows.append(
                {
                    "date": date,
                    "store_nbr": store,
                    "family": "GROCERY I",
                    "sales": float(100 + store * 10 + date.dayofweek * 5),
                }
            )
    return pd.DataFrame(rows)


class TestProphetForecaster:
    """Tests for ProphetForecaster class."""

    def test_fit_predict(self) -> None:
        """Model should train and produce forecasts."""
        df = _make_prophet_df()
        model = ProphetForecaster(forecast_horizon=7)
        model.fit(df)
        forecast = model.predict(periods=7)

        assert isinstance(forecast, ProphetForecast)
        assert "yhat" in forecast.forecast_df.columns
        assert "yhat_lower" in forecast.forecast_df.columns
        assert "yhat_upper" in forecast.forecast_df.columns

    def test_confidence_intervals_ordered(self) -> None:
        """Lower bound should be below upper bound."""
        df = _make_prophet_df()
        model = ProphetForecaster(forecast_horizon=7)
        model.fit(df)
        forecast = model.predict(periods=7)

        fdf = forecast.forecast_df
        assert (fdf["yhat_lower"] <= fdf["yhat_upper"]).all()

    def test_forecast_length(self) -> None:
        """Forecast should include historical + future periods."""
        df = _make_prophet_df(90)
        model = ProphetForecaster(forecast_horizon=14)
        model.fit(df)
        forecast = model.predict(periods=14)

        assert len(forecast.forecast_df) == 90 + 14

    def test_predict_without_fit_raises(self) -> None:
        """Predicting without training should raise RuntimeError."""
        model = ProphetForecaster()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded model should produce same predictions."""
        df = _make_prophet_df()
        model = ProphetForecaster(forecast_horizon=7)
        model.fit(df)
        forecast1 = model.predict(periods=7)

        save_path = str(tmp_path / "prophet.pkl")
        model.save(save_path)

        loaded = ProphetForecaster.load(save_path)
        forecast2 = loaded.predict(periods=7)

        pd.testing.assert_frame_equal(
            forecast1.forecast_df.reset_index(drop=True),
            forecast2.forecast_df.reset_index(drop=True),
        )

    def test_with_holidays(self) -> None:
        """Model should accept holiday data."""
        df = _make_prophet_df()
        holidays = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-02-27"],
                "description": ["New Year", "Carnival"],
            }
        )
        model = ProphetForecaster(forecast_horizon=7)
        model.fit(df, holidays_df=holidays)
        forecast = model.predict()
        assert len(forecast.forecast_df) > 0

    def test_default_horizon(self) -> None:
        """Default forecast horizon should be 28."""
        model = ProphetForecaster()
        assert model.forecast_horizon == 28


class TestProphetMultiStore:
    """Tests for ProphetMultiStore class."""

    def test_fit_all(self) -> None:
        """Should train one model per store-family group."""
        df = _make_multi_store_df(90)
        multi = ProphetMultiStore(forecast_horizon=7)
        multi.fit_all(df)
        assert len(multi.models) == 2  # 2 stores

    def test_predict_all(self) -> None:
        """Should produce forecasts for all models."""
        df = _make_multi_store_df(90)
        multi = ProphetMultiStore(forecast_horizon=7)
        multi.fit_all(df)
        result = multi.predict_all(periods=7)
        assert "store_nbr" in result.columns
        assert "family" in result.columns
        assert len(result) > 0
