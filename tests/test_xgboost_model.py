"""Tests for the XGBoost forecasting model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import (
    XGBConfig,
    XGBForecast,
    XGBoostForecaster,
    XGBoostMultiStore,
)


def _make_xgb_df(n: int = 200) -> pd.DataFrame:
    """Create sample data with numeric features for XGBoost."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": np.random.normal(100, 20, n).clip(0),
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(5, 2, n),
            "feature_c": np.random.randint(0, 10, n).astype(float),
        }
    )


def _make_multi_store_df(n: int = 200) -> pd.DataFrame:
    """Create sample data with multiple stores."""
    rows = []
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    for date in dates:
        for store in [1, 2]:
            rows.append(
                {
                    "date": date,
                    "store_nbr": store,
                    "family": "GROCERY I",
                    "sales": float(np.random.normal(100, 10)),
                    "feat_x": float(np.random.normal(0, 1)),
                }
            )
    return pd.DataFrame(rows)


class TestXGBoostForecaster:
    """Tests for XGBoostForecaster class."""

    def test_fit_predict(self) -> None:
        """Model should train and produce predictions."""
        df = _make_xgb_df()
        model = XGBoostForecaster(XGBConfig(n_estimators=10))
        model.fit(df, target_col="sales")
        forecast = model.predict(df)

        assert isinstance(forecast, XGBForecast)
        assert len(forecast.predictions) == len(df)
        assert len(forecast.feature_importance) > 0

    def test_predictions_shape(self) -> None:
        """Prediction length should match input rows."""
        df = _make_xgb_df(100)
        model = XGBoostForecaster(XGBConfig(n_estimators=10))
        model.fit(df)
        forecast = model.predict(df)
        assert forecast.predictions.shape == (100,)

    def test_feature_importance_sums(self) -> None:
        """Feature importance values should sum to approximately 1."""
        df = _make_xgb_df()
        model = XGBoostForecaster(XGBConfig(n_estimators=50))
        model.fit(df)
        forecast = model.predict(df)
        total = sum(forecast.feature_importance.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_without_fit_raises(self) -> None:
        """Predicting without training should raise RuntimeError."""
        model = XGBoostForecaster()
        df = _make_xgb_df()
        with pytest.raises(RuntimeError):
            model.predict(df)

    def test_get_top_features(self) -> None:
        """Top features should be sorted by importance descending."""
        df = _make_xgb_df()
        model = XGBoostForecaster(XGBConfig(n_estimators=50))
        model.fit(df)
        top = model.get_top_features(n=3)
        assert len(top) == 3
        importances = [t[1] for t in top]
        assert importances == sorted(importances, reverse=True)

    def test_get_top_features_without_fit_raises(self) -> None:
        """Querying features without training should raise RuntimeError."""
        model = XGBoostForecaster()
        with pytest.raises(RuntimeError):
            model.get_top_features()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded model should produce same predictions."""
        df = _make_xgb_df()
        model = XGBoostForecaster(XGBConfig(n_estimators=10))
        model.fit(df)
        pred1 = model.predict(df).predictions

        save_path = str(tmp_path / "xgb_model")
        model.save(save_path)

        loaded = XGBoostForecaster.load(save_path)
        pred2 = loaded.predict(df).predictions
        np.testing.assert_allclose(pred1, pred2, atol=1e-5)

    def test_save_without_model_raises(self, tmp_path: Path) -> None:
        """Saving without training should raise RuntimeError."""
        model = XGBoostForecaster()
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "no_model"))

    def test_with_eval_set(self) -> None:
        """Training with eval set should work without error."""
        df = _make_xgb_df(200)
        train_df = df.iloc[:160]
        val_df = df.iloc[160:]
        model = XGBoostForecaster(XGBConfig(n_estimators=10))
        model.fit(train_df, eval_df=val_df)
        forecast = model.predict(val_df)
        assert len(forecast.predictions) == len(val_df)

    def test_handles_nan_features(self) -> None:
        """Model should handle NaN values in features."""
        df = _make_xgb_df(100)
        df.loc[10:20, "feature_a"] = np.nan
        model = XGBoostForecaster(XGBConfig(n_estimators=10))
        model.fit(df)
        forecast = model.predict(df)
        assert len(forecast.predictions) == 100

    def test_config_defaults(self) -> None:
        """Default config should have expected values."""
        config = XGBConfig()
        assert config.n_estimators == 1000
        assert config.random_state == 42


class TestXGBoostMultiStore:
    """Tests for XGBoostMultiStore class."""

    def test_fit_all(self) -> None:
        """Should train models for each store-family group."""
        df = _make_multi_store_df(200)
        multi = XGBoostMultiStore(XGBConfig(n_estimators=10))
        multi.fit_all(df, min_samples=50)
        assert len(multi.models) == 2

    def test_predict_all(self) -> None:
        """Should produce predictions for all trained models."""
        df = _make_multi_store_df(200)
        multi = XGBoostMultiStore(XGBConfig(n_estimators=10))
        multi.fit_all(df, min_samples=50)
        result = multi.predict_all(df)
        assert "prediction" in result.columns
        assert len(result) > 0

    def test_skip_small_groups(self) -> None:
        """Groups smaller than min_samples should be skipped."""
        df = _make_multi_store_df(60)
        multi = XGBoostMultiStore(XGBConfig(n_estimators=10))
        multi.fit_all(df, min_samples=100)
        assert len(multi.models) == 0

    def test_predict_all_empty(self) -> None:
        """predict_all with no models should return empty DataFrame."""
        multi = XGBoostMultiStore()
        df = _make_multi_store_df(50)
        result = multi.predict_all(df)
        assert len(result) == 0
