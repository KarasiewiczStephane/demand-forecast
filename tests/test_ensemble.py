"""Tests for the ensemble model and model registry."""

from pathlib import Path

import numpy as np

from src.models.ensemble import (
    EnsembleForecast,
    EnsembleForecaster,
    ModelRegistry,
)


def _make_predictions() -> dict[str, np.ndarray]:
    """Create sample predictions from three models."""
    np.random.seed(42)
    n = 50
    return {
        "prophet": np.random.normal(100, 10, n),
        "lstm": np.random.normal(100, 12, n),
        "xgboost": np.random.normal(100, 8, n),
    }


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster class."""

    def test_equal_weights(self) -> None:
        """Default weights should be approximately 1/3 each."""
        ensemble = EnsembleForecaster()
        for w in ensemble.weights.values():
            assert abs(w - 1 / 3) < 1e-10

    def test_predict_shape(self) -> None:
        """Ensemble predictions should match input length."""
        preds = _make_predictions()
        ensemble = EnsembleForecaster()
        result = ensemble.predict(preds)
        assert len(result.predictions) == 50

    def test_predict_returns_forecast(self) -> None:
        """Predict should return an EnsembleForecast dataclass."""
        preds = _make_predictions()
        ensemble = EnsembleForecaster()
        result = ensemble.predict(preds)
        assert isinstance(result, EnsembleForecast)
        assert result.weights is not None
        assert "prophet" in result.individual_predictions

    def test_confidence_intervals_ordered(self) -> None:
        """Lower bound should be below upper bound."""
        preds = _make_predictions()
        ensemble = EnsembleForecaster()
        result = ensemble.predict(preds)
        assert (result.confidence_lower <= result.confidence_upper).all()

    def test_confidence_intervals_with_explicit(self) -> None:
        """Explicit CIs should be used when provided."""
        preds = _make_predictions()
        cis = {
            "prophet": (preds["prophet"] - 10, preds["prophet"] + 10),
            "lstm": (preds["lstm"] - 12, preds["lstm"] + 12),
            "xgboost": (preds["xgboost"] - 8, preds["xgboost"] + 8),
        }
        ensemble = EnsembleForecaster()
        result = ensemble.predict(preds, confidence_intervals=cis)
        assert (result.confidence_lower <= result.confidence_upper).all()

    def test_optimize_weights_valid(self) -> None:
        """Optimized weights should sum to 1 and be non-negative."""
        np.random.seed(42)
        preds = _make_predictions()
        y_true = preds["xgboost"] + np.random.normal(0, 1, 50)

        ensemble = EnsembleForecaster()
        weights = ensemble.optimize_weights(preds, y_true)

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6
        for w in weights.values():
            assert w >= -1e-10

    def test_optimize_improves_rmse(self) -> None:
        """Optimized weights should not be worse than equal weights."""
        np.random.seed(42)
        preds = _make_predictions()
        y_true = preds["xgboost"] + np.random.normal(0, 1, 50)

        equal = EnsembleForecaster()
        equal_pred = equal._weighted_average(preds)
        equal_rmse = EnsembleForecaster._compute_rmse(y_true, equal_pred)

        optimized = EnsembleForecaster()
        optimized.optimize_weights(preds, y_true)
        opt_pred = optimized._weighted_average(preds)
        opt_rmse = EnsembleForecaster._compute_rmse(y_true, opt_pred)

        assert opt_rmse <= equal_rmse + 1e-6

    def test_optimized_flag(self) -> None:
        """Optimized flag should be set after optimization."""
        preds = _make_predictions()
        y_true = np.random.normal(100, 5, 50)
        ensemble = EnsembleForecaster()
        assert ensemble.optimized is False
        ensemble.optimize_weights(preds, y_true)
        assert ensemble.optimized is True

    def test_custom_weights(self) -> None:
        """Custom weights should be used for prediction."""
        preds = _make_predictions()
        weights = {"prophet": 0.0, "lstm": 0.0, "xgboost": 1.0}
        ensemble = EnsembleForecaster(weights=weights)
        result = ensemble.predict(preds)
        np.testing.assert_allclose(result.predictions, preds["xgboost"])

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded ensemble should preserve weights."""
        weights = {"prophet": 0.2, "lstm": 0.3, "xgboost": 0.5}
        ensemble = EnsembleForecaster(weights=weights)
        ensemble.optimized = True

        save_path = str(tmp_path / "ensemble.json")
        ensemble.save(save_path)

        loaded = EnsembleForecaster.load(save_path)
        assert loaded.weights == weights
        assert loaded.optimized is True

    def test_rmse_computation(self) -> None:
        """RMSE should match manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(EnsembleForecaster._compute_rmse(y_true, y_pred) - expected) < 1e-10


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_register_and_retrieve(self, tmp_path: Path) -> None:
        """Registered model should be retrievable."""
        registry = ModelRegistry(str(tmp_path / "checkpoints"))
        registry.register(
            model_type="xgboost",
            version="1.0",
            metrics={"rmse": 10.5},
            path="checkpoints/xgb.json",
        )
        latest = registry.get_latest("xgboost")
        assert latest is not None
        assert latest["version"] == "1.0"

    def test_get_latest_none(self, tmp_path: Path) -> None:
        """get_latest should return None for unregistered types."""
        registry = ModelRegistry(str(tmp_path / "checkpoints"))
        assert registry.get_latest("prophet") is None

    def test_multiple_versions(self, tmp_path: Path) -> None:
        """get_latest should return the most recent version."""
        registry = ModelRegistry(str(tmp_path / "checkpoints"))
        registry.register("lstm", "1.0", {"rmse": 15}, "p1")
        registry.register("lstm", "2.0", {"rmse": 12}, "p2")
        latest = registry.get_latest("lstm")
        assert latest is not None
        assert latest["version"] == "2.0"

    def test_persistence(self, tmp_path: Path) -> None:
        """Registry should persist across instances."""
        path = str(tmp_path / "checkpoints")
        registry1 = ModelRegistry(path)
        registry1.register("prophet", "1.0", {}, "p")

        registry2 = ModelRegistry(path)
        assert registry2.get_latest("prophet") is not None
