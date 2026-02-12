"""Tests for forecasting evaluation metrics."""

import numpy as np

from src.evaluation.metrics import compute_all_metrics, mae, mape, rmse, smape


class TestMAPE:
    """Tests for Mean Absolute Percentage Error."""

    def test_perfect_predictions(self) -> None:
        """MAPE should be zero for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == 0.0

    def test_known_value(self) -> None:
        """MAPE should match hand-calculated result."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.1 + 0.1 => mean=0.1 => 10%
        assert abs(mape(y_true, y_pred) - 10.0) < 1e-10

    def test_zero_true_excluded(self) -> None:
        """Rows where y_true is zero should be excluded."""
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 110.0])
        # Only second element: |10/100| = 10%
        assert abs(mape(y_true, y_pred) - 10.0) < 1e-10

    def test_all_zeros(self) -> None:
        """MAPE should return 0.0 when all true values are zero."""
        y_true = np.zeros(5)
        y_pred = np.ones(5)
        assert mape(y_true, y_pred) == 0.0


class TestRMSE:
    """Tests for Root Mean Squared Error."""

    def test_perfect_predictions(self) -> None:
        """RMSE should be zero for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self) -> None:
        """RMSE should match manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_single_element(self) -> None:
        """RMSE of a single-element array."""
        y_true = np.array([10.0])
        y_pred = np.array([13.0])
        assert abs(rmse(y_true, y_pred) - 3.0) < 1e-10


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_predictions(self) -> None:
        """MAE should be zero for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known_value(self) -> None:
        """MAE should match manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        # |1| + |0| + |2| = 3 / 3 = 1.0
        assert abs(mae(y_true, y_pred) - 1.0) < 1e-10

    def test_symmetric(self) -> None:
        """MAE should be symmetric."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([3.0, 4.0])
        assert abs(mae(y_true, y_pred) - mae(y_pred, y_true)) < 1e-10


class TestSMAPE:
    """Tests for Symmetric Mean Absolute Percentage Error."""

    def test_perfect_predictions(self) -> None:
        """SMAPE should be zero for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == 0.0

    def test_all_zeros(self) -> None:
        """SMAPE should return 0.0 when both arrays are zero."""
        y = np.zeros(5)
        assert smape(y, y) == 0.0

    def test_bounded(self) -> None:
        """SMAPE should be between 0 and 200."""
        rng = np.random.default_rng(42)
        y_true = rng.uniform(1, 100, 100)
        y_pred = rng.uniform(1, 100, 100)
        result = smape(y_true, y_pred)
        assert 0.0 <= result <= 200.0

    def test_symmetry(self) -> None:
        """SMAPE should be symmetric in y_true and y_pred."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 35.0])
        assert abs(smape(y_true, y_pred) - smape(y_pred, y_true)) < 1e-10


class TestComputeAllMetrics:
    """Tests for the compute_all_metrics convenience function."""

    def test_returns_all_keys(self) -> None:
        """Result should contain all four metric keys."""
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all_metrics(y, y)
        assert set(result.keys()) == {"mape", "rmse", "mae", "smape"}

    def test_perfect_all_zero(self) -> None:
        """All metrics should be zero for perfect predictions."""
        y = np.array([1.0, 2.0, 3.0])
        result = compute_all_metrics(y, y)
        for v in result.values():
            assert v == 0.0

    def test_values_are_float(self) -> None:
        """All metric values should be plain floats."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])
        result = compute_all_metrics(y_true, y_pred)
        for v in result.values():
            assert isinstance(v, float)
