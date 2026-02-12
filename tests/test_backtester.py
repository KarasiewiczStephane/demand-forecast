"""Tests for the backtesting framework and report generator."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtester import (
    BacktestReport,
    BacktestResult,
    BacktestWindow,
    Backtester,
)
from src.evaluation.report_generator import ReportGenerator


def _make_daily_df(n_days: int = 500) -> pd.DataFrame:
    """Create a synthetic daily sales DataFrame.

    Args:
        n_days: Number of days of data.

    Returns:
        DataFrame with date and sales columns.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    sales = (
        100 + 10 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + rng.normal(0, 5, n_days)
    )
    return pd.DataFrame({"date": dates, "sales": sales})


def _dummy_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Predict the training mean for every test row."""
    return np.full(len(test_df), train_df["sales"].mean())


def _perfect_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Return the actual test values (perfect model)."""
    return test_df["sales"].values.copy()


class TestBacktestWindowGeneration:
    """Tests for window generation logic."""

    def test_generates_windows(self) -> None:
        """Should produce at least one window for long enough data."""
        df = _make_daily_df(500)
        bt = Backtester(train_window_months=6, test_window_days=28, step_days=7)
        windows = bt._generate_windows(df)
        assert len(windows) > 0

    def test_window_dates_ordered(self) -> None:
        """Train end should be before test start, test start before test end."""
        df = _make_daily_df(500)
        bt = Backtester(train_window_months=6, test_window_days=28, step_days=7)
        for w in bt._generate_windows(df):
            assert w.train_start <= w.train_end
            assert w.train_end < w.test_start
            assert w.test_start <= w.test_end

    def test_no_windows_for_short_data(self) -> None:
        """Very short data should yield no windows."""
        df = _make_daily_df(30)
        bt = Backtester(train_window_months=12, test_window_days=28, step_days=7)
        windows = bt._generate_windows(df)
        assert len(windows) == 0

    def test_window_ids_sequential(self) -> None:
        """Window IDs should be 0, 1, 2, ..."""
        df = _make_daily_df(500)
        bt = Backtester(train_window_months=6, test_window_days=28, step_days=7)
        windows = bt._generate_windows(df)
        for i, w in enumerate(windows):
            assert w.window_id == i

    def test_step_days_spacing(self) -> None:
        """Consecutive windows should be step_days apart."""
        df = _make_daily_df(500)
        bt = Backtester(train_window_months=6, test_window_days=14, step_days=14)
        windows = bt._generate_windows(df)
        for i in range(1, len(windows)):
            delta = (windows[i].train_end - windows[i - 1].train_end).days
            assert delta == 14


class TestBacktesterRun:
    """Tests for the full backtesting run."""

    def test_run_with_single_model(self) -> None:
        """Should complete without error for one model."""
        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=14,
            min_train_samples=30,
        )
        report = bt.run(df, {"mean_model": _dummy_model})
        assert isinstance(report, BacktestReport)
        assert len(report.results) > 0

    def test_run_metrics_computed(self) -> None:
        """Metrics should be computed for each model in each window."""
        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=30,
            min_train_samples=30,
        )
        report = bt.run(df, {"mean_model": _dummy_model})
        for result in report.results:
            assert "mean_model" in result.metrics
            assert "rmse" in result.metrics["mean_model"]

    def test_aggregate_metrics_present(self) -> None:
        """Aggregate metrics should have mean and std keys."""
        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=30,
            min_train_samples=30,
        )
        report = bt.run(df, {"mean_model": _dummy_model})
        agg = report.aggregate_metrics["mean_model"]
        assert "rmse_mean" in agg
        assert "rmse_std" in agg
        assert "mape_mean" in agg

    def test_two_models_dm_test(self) -> None:
        """DM test should be computed between two models."""
        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=30,
            min_train_samples=30,
        )
        report = bt.run(
            df,
            {"mean_model": _dummy_model, "perfect": _perfect_model},
        )
        assert len(report.dm_test_results) > 0
        for pair_result in report.dm_test_results.values():
            assert "statistic" in pair_result
            assert "p_value" in pair_result

    def test_perfect_model_zero_rmse(self) -> None:
        """A perfect model should have zero RMSE in each window."""
        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=30,
            min_train_samples=30,
        )
        report = bt.run(df, {"perfect": _perfect_model})
        for result in report.results:
            assert result.metrics["perfect"]["rmse"] < 1e-10

    def test_model_error_handled(self) -> None:
        """A model that raises should not crash the run."""

        def _failing_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
            raise ValueError("intentional failure")

        df = _make_daily_df(500)
        bt = Backtester(
            train_window_months=6,
            test_window_days=14,
            step_days=30,
            min_train_samples=30,
        )
        report = bt.run(
            df,
            {"ok": _dummy_model, "bad": _failing_model},
        )
        # The ok model should still produce results
        for result in report.results:
            assert "ok" in result.metrics
            assert "bad" not in result.metrics

    def test_empty_report_no_crash(self) -> None:
        """Short data yielding no windows should return empty report."""
        df = _make_daily_df(30)
        bt = Backtester(
            train_window_months=12,
            test_window_days=28,
            step_days=7,
            min_train_samples=30,
        )
        report = bt.run(df, {"mean_model": _dummy_model})
        assert len(report.results) == 0
        assert len(report.aggregate_metrics) == 0
        assert len(report.dm_test_results) == 0


class TestDieboldMariano:
    """Tests for the Diebold-Mariano test implementation."""

    def test_identical_errors(self) -> None:
        """Identical errors should give DM stat ~0 and p-value ~1."""
        e = np.random.default_rng(42).normal(0, 1, 100)
        stat, p = Backtester._diebold_mariano(e, e)
        assert abs(stat) < 1e-10
        assert abs(p - 1.0) < 1e-10

    def test_clearly_different_errors(self) -> None:
        """One much worse model should give a significant result."""
        rng = np.random.default_rng(42)
        e1 = rng.normal(0, 1, 500)
        e2 = rng.normal(0, 10, 500)
        stat, p = Backtester._diebold_mariano(e1, e2)
        # Model 1 has much smaller errors
        assert stat < 0
        assert p < 0.05

    def test_returns_floats(self) -> None:
        """DM test should return plain floats."""
        rng = np.random.default_rng(42)
        e1 = rng.normal(0, 1, 50)
        e2 = rng.normal(0, 2, 50)
        stat, p = Backtester._diebold_mariano(e1, e2)
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestReportGenerator:
    """Tests for CSV report generation."""

    def test_generate_csv(self, tmp_path: Path) -> None:
        """Should write a valid CSV with expected columns."""
        window = BacktestWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-01-28"),
            window_id=0,
        )
        result = BacktestResult(
            window=window,
            y_true=np.array([1.0, 2.0]),
            predictions={"model_a": np.array([1.1, 2.1])},
            metrics={"model_a": {"rmse": 0.1, "mape": 5.0, "mae": 0.1, "smape": 5.0}},
        )
        report = BacktestReport(
            results=[result],
            aggregate_metrics={"model_a": {"rmse_mean": 0.1}},
        )

        out = str(tmp_path / "report.csv")
        df = ReportGenerator.generate_csv_report(report, out)

        assert Path(out).exists()
        assert len(df) == 1
        assert "model" in df.columns
        assert "rmse" in df.columns
        assert "window_id" in df.columns

    def test_multiple_models_rows(self, tmp_path: Path) -> None:
        """Each model in each window should produce one row."""
        window = BacktestWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-01-28"),
            window_id=0,
        )
        result = BacktestResult(
            window=window,
            y_true=np.array([1.0]),
            predictions={
                "model_a": np.array([1.1]),
                "model_b": np.array([1.2]),
            },
            metrics={
                "model_a": {"rmse": 0.1},
                "model_b": {"rmse": 0.2},
            },
        )
        report = BacktestReport(
            results=[result],
            aggregate_metrics={},
        )

        out = str(tmp_path / "report.csv")
        df = ReportGenerator.generate_csv_report(report, out)
        assert len(df) == 2

    def test_empty_report(self, tmp_path: Path) -> None:
        """Empty report should produce an empty CSV."""
        report = BacktestReport(results=[], aggregate_metrics={})
        out = str(tmp_path / "report.csv")
        df = ReportGenerator.generate_csv_report(report, out)
        assert len(df) == 0
