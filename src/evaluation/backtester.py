"""Rolling window cross-validation backtesting framework.

Provides configurable train/test window generation, multi-model
evaluation, metric aggregation, and Diebold-Mariano statistical
comparison between model pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestWindow:
    """A single backtesting time window.

    Attributes:
        train_start: Start date for the training period.
        train_end: End date for the training period.
        test_start: Start date for the test period.
        test_end: End date for the test period.
        window_id: Sequential identifier for this window.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    window_id: int


@dataclass
class BacktestResult:
    """Results from evaluating models on a single window.

    Attributes:
        window: The backtest window definition.
        y_true: Ground truth values for the test period.
        predictions: Mapping of model name to prediction array.
        metrics: Mapping of model name to metrics dictionary.
    """

    window: BacktestWindow
    y_true: np.ndarray
    predictions: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]


@dataclass
class BacktestReport:
    """Aggregated report across all backtest windows.

    Attributes:
        results: Per-window results.
        aggregate_metrics: Mean and std of metrics per model.
        dm_test_results: Diebold-Mariano test results per model pair.
    """

    results: list[BacktestResult]
    aggregate_metrics: dict[str, dict[str, float]]
    dm_test_results: dict[tuple[str, str], dict[str, float]] = field(
        default_factory=dict
    )


class Backtester:
    """Rolling window cross-validation backtester.

    Generates expanding or rolling time windows, runs user-supplied
    model fit-predict callables on each window, aggregates metrics,
    and performs Diebold-Mariano pairwise significance tests.

    Attributes:
        train_window_months: Minimum training months before first test.
        test_window_days: Length of each test window in days.
        step_days: Step size between consecutive windows.
        min_train_samples: Minimum training rows required.
    """

    def __init__(
        self,
        train_window_months: int = 12,
        test_window_days: int = 28,
        step_days: int = 7,
        min_train_samples: int = 365,
    ) -> None:
        """Initialize the backtester.

        Args:
            train_window_months: Months of data before first test window.
            test_window_days: Number of days in each test window.
            step_days: Days between successive test window starts.
            min_train_samples: Minimum rows required in training set.
        """
        self.train_window_months = train_window_months
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_train_samples = min_train_samples

    def _generate_windows(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> list[BacktestWindow]:
        """Generate rolling backtest windows from the data range.

        Args:
            df: Input DataFrame with a date column.
            date_col: Name of the date column.

        Returns:
            List of BacktestWindow instances.
        """
        dates = pd.to_datetime(df[date_col])
        min_date = dates.min()
        max_date = dates.max()

        windows: list[BacktestWindow] = []
        window_id = 0

        current_train_end = min_date + pd.DateOffset(months=self.train_window_months)

        while current_train_end + timedelta(days=self.test_window_days) <= max_date:
            window = BacktestWindow(
                train_start=min_date,
                train_end=current_train_end,
                test_start=current_train_end + timedelta(days=1),
                test_end=current_train_end + timedelta(days=self.test_window_days),
                window_id=window_id,
            )
            windows.append(window)
            current_train_end += timedelta(days=self.step_days)
            window_id += 1

        return windows

    def run(
        self,
        df: pd.DataFrame,
        models: dict[str, Callable],
        target_col: str = "sales",
        date_col: str = "date",
    ) -> BacktestReport:
        """Run backtesting across all rolling windows.

        Each model callable receives ``(train_df, test_df)`` and must
        return a 1-D numpy array of predictions aligned with the test
        set rows.

        Args:
            df: Full dataset DataFrame.
            models: Mapping of model name to fit-predict callable.
            target_col: Name of the target column.
            date_col: Name of the date column.

        Returns:
            BacktestReport with per-window and aggregate results.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        windows = self._generate_windows(df, date_col)
        results: list[BacktestResult] = []

        for window in windows:
            train_mask = (df[date_col] >= window.train_start) & (
                df[date_col] <= window.train_end
            )
            test_mask = (df[date_col] >= window.test_start) & (
                df[date_col] <= window.test_end
            )

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) < self.min_train_samples:
                continue

            if test_df.empty:
                continue

            y_true = test_df[target_col].values
            predictions: dict[str, np.ndarray] = {}
            metrics: dict[str, dict[str, float]] = {}

            for model_name, fit_predict_fn in models.items():
                try:
                    pred = fit_predict_fn(train_df, test_df)
                    predictions[model_name] = pred
                    metrics[model_name] = compute_all_metrics(y_true, pred)
                except Exception:
                    logger.exception(
                        "Error in %s at window %d",
                        model_name,
                        window.window_id,
                    )

            if predictions:
                results.append(
                    BacktestResult(
                        window=window,
                        y_true=y_true,
                        predictions=predictions,
                        metrics=metrics,
                    )
                )

        aggregate = self._aggregate_metrics(results)
        dm_results = self._run_dm_tests(results)

        return BacktestReport(
            results=results,
            aggregate_metrics=aggregate,
            dm_test_results=dm_results,
        )

    @staticmethod
    def _aggregate_metrics(
        results: list[BacktestResult],
    ) -> dict[str, dict[str, float]]:
        """Compute mean and std of each metric across windows.

        Args:
            results: List of per-window backtest results.

        Returns:
            Nested dict: model -> ``{metric_mean, metric_std, ...}``.
        """
        model_metrics: dict[str, dict[str, list[float]]] = {}

        for result in results:
            for model_name, metrics in result.metrics.items():
                if model_name not in model_metrics:
                    model_metrics[model_name] = {k: [] for k in metrics}
                for metric_name, value in metrics.items():
                    model_metrics[model_name][metric_name].append(value)

        aggregate: dict[str, dict[str, float]] = {}
        for model_name, metrics in model_metrics.items():
            agg: dict[str, float] = {}
            for k, v in metrics.items():
                agg[f"{k}_mean"] = float(np.mean(v))
                agg[f"{k}_std"] = float(np.std(v))
            aggregate[model_name] = agg

        return aggregate

    @staticmethod
    def _run_dm_tests(
        results: list[BacktestResult],
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Run Diebold-Mariano tests for all model pairs.

        Args:
            results: List of per-window backtest results.

        Returns:
            Mapping of ``(model_a, model_b)`` to DM statistic and p-value.
        """
        if not results:
            return {}

        model_names = sorted(
            {name for result in results for name in result.predictions}
        )
        dm_results: dict[tuple[str, str], dict[str, float]] = {}

        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1 :]:
                e1: list[float] = []
                e2: list[float] = []

                for result in results:
                    if model1 in result.predictions and model2 in result.predictions:
                        err1 = result.y_true - result.predictions[model1]
                        err2 = result.y_true - result.predictions[model2]
                        e1.extend(err1.tolist())
                        e2.extend(err2.tolist())

                if len(e1) > 1:
                    dm_stat, p_value = Backtester._diebold_mariano(
                        np.array(e1), np.array(e2)
                    )
                    dm_results[(model1, model2)] = {
                        "statistic": dm_stat,
                        "p_value": p_value,
                    }

        return dm_results

    @staticmethod
    def _diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
        """Perform the Diebold-Mariano test on two error series.

        Compares squared-error loss between two sets of forecast errors
        using a standard normal approximation.

        Args:
            e1: Forecast errors from model 1.
            e2: Forecast errors from model 2.

        Returns:
            Tuple of (DM statistic, two-sided p-value).
        """
        d = e1**2 - e2**2
        mean_d = float(np.mean(d))
        var_d = float(np.var(d, ddof=1))

        if var_d == 0:
            return 0.0, 1.0

        n = len(d)
        dm_stat = mean_d / np.sqrt(var_d / n)
        p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))

        return float(dm_stat), p_value
