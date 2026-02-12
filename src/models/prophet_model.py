"""Prophet forecasting model implementation.

Wraps Facebook Prophet for time series forecasting with support
for Ecuadorian holidays, custom regressors, confidence intervals,
and per-store/family multi-model training.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


@dataclass
class ProphetForecast:
    """Container for Prophet model forecast results.

    Attributes:
        forecast_df: DataFrame with ds, yhat, yhat_lower, yhat_upper columns.
        model: Trained Prophet model instance.
        metrics: Dictionary of evaluation metrics.
    """

    forecast_df: pd.DataFrame
    model: Prophet
    metrics: dict[str, float]


class ProphetForecaster:
    """Prophet model wrapper for demand forecasting.

    Attributes:
        forecast_horizon: Number of periods to forecast.
        yearly_seasonality: Whether to fit yearly seasonal component.
        weekly_seasonality: Whether to fit weekly seasonal component.
        daily_seasonality: Whether to fit daily seasonal component.
        changepoint_prior_scale: Regularization for trend changepoints.
        seasonality_prior_scale: Regularization for seasonality.
        holidays_prior_scale: Regularization for holiday effects.
    """

    def __init__(
        self,
        forecast_horizon: int = 28,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
    ) -> None:
        """Initialize the Prophet forecaster.

        Args:
            forecast_horizon: Number of days to forecast ahead.
            yearly_seasonality: Enable yearly seasonality.
            weekly_seasonality: Enable weekly seasonality.
            daily_seasonality: Enable daily seasonality.
            changepoint_prior_scale: Flexibility of trend changepoints.
            seasonality_prior_scale: Flexibility of seasonality.
            holidays_prior_scale: Flexibility of holiday effects.
        """
        self.forecast_horizon = forecast_horizon
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.model: Prophet | None = None

    def _create_holidays_df(self, holidays_df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataset holidays to Prophet format.

        Args:
            holidays_df: DataFrame with 'date' and 'description' columns.

        Returns:
            Prophet-formatted holidays DataFrame.
        """
        prophet_holidays = holidays_df[["date", "description"]].copy()
        prophet_holidays.columns = ["ds", "holiday"]
        prophet_holidays["ds"] = pd.to_datetime(prophet_holidays["ds"])
        prophet_holidays["lower_window"] = -1
        prophet_holidays["upper_window"] = 1
        return prophet_holidays

    def _prepare_data(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: str = "sales",
    ) -> pd.DataFrame:
        """Prepare data for Prophet (requires 'ds' and 'y' columns).

        Args:
            df: Input DataFrame with date and target columns.
            date_col: Name of the date column.
            target_col: Name of the target column.

        Returns:
            DataFrame with 'ds' and 'y' columns for Prophet.
        """
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        return prophet_df.dropna()

    def fit(
        self,
        df: pd.DataFrame,
        holidays_df: pd.DataFrame | None = None,
        regressors: list[str] | None = None,
    ) -> ProphetForecaster:
        """Train the Prophet model.

        Args:
            df: Training data with date and sales columns.
            holidays_df: Optional holidays DataFrame.
            regressors: Optional list of additional regressor column names.

        Returns:
            Self for method chaining.
        """
        prophet_df = self._prepare_data(df)

        holidays = (
            self._create_holidays_df(holidays_df) if holidays_df is not None else None
        )

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays=holidays,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=0.95,
        )

        if regressors:
            for reg in regressors:
                self.model.add_regressor(reg)
                prophet_df[reg] = df[reg].values[: len(prophet_df)]

        self.model.fit(prophet_df)
        logger.info("Prophet model trained on %d data points", len(prophet_df))
        return self

    def predict(
        self,
        periods: int | None = None,
        future_regressors: pd.DataFrame | None = None,
    ) -> ProphetForecast:
        """Generate forecast with confidence intervals.

        Args:
            periods: Number of periods to forecast (defaults to horizon).
            future_regressors: Future values for additional regressors.

        Returns:
            ProphetForecast with predictions and confidence intervals.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        periods = periods or self.forecast_horizon
        future = self.model.make_future_dataframe(periods=periods)

        if future_regressors is not None:
            for col in future_regressors.columns:
                if col != "ds":
                    future = future.merge(
                        future_regressors[["ds", col]], on="ds", how="left"
                    )

        forecast = self.model.predict(future)

        return ProphetForecast(
            forecast_df=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            model=self.model,
            metrics={},
        )

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path for saving the model.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Prophet model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> ProphetForecaster:
        """Load model from disk.

        Args:
            path: File path to load the model from.

        Returns:
            ProphetForecaster with the loaded model.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        forecaster = cls()
        forecaster.model = model
        return forecaster


class ProphetMultiStore:
    """Train Prophet models for multiple store/family combinations.

    Attributes:
        prophet_kwargs: Keyword arguments passed to ProphetForecaster.
        models: Dictionary mapping (store, family) to trained forecasters.
    """

    def __init__(self, **prophet_kwargs: Any) -> None:
        """Initialize with Prophet configuration.

        Args:
            **prophet_kwargs: Arguments forwarded to ProphetForecaster.
        """
        self.prophet_kwargs = prophet_kwargs
        self.models: dict[tuple[int, str], ProphetForecaster] = {}

    def fit_all(
        self,
        df: pd.DataFrame,
        holidays_df: pd.DataFrame | None = None,
    ) -> ProphetMultiStore:
        """Train one model per store-family group.

        Args:
            df: Full training DataFrame with store_nbr and family.
            holidays_df: Optional holidays data.

        Returns:
            Self for method chaining.
        """
        groups = df.groupby(["store_nbr", "family"])

        for (store, family), group_df in groups:
            forecaster = ProphetForecaster(**self.prophet_kwargs)
            forecaster.fit(group_df, holidays_df)
            self.models[(int(store), str(family))] = forecaster

        logger.info("Trained %d Prophet models", len(self.models))
        return self

    def predict_all(self, periods: int = 28) -> pd.DataFrame:
        """Generate forecasts for all models.

        Args:
            periods: Number of periods to forecast.

        Returns:
            DataFrame with forecasts for all store-family combinations.
        """
        all_forecasts = []

        for (store, family), model in self.models.items():
            forecast = model.predict(periods)
            forecast_df = forecast.forecast_df.copy()
            forecast_df["store_nbr"] = store
            forecast_df["family"] = family
            all_forecasts.append(forecast_df)

        return pd.concat(all_forecasts, ignore_index=True)
