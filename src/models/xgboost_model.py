"""XGBoost gradient boosting model for time series forecasting.

Provides configurable XGBoost regression with feature importance
extraction, per-store multi-model training, and model persistence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


@dataclass
class XGBConfig:
    """Hyperparameter configuration for XGBoost.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        subsample: Row sampling ratio per tree.
        colsample_bytree: Column sampling ratio per tree.
        min_child_weight: Minimum sum of instance weight in a child.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        early_stopping_rounds: Rounds without improvement before stopping.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel threads.
    """

    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBForecast:
    """Container for XGBoost prediction results.

    Attributes:
        predictions: Array of predicted values.
        feature_importance: Mapping of feature name to importance score.
        metrics: Optional dictionary of evaluation metrics.
    """

    predictions: np.ndarray
    feature_importance: dict[str, float]
    metrics: dict[str, float] = field(default_factory=dict)


class XGBoostForecaster:
    """XGBoost regression model for demand forecasting.

    Attributes:
        config: Model hyperparameters.
        model: Trained XGBRegressor instance.
        feature_names: Ordered list of feature column names.
    """

    def __init__(self, config: XGBConfig | None = None) -> None:
        """Initialize the XGBoost forecaster.

        Args:
            config: Hyperparameter configuration (defaults to XGBConfig()).
        """
        self.config = config or XGBConfig()
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    def _get_feature_columns(
        self, df: pd.DataFrame, exclude_cols: list[str]
    ) -> list[str]:
        """Identify numeric feature columns excluding specified columns.

        Args:
            df: Input DataFrame.
            exclude_cols: Columns to exclude from features.

        Returns:
            Sorted list of feature column names.
        """
        exclude = set(exclude_cols + ["date", "id", "store_nbr", "family"])
        return sorted(
            c
            for c in df.columns
            if c not in exclude
            and df[c].dtype in ("int64", "float64", "int32", "float32")
        )

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        exclude_cols: list[str] | None = None,
        eval_df: pd.DataFrame | None = None,
    ) -> XGBoostForecaster:
        """Train the XGBoost model.

        Args:
            df: Training DataFrame.
            target_col: Target column name.
            exclude_cols: Additional columns to exclude from features.
            eval_df: Optional validation DataFrame for early stopping.

        Returns:
            Self for method chaining.
        """
        exclude_cols = exclude_cols or []
        self.feature_names = self._get_feature_columns(df, [target_col] + exclude_cols)

        x = np.nan_to_num(df[self.feature_names].values, nan=0.0)
        y = df[target_col].values

        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            tree_method="hist",
        )

        eval_set = None
        if eval_df is not None:
            x_eval = np.nan_to_num(eval_df[self.feature_names].values, nan=0.0)
            y_eval = eval_df[target_col].values
            eval_set = [(x_eval, y_eval)]

        self.model.fit(x, y, eval_set=eval_set, verbose=False)

        logger.info(
            "XGBoost model trained on %d samples with %d features",
            len(x),
            len(self.feature_names),
        )
        return self

    def predict(self, df: pd.DataFrame) -> XGBForecast:
        """Generate predictions for the input DataFrame.

        Args:
            df: DataFrame with the same feature columns as training.

        Returns:
            XGBForecast with predictions and feature importance.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        x = np.nan_to_num(df[self.feature_names].values, nan=0.0)
        predictions = self.model.predict(x)

        importance = dict(zip(self.feature_names, self.model.feature_importances_))

        return XGBForecast(predictions=predictions, feature_importance=importance)

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Get the top N most important features.

        Args:
            n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples sorted descending.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before querying features")

        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n]

    def save(self, path: str) -> None:
        """Save model and metadata to disk.

        Args:
            path: Base path for saving (creates .json and _metadata.json).
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"{path}.json")

        metadata = {
            "feature_names": self.feature_names,
            "config": self.config.__dict__,
        }
        with open(f"{path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> XGBoostForecaster:
        """Load model and metadata from disk.

        Args:
            path: Base path used during save.

        Returns:
            XGBoostForecaster with restored model and feature names.
        """
        with open(f"{path}_metadata.json") as f:
            metadata = json.load(f)

        forecaster = cls(XGBConfig(**metadata["config"]))
        forecaster.model = xgb.XGBRegressor()
        forecaster.model.load_model(f"{path}.json")
        forecaster.feature_names = metadata["feature_names"]
        return forecaster


class XGBoostMultiStore:
    """Train XGBoost models per store-family combination.

    Attributes:
        config: Shared hyperparameter configuration.
        models: Dictionary mapping (store, family) to trained forecasters.
    """

    def __init__(self, config: XGBConfig | None = None) -> None:
        """Initialize with optional shared configuration.

        Args:
            config: XGBConfig applied to all sub-models.
        """
        self.config = config or XGBConfig()
        self.models: dict[tuple[int, str], XGBoostForecaster] = {}

    def fit_all(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        min_samples: int = 100,
    ) -> XGBoostMultiStore:
        """Train one model per store-family group.

        Args:
            df: Full training DataFrame with store_nbr and family.
            target_col: Target column name.
            min_samples: Minimum rows required to train a model.

        Returns:
            Self for method chaining.
        """
        groups = df.groupby(["store_nbr", "family"])

        for (store, family), group_df in groups:
            if len(group_df) < min_samples:
                continue

            split_idx = int(len(group_df) * 0.8)
            train_df = group_df.iloc[:split_idx]
            val_df = group_df.iloc[split_idx:]

            forecaster = XGBoostForecaster(self.config)
            forecaster.fit(train_df, target_col, eval_df=val_df)
            self.models[(int(store), str(family))] = forecaster

        logger.info("Trained %d XGBoost models", len(self.models))
        return self

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all trained models.

        Args:
            df: DataFrame with store_nbr and family columns.

        Returns:
            DataFrame with date, store_nbr, family, and prediction columns.
        """
        results = []

        for (store, family), group_df in df.groupby(["store_nbr", "family"]):
            if (store, family) not in self.models:
                continue

            forecast = self.models[(store, family)].predict(group_df)
            result_df = group_df[["date", "store_nbr", "family"]].copy()
            result_df["prediction"] = forecast.predictions
            results.append(result_df)

        if not results:
            return pd.DataFrame(columns=["date", "store_nbr", "family", "prediction"])

        return pd.concat(results, ignore_index=True)
