"""Ensemble forecasting model combining Prophet, LSTM, and XGBoost.

Implements weighted averaging with optional weight optimization
via constrained minimization on a validation set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

MODEL_NAMES = ["prophet", "lstm", "xgboost"]


@dataclass
class EnsembleForecast:
    """Container for ensemble prediction results.

    Attributes:
        predictions: Weighted average predictions.
        weights: Model name to weight mapping.
        individual_predictions: Per-model raw predictions.
        confidence_lower: Lower confidence bound.
        confidence_upper: Upper confidence bound.
    """

    predictions: np.ndarray
    weights: dict[str, float]
    individual_predictions: dict[str, np.ndarray]
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray


class EnsembleForecaster:
    """Combines multiple model predictions using weighted averaging.

    Supports equal weighting and SLSQP-optimized weighting on
    a validation set to minimize RMSE.

    Attributes:
        weights: Current model weights.
        optimized: Whether weights have been optimized.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize the ensemble forecaster.

        Args:
            weights: Initial model weights (defaults to equal weights).
        """
        self.weights = weights or {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}
        self.optimized = False

    def _weighted_average(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted average of model predictions.

        Args:
            predictions: Mapping of model name to prediction array.

        Returns:
            Weighted average prediction array.
        """
        weighted_sum = np.zeros_like(next(iter(predictions.values())))
        total_weight = 0.0

        for name, preds in predictions.items():
            if name in self.weights:
                weighted_sum = weighted_sum + self.weights[name] * preds
                total_weight += self.weights[name]

        if total_weight == 0:
            return weighted_sum

        return weighted_sum / total_weight

    @staticmethod
    def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            RMSE value.
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def optimize_weights(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        method: str = "SLSQP",
    ) -> dict[str, float]:
        """Optimize ensemble weights to minimize RMSE on validation data.

        Finds non-negative weights summing to 1 that minimize RMSE.

        Args:
            predictions: Per-model predictions on the validation set.
            y_true: Ground truth values for the validation set.
            method: Scipy optimization method.

        Returns:
            Optimized model weights dictionary.
        """
        names = [n for n in MODEL_NAMES if n in predictions]
        pred_matrix = np.column_stack([predictions[n] for n in names])

        def objective(w: np.ndarray) -> float:
            ensemble_pred = np.dot(pred_matrix, w)
            return self._compute_rmse(y_true, ensemble_pred)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in names]
        x0 = np.array([1.0 / len(names)] * len(names))

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            self.weights = dict(zip(names, result.x))
            self.optimized = True
            logger.info("Optimized weights: %s", self.weights)
        else:
            logger.warning("Weight optimization did not converge")

        return self.weights

    def predict(
        self,
        predictions: dict[str, np.ndarray],
        confidence_intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> EnsembleForecast:
        """Generate ensemble predictions with confidence intervals.

        Args:
            predictions: Per-model prediction arrays.
            confidence_intervals: Optional per-model (lower, upper) tuples.

        Returns:
            EnsembleForecast with combined predictions.
        """
        ensemble_pred = self._weighted_average(predictions)

        if confidence_intervals:
            lower = self._weighted_average(
                {k: v[0] for k, v in confidence_intervals.items()}
            )
            upper = self._weighted_average(
                {k: v[1] for k, v in confidence_intervals.items()}
            )
        else:
            pred_array = np.array(list(predictions.values()))
            std = np.std(pred_array, axis=0)
            lower = ensemble_pred - 1.96 * std
            upper = ensemble_pred + 1.96 * std

        return EnsembleForecast(
            predictions=ensemble_pred,
            weights=self.weights.copy(),
            individual_predictions=predictions.copy(),
            confidence_lower=lower,
            confidence_upper=upper,
        )

    def save(self, path: str) -> None:
        """Save ensemble configuration to disk.

        Args:
            path: File path for saving.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        config = {"weights": self.weights, "optimized": self.optimized}
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Ensemble config saved to %s", path)

    @classmethod
    def load(cls, path: str) -> EnsembleForecaster:
        """Load ensemble configuration from disk.

        Args:
            path: File path to load from.

        Returns:
            EnsembleForecaster with restored weights.
        """
        with open(path) as f:
            config = json.load(f)
        ensemble = cls(weights=config["weights"])
        ensemble.optimized = config["optimized"]
        return ensemble


class ModelRegistry:
    """Simple model versioning registry backed by a JSON file.

    Attributes:
        base_path: Root directory for model checkpoints.
        registry_file: Path to the registry JSON file.
        registry: In-memory registry data.
    """

    def __init__(self, base_path: str = "models/checkpoints") -> None:
        """Initialize the registry.

        Args:
            base_path: Root directory for model checkpoints.
        """
        self.base_path = Path(base_path)
        self.registry_file = self.base_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk or create empty."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry: dict = json.load(f)
        else:
            self.registry = {"models": []}

    def register(
        self,
        model_type: str,
        version: str,
        metrics: dict[str, float],
        path: str,
    ) -> None:
        """Register a new model entry.

        Args:
            model_type: Type of model (prophet, lstm, xgboost, ensemble).
            version: Version identifier string.
            metrics: Performance metrics dictionary.
            path: Path to the model checkpoint.
        """
        from datetime import datetime

        entry = {
            "type": model_type,
            "version": version,
            "metrics": metrics,
            "path": path,
            "timestamp": datetime.now().isoformat(),
        }
        self.registry["models"].append(entry)
        self._save_registry()
        logger.info("Registered model: %s v%s", model_type, version)

    def _save_registry(self) -> None:
        """Persist registry to disk."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def get_latest(self, model_type: str) -> dict | None:
        """Get the most recently registered model of a given type.

        Args:
            model_type: Type of model to look up.

        Returns:
            Registry entry dict or None if not found.
        """
        matches = [m for m in self.registry["models"] if m["type"] == model_type]
        return matches[-1] if matches else None
