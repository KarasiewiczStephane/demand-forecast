"""Anomaly detection pipeline for sales data.

Provides dual detection using IQR-based univariate analysis and
Isolation Forest multivariate detection. Includes configurable
handling strategies (interpolation, capping, removal) and
report generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class AnomalyMethod(Enum):
    """Available anomaly detection methods."""

    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class HandlingStrategy(Enum):
    """Strategies for handling detected anomalies."""

    INTERPOLATE = "interpolate"
    CAP = "cap"
    REMOVE = "remove"


@dataclass
class AnomalyRecord:
    """Record of a single detected anomaly.

    Attributes:
        date: Date when the anomaly occurred.
        store_nbr: Store number where the anomaly was found.
        family: Product family associated with the anomaly.
        original_value: Original sales value before handling.
        method: Detection method that flagged the anomaly.
        action: Handling strategy applied to the anomaly.
        new_value: Value after handling, or None if removed.
    """

    date: str
    store_nbr: int
    family: str
    original_value: float
    method: str
    action: str
    new_value: float | None


class AnomalyDetector:
    """Detects anomalies using IQR and Isolation Forest methods.

    Attributes:
        iqr_multiplier: Multiplier for the IQR range boundaries.
        contamination: Expected proportion of outliers for Isolation Forest.
        anomaly_records: List of recorded anomalies.
    """

    def __init__(
        self, iqr_multiplier: float = 1.5, contamination: float = 0.01
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            iqr_multiplier: IQR range multiplier for outlier bounds.
            contamination: Fraction of outliers for Isolation Forest.
        """
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.anomaly_records: list[AnomalyRecord] = []

    def detect_iqr(self, series: pd.Series) -> pd.Series:
        """Detect univariate outliers using the IQR method.

        Args:
            series: Numeric series to analyze.

        Returns:
            Boolean series where True indicates an anomaly.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return (series < lower) | (series > upper)

    def detect_isolation_forest(
        self, df: pd.DataFrame, features: list[str]
    ) -> pd.Series:
        """Detect multivariate anomalies using Isolation Forest.

        Args:
            df: DataFrame containing the feature columns.
            features: List of column names to use as features.

        Returns:
            Boolean series where True indicates an anomaly.
        """
        clf = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        x = df[features].fillna(df[features].median())
        predictions = clf.fit_predict(x)
        return pd.Series(predictions == -1, index=df.index)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        target_col: str = "sales",
        multivariate_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run both detection methods and combine results.

        Args:
            df: DataFrame with sales data.
            target_col: Column name for the target variable.
            multivariate_features: Feature columns for Isolation Forest.

        Returns:
            DataFrame with anomaly flags appended.
        """
        result = df.copy()
        result["anomaly_iqr"] = self.detect_iqr(df[target_col])

        if multivariate_features:
            result["anomaly_if"] = self.detect_isolation_forest(
                df, multivariate_features
            )
        else:
            result["anomaly_if"] = False

        result["is_anomaly"] = result["anomaly_iqr"] | result["anomaly_if"]

        total = result["is_anomaly"].sum()
        logger.info("Detected %d anomalies out of %d rows", total, len(result))
        return result


class AnomalyHandler:
    """Applies handling strategies to detected anomalies.

    Attributes:
        detector: AnomalyDetector instance for recording anomalies.
    """

    def __init__(self, detector: AnomalyDetector) -> None:
        """Initialize the handler with a detector.

        Args:
            detector: AnomalyDetector that holds anomaly records.
        """
        self.detector = detector

    def handle_anomalies(
        self,
        df: pd.DataFrame,
        strategy: HandlingStrategy,
        target_col: str = "sales",
    ) -> pd.DataFrame:
        """Apply the specified handling strategy to anomalies.

        Args:
            df: DataFrame with 'is_anomaly' column from detection.
            strategy: How to handle the detected anomalies.
            target_col: Column containing the values to adjust.

        Returns:
            DataFrame with anomalies handled according to strategy.
        """
        result = df.copy()
        anomaly_mask = result["is_anomaly"]

        if strategy == HandlingStrategy.INTERPOLATE:
            result.loc[anomaly_mask, target_col] = np.nan
            result[target_col] = result[target_col].interpolate(method="linear")
            result[target_col] = result[target_col].bfill().ffill()

        elif strategy == HandlingStrategy.CAP:
            q1 = result[target_col].quantile(0.25)
            q3 = result[target_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            result[target_col] = result[target_col].clip(lower, upper)

        elif strategy == HandlingStrategy.REMOVE:
            result = result[~anomaly_mask].copy()

        # Record anomalies
        for idx in df[anomaly_mask].index:
            method = "iqr" if df.loc[idx, "anomaly_iqr"] else "isolation_forest"
            new_val = result.loc[idx, target_col] if idx in result.index else None
            self.detector.anomaly_records.append(
                AnomalyRecord(
                    date=str(df.loc[idx, "date"]),
                    store_nbr=int(df.loc[idx, "store_nbr"]),
                    family=str(df.loc[idx, "family"]),
                    original_value=float(df.loc[idx, target_col]),
                    method=method,
                    action=strategy.value,
                    new_value=float(new_val) if new_val is not None else None,
                )
            )

        logger.info(
            "Handled %d anomalies with strategy: %s",
            anomaly_mask.sum(),
            strategy.value,
        )
        return result


class AnomalyReportGenerator:
    """Generates anomaly reports as CSV files."""

    def generate_report(
        self, records: list[AnomalyRecord], output_path: str
    ) -> pd.DataFrame:
        """Generate an anomaly report DataFrame and save to CSV.

        Args:
            records: List of AnomalyRecord instances.
            output_path: File path to save the CSV report.

        Returns:
            DataFrame containing the report data.
        """
        if not records:
            report_df = pd.DataFrame(
                columns=[
                    "date",
                    "store_nbr",
                    "family",
                    "original_value",
                    "method",
                    "action",
                    "new_value",
                ]
            )
        else:
            report_df = pd.DataFrame([r.__dict__ for r in records])

        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)
        logger.info(
            "Anomaly report saved to %s (%d records)", output_path, len(report_df)
        )
        return report_df
