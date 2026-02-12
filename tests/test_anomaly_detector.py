"""Tests for the anomaly detection pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.anomaly_detector import (
    AnomalyDetector,
    AnomalyHandler,
    AnomalyMethod,
    AnomalyRecord,
    AnomalyReportGenerator,
    HandlingStrategy,
)


def _make_sales_df(
    n: int = 100, outlier_indices: list[int] | None = None
) -> pd.DataFrame:
    """Create a sample sales DataFrame with optional outliers."""
    np.random.seed(42)
    sales = np.random.normal(100, 10, n)
    if outlier_indices:
        for idx in outlier_indices:
            sales[idx] = 500.0
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": sales,
        }
    )


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_iqr_detects_outliers(self) -> None:
        """IQR method should detect extreme values."""
        detector = AnomalyDetector()
        series = pd.Series([10, 11, 12, 13, 14, 15, 100])
        result = detector.detect_iqr(series)
        assert result.iloc[-1] is np.True_

    def test_iqr_no_outliers(self) -> None:
        """IQR should return all False for uniform data."""
        detector = AnomalyDetector()
        series = pd.Series([10, 11, 12, 13, 14])
        result = detector.detect_iqr(series)
        assert not result.any()

    def test_isolation_forest_detects_anomalies(self) -> None:
        """Isolation Forest should detect synthetic anomalies."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (200, 2))
        anomaly_data = np.array([[10, 10], [11, 11], [-10, -10]])
        data = np.vstack([normal_data, anomaly_data])
        df = pd.DataFrame(data, columns=["f1", "f2"])

        detector = AnomalyDetector(contamination=0.02)
        result = detector.detect_isolation_forest(df, ["f1", "f2"])
        assert result.sum() > 0

    def test_detect_anomalies_combined(self) -> None:
        """Combined detection should flag IQR outliers."""
        df = _make_sales_df(100, outlier_indices=[50, 75])
        detector = AnomalyDetector()
        result = detector.detect_anomalies(df, target_col="sales")
        assert "is_anomaly" in result.columns
        assert "anomaly_iqr" in result.columns
        assert "anomaly_if" in result.columns

    def test_detect_no_anomalies(self) -> None:
        """Clean data should have zero anomalies with IQR only."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D"),
                "store_nbr": 1,
                "family": "GROCERY I",
                "sales": [10, 11, 12, 13, 14, 15, 14, 13, 12, 11],
            }
        )
        detector = AnomalyDetector()
        result = detector.detect_anomalies(df, target_col="sales")
        assert result["is_anomaly"].sum() == 0

    def test_detect_with_multivariate(self) -> None:
        """Multivariate detection should use provided features."""
        df = _make_sales_df(200)
        df["feature1"] = np.random.normal(0, 1, 200)
        detector = AnomalyDetector()
        result = detector.detect_anomalies(
            df, target_col="sales", multivariate_features=["sales", "feature1"]
        )
        assert "anomaly_if" in result.columns

    def test_enum_values(self) -> None:
        """Enums should have expected string values."""
        assert AnomalyMethod.IQR.value == "iqr"
        assert HandlingStrategy.INTERPOLATE.value == "interpolate"
        assert HandlingStrategy.CAP.value == "cap"
        assert HandlingStrategy.REMOVE.value == "remove"


class TestAnomalyHandler:
    """Tests for AnomalyHandler class."""

    def _get_detected_df(self) -> tuple[pd.DataFrame, AnomalyDetector]:
        """Helper to create a detected anomaly DataFrame."""
        df = _make_sales_df(100, outlier_indices=[50])
        detector = AnomalyDetector()
        detected = detector.detect_anomalies(df, target_col="sales")
        return detected, detector

    def test_interpolate_strategy(self) -> None:
        """Interpolation should replace anomalies with interpolated values."""
        detected, detector = self._get_detected_df()
        handler = AnomalyHandler(detector)
        result = handler.handle_anomalies(
            detected, HandlingStrategy.INTERPOLATE, "sales"
        )
        assert len(result) == len(detected)
        assert not result["sales"].isna().any()

    def test_cap_strategy(self) -> None:
        """Capping should limit values to IQR bounds."""
        detected, detector = self._get_detected_df()
        handler = AnomalyHandler(detector)
        result = handler.handle_anomalies(detected, HandlingStrategy.CAP, "sales")
        assert len(result) == len(detected)
        q1 = detected["sales"].quantile(0.25)
        q3 = detected["sales"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        assert result["sales"].max() <= upper + 1e-10

    def test_remove_strategy(self) -> None:
        """Removal should reduce row count by number of anomalies."""
        detected, detector = self._get_detected_df()
        n_anomalies = detected["is_anomaly"].sum()
        handler = AnomalyHandler(detector)
        result = handler.handle_anomalies(detected, HandlingStrategy.REMOVE, "sales")
        assert len(result) == len(detected) - n_anomalies

    def test_anomaly_records_populated(self) -> None:
        """Handler should populate anomaly records on the detector."""
        detected, detector = self._get_detected_df()
        handler = AnomalyHandler(detector)
        handler.handle_anomalies(detected, HandlingStrategy.CAP, "sales")
        assert len(detector.anomaly_records) > 0
        assert isinstance(detector.anomaly_records[0], AnomalyRecord)

    def test_record_has_correct_fields(self) -> None:
        """Anomaly records should have all required fields."""
        detected, detector = self._get_detected_df()
        handler = AnomalyHandler(detector)
        handler.handle_anomalies(detected, HandlingStrategy.CAP, "sales")
        record = detector.anomaly_records[0]
        assert record.date is not None
        assert record.store_nbr == 1
        assert record.family == "GROCERY I"
        assert record.action == "cap"


class TestAnomalyReportGenerator:
    """Tests for AnomalyReportGenerator class."""

    def test_generate_report_with_records(self, tmp_path: Path) -> None:
        """Report should be saved as CSV with correct columns."""
        records = [
            AnomalyRecord(
                date="2023-01-15",
                store_nbr=1,
                family="GROCERY I",
                original_value=500.0,
                method="iqr",
                action="cap",
                new_value=150.0,
            )
        ]
        output_path = str(tmp_path / "report.csv")
        generator = AnomalyReportGenerator()
        report = generator.generate_report(records, output_path)

        assert len(report) == 1
        assert Path(output_path).exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 1
        assert "original_value" in loaded.columns

    def test_generate_empty_report(self, tmp_path: Path) -> None:
        """Empty records should produce a CSV with headers only."""
        output_path = str(tmp_path / "empty_report.csv")
        generator = AnomalyReportGenerator()
        report = generator.generate_report([], output_path)
        assert len(report) == 0
        assert Path(output_path).exists()

    def test_report_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Report generator should create parent directories."""
        output_path = str(tmp_path / "nested" / "dir" / "report.csv")
        generator = AnomalyReportGenerator()
        generator.generate_report([], output_path)
        assert Path(output_path).exists()
