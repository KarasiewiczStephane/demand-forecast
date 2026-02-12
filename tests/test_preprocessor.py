"""Tests for data validation and preprocessing modules."""

from pathlib import Path

import pandas as pd

from src.data.preprocessor import DataPreprocessor, DataValidator, ValidationResult


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_valid_data(self) -> None:
        """Valid data should produce is_valid=True."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=30, freq="D"),
                "sales": range(30),
            }
        )
        result = DataValidator().validate_sales_data(df)
        assert result.is_valid is True
        assert result.negative_sales_count == 0
        assert len(result.missing_dates) == 0

    def test_missing_dates_detected(self) -> None:
        """Gaps in the date series should be detected."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"])
        df = pd.DataFrame({"date": dates, "sales": [10, 20, 30]})
        result = DataValidator().validate_sales_data(df)
        assert result.is_valid is False
        assert len(result.missing_dates) == 2  # Jan 3, Jan 4

    def test_negative_sales_detected(self) -> None:
        """Negative sales values should be flagged."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": [10, -5, 20, -3, 15],
            }
        )
        result = DataValidator().validate_sales_data(df)
        assert result.is_valid is False
        assert result.negative_sales_count == 2

    def test_missing_values_counted(self) -> None:
        """Missing values should be counted per column."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": [10, None, 20, None, 15],
                "extra": [1, 2, None, 4, 5],
            }
        )
        result = DataValidator().validate_sales_data(df)
        assert result.missing_values_count["sales"] == 2
        assert result.missing_values_count["extra"] == 1

    def test_warnings_generated(self) -> None:
        """Warnings should be generated for each issue found."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-03"])
        df = pd.DataFrame({"date": dates, "sales": [-1, 10]})
        result = DataValidator().validate_sales_data(df)
        assert len(result.warnings) >= 2

    def test_validation_result_type(self) -> None:
        """Result should be a ValidationResult dataclass."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "sales": [1, 2, 3],
            }
        )
        result = DataValidator().validate_sales_data(df)
        assert isinstance(result, ValidationResult)

    def test_all_zero_sales(self) -> None:
        """Zero sales should not be flagged as negative."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="D"),
                "sales": [0, 0, 0, 0, 0],
            }
        )
        result = DataValidator().validate_sales_data(df)
        assert result.negative_sales_count == 0


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_load_store_sales_data(self, tmp_path: Path) -> None:
        """Should load all CSV files that exist."""
        # Create sample CSV files
        pd.DataFrame({"date": ["2023-01-01"], "sales": [10]}).to_csv(
            tmp_path / "train.csv", index=False
        )
        pd.DataFrame({"store_nbr": [1], "city": ["Quito"]}).to_csv(
            tmp_path / "stores.csv", index=False
        )

        preprocessor = DataPreprocessor()
        datasets = preprocessor.load_store_sales_data(str(tmp_path))

        assert "train" in datasets
        assert "stores" in datasets
        assert len(datasets["train"]) == 1

    def test_missing_files_handled(self, tmp_path: Path) -> None:
        """Missing files should be skipped without error."""
        preprocessor = DataPreprocessor()
        datasets = preprocessor.load_store_sales_data(str(tmp_path))
        assert len(datasets) == 0

    def test_merge_datasets(self) -> None:
        """Merge should combine train with stores and oil."""
        datasets = {
            "train": pd.DataFrame(
                {
                    "date": ["2023-01-01", "2023-01-01"],
                    "store_nbr": [1, 2],
                    "sales": [100, 200],
                }
            ),
            "stores": pd.DataFrame(
                {
                    "store_nbr": [1, 2],
                    "city": ["Quito", "Guayaquil"],
                }
            ),
            "oil": pd.DataFrame(
                {
                    "date": ["2023-01-01"],
                    "dcoilwtico": [75.5],
                }
            ),
            "transactions": pd.DataFrame(
                {
                    "date": ["2023-01-01", "2023-01-01"],
                    "store_nbr": [1, 2],
                    "transactions": [1500, 2000],
                }
            ),
        }

        preprocessor = DataPreprocessor()
        merged = preprocessor.merge_datasets(datasets)

        assert "city" in merged.columns
        assert "dcoilwtico" in merged.columns
        assert "transactions" in merged.columns
        assert len(merged) == 2

    def test_merge_without_optional_datasets(self) -> None:
        """Merge should work with only the train dataset."""
        datasets = {
            "train": pd.DataFrame(
                {
                    "date": ["2023-01-01"],
                    "store_nbr": [1],
                    "sales": [100],
                }
            ),
        }

        preprocessor = DataPreprocessor()
        merged = preprocessor.merge_datasets(datasets)
        assert len(merged) == 1
        assert "sales" in merged.columns
