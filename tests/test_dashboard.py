"""Tests for dashboard utility functions and page rendering logic."""

from pathlib import Path

import pandas as pd

from src.dashboard.utils import get_store_families


class TestGetStoreFamilies:
    """Tests for the get_store_families helper."""

    def test_extracts_stores_and_families(self) -> None:
        """Should return sorted unique stores and families."""
        df = pd.DataFrame(
            {
                "store_nbr": [2, 1, 3, 1],
                "family": ["B", "A", "A", "C"],
                "sales": [10, 20, 30, 40],
            }
        )
        stores, families = get_store_families(df)
        assert stores == [1, 2, 3]
        assert families == ["A", "B", "C"]

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame should return empty lists."""
        df = pd.DataFrame()
        stores, families = get_store_families(df)
        assert stores == []
        assert families == []

    def test_missing_columns(self) -> None:
        """Missing columns should return empty lists gracefully."""
        df = pd.DataFrame({"sales": [1, 2, 3]})
        stores, families = get_store_families(df)
        assert stores == []
        assert families == []


class TestDashboardImports:
    """Verify that dashboard modules are importable."""

    def test_import_app(self) -> None:
        """The app module should be importable."""
        import src.dashboard.app  # noqa: F401

    def test_import_utils(self) -> None:
        """The utils module should be importable."""
        import src.dashboard.utils  # noqa: F401

    def test_import_forecast_page(self) -> None:
        """The forecast page module should be importable."""
        import src.dashboard.pages.forecast  # noqa: F401

    def test_import_comparison_page(self) -> None:
        """The comparison page module should be importable."""
        import src.dashboard.pages.comparison  # noqa: F401

    def test_import_anomaly_page(self) -> None:
        """The anomaly page module should be importable."""
        import src.dashboard.pages.anomaly  # noqa: F401

    def test_import_seasonality_page(self) -> None:
        """The seasonality page module should be importable."""
        import src.dashboard.pages.seasonality  # noqa: F401

    def test_import_backtesting_page(self) -> None:
        """The backtesting page module should be importable."""
        import src.dashboard.pages.backtesting  # noqa: F401


class TestLoadBacktestReport:
    """Tests for load_backtest_report helper."""

    def test_missing_file_returns_empty(self) -> None:
        """Should return an empty DataFrame when file doesn't exist."""
        from src.dashboard.utils import load_backtest_report

        load_backtest_report.clear()
        df = load_backtest_report("/nonexistent/path/report.csv")
        assert df.empty

    def test_loads_existing_csv(self, tmp_path: Path) -> None:
        """Should load a CSV correctly."""
        from src.dashboard.utils import load_backtest_report

        load_backtest_report.clear()
        csv_path = str(tmp_path / "report.csv")
        pd.DataFrame({"model": ["a"], "rmse": [1.0]}).to_csv(csv_path, index=False)
        df = load_backtest_report(csv_path)
        assert len(df) == 1
        assert "model" in df.columns


class TestLoadAnomalyReport:
    """Tests for load_anomaly_report helper."""

    def test_missing_file_returns_empty(self) -> None:
        """Should return empty DataFrame for nonexistent path."""
        from src.dashboard.utils import load_anomaly_report

        load_anomaly_report.clear()
        df = load_anomaly_report("/nonexistent/anomaly.csv")
        assert df.empty
