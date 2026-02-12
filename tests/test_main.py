"""Tests for the main CLI entry point."""

from unittest.mock import patch

import pytest

from src.main import main, setup_logging


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """setup_logging should configure root logger without errors."""
        setup_logging()

    def test_setup_logging_custom_config(self, tmp_config: object) -> None:
        """setup_logging should accept a custom config path."""
        setup_logging(str(tmp_config))


class TestMain:
    """Tests for the main CLI dispatcher."""

    def test_no_command_exits(self) -> None:
        """No command should print help and exit with code 0."""
        with patch("sys.argv", ["main"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_download_command(self) -> None:
        """Download command should be recognized."""
        with patch("sys.argv", ["main", "download"]):
            main()

    def test_detect_anomalies_command(self) -> None:
        """Detect-anomalies command should be recognized."""
        with patch("sys.argv", ["main", "detect-anomalies"]):
            main()

    def test_train_command(self) -> None:
        """Train command should be recognized."""
        with patch("sys.argv", ["main", "train", "--models", "all"]):
            main()

    def test_backtest_command(self) -> None:
        """Backtest command should be recognized."""
        with patch("sys.argv", ["main", "backtest"]):
            main()

    def test_dashboard_command(self) -> None:
        """Dashboard command should invoke streamlit."""
        with (
            patch("sys.argv", ["main", "dashboard"]),
            patch("subprocess.run") as mock_run,
        ):
            main()
            mock_run.assert_called_once()
