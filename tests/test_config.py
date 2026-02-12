"""Tests for configuration management module."""

from pathlib import Path

import pytest

from src.utils.config import Config, load_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, tmp_config: Path) -> None:
        """Valid YAML should produce a Config with all sections."""
        config = load_config(str(tmp_config))
        assert isinstance(config, Config)
        assert "raw_path" in config.data
        assert "forecast_horizons" in config.models
        assert "train_window_months" in config.backtesting
        assert "level" in config.logging

    def test_config_data_values(self, tmp_config: Path) -> None:
        """Config data section should contain expected keys."""
        config = load_config(str(tmp_config))
        assert (
            config.data["kaggle_competition"] == "store-sales-time-series-forecasting"
        )

    def test_config_models_values(self, tmp_config: Path) -> None:
        """Config models section should have correct forecast horizons."""
        config = load_config(str(tmp_config))
        assert config.models["forecast_horizons"] == [7, 14, 28]
        assert config.models["lstm_lookback"] == 28

    def test_config_backtesting_values(self, tmp_config: Path) -> None:
        """Config backtesting section should have correct window settings."""
        config = load_config(str(tmp_config))
        assert config.backtesting["train_window_months"] == 12
        assert config.backtesting["test_window_days"] == 28

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed YAML should raise an error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("data:\n  key: [unterminated")
        with pytest.raises(Exception):
            load_config(str(bad_file))

    def test_default_config_path(self) -> None:
        """Default config path should load configs/config.yaml."""
        config = load_config()
        assert isinstance(config, Config)
        assert config.logging["level"] == "INFO"
