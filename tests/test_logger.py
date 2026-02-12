"""Tests for the structured logging module."""

import logging
from pathlib import Path

from src.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for the setup_logger function."""

    def test_returns_logger(self, tmp_config: Path) -> None:
        """setup_logger should return a Logger instance."""
        logger = setup_logger("test_module", config_path=str(tmp_config))
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self, tmp_config: Path) -> None:
        """Logger name should match the provided name."""
        logger = setup_logger("my_module", config_path=str(tmp_config))
        assert logger.name == "my_module"

    def test_logger_has_handler(self, tmp_config: Path) -> None:
        """Logger should have at least one handler configured."""
        logger = setup_logger("handler_test", config_path=str(tmp_config))
        assert len(logger.handlers) >= 1

    def test_logger_level(self, tmp_config: Path) -> None:
        """Logger level should match config setting."""
        logger = setup_logger("level_test", config_path=str(tmp_config))
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self, tmp_config: Path) -> None:
        """Calling setup_logger twice should not add duplicate handlers."""
        logger1 = setup_logger("dup_test", config_path=str(tmp_config))
        count1 = len(logger1.handlers)
        logger2 = setup_logger("dup_test", config_path=str(tmp_config))
        assert len(logger2.handlers) == count1

    def test_logger_output(self, tmp_config: Path, capfd: object) -> None:
        """Logger should produce formatted output."""
        logger = setup_logger("output_test", config_path=str(tmp_config))
        logger.info("test message")
        captured = capfd.readouterr()  # type: ignore[attr-defined]
        assert "test message" in captured.err
