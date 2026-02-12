"""Structured logging setup for the demand forecasting system.

Provides a factory function to create configured loggers with
consistent formatting across all modules.
"""

import logging

from src.utils.config import load_config


def setup_logger(name: str, config_path: str = "configs/config.yaml") -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        config_path: Path to the YAML configuration file.

    Returns:
        Configured logger instance with stream handler.
    """
    config = load_config(config_path)
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(config.logging["level"])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config.logging["format"]))
        logger.addHandler(handler)

    return logger
