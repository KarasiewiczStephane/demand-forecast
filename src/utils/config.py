"""Configuration management for the demand forecasting system.

Loads YAML configuration files and provides structured access
to project settings including data paths, model parameters,
backtesting configuration, and logging settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Application configuration container.

    Attributes:
        data: Data paths and dataset configuration.
        models: Model training parameters and checkpoint paths.
        backtesting: Backtesting window and step configuration.
        logging: Logging level and format settings.
    """

    data: dict[str, Any]
    models: dict[str, Any]
    backtesting: dict[str, Any]
    logging: dict[str, Any]


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with parsed settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)
