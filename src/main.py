"""Entry point for the demand forecasting system.

Provides CLI commands for data download, model training,
backtesting, and launching the dashboard.
"""

import argparse
import logging
import sys

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def setup_logging(config_path: str = "configs/config.yaml") -> None:
    """Configure root logger from config file.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=config.logging["level"],
        format=config.logging["format"],
    )


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        description="Demand Forecasting System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("download", help="Download dataset from Kaggle")
    subparsers.add_parser("detect-anomalies", help="Run anomaly detection pipeline")

    train_parser = subparsers.add_parser("train", help="Train forecasting models")
    train_parser.add_argument(
        "--models",
        default="all",
        choices=["all", "prophet", "lstm", "xgboost"],
        help="Which models to train",
    )

    subparsers.add_parser("backtest", help="Run backtesting evaluation")
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    args = parser.parse_args()
    setup_logging()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logger.info("Running command: %s", args.command)

    if args.command == "dashboard":
        import subprocess

        subprocess.run(
            ["streamlit", "run", "src/dashboard/app.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
