"""Entry point for the demand forecasting system.

Provides CLI commands for data download, model training,
backtesting, and launching the dashboard.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def setup_logging(config_path: str = "configs/config.yaml") -> None:
    """Configure root logger from config file."""
    config = load_config(config_path)
    logging.basicConfig(
        level=config.logging["level"],
        format=config.logging["format"],
    )


def _load_and_preprocess(config) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load raw data, merge datasets, and engineer features."""
    from src.data.feature_engineer import FeatureEngineer
    from src.data.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor()
    datasets = preprocessor.load_store_sales_data(config.data["raw_path"])
    df = preprocessor.merge_datasets(datasets)
    holidays_df = datasets.get("holidays")

    fe = FeatureEngineer()
    df = fe.engineer_all_features(df, holidays_df=holidays_df)
    return df, holidays_df


def run_train(args, config) -> None:
    """Train forecasting models and save checkpoints."""
    df, holidays_df = _load_and_preprocess(config)
    checkpoint_dir = Path(config.models["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Use a subset for tractable training: stores 1-3
    subset = df[df["store_nbr"].isin([1, 2, 3])].copy()
    logger.info("Training on subset: %d rows", len(subset))

    models_to_train = args.models
    horizon = config.models["forecast_horizons"][0]  # 7

    if models_to_train in ("all", "prophet"):
        logger.info("Training Prophet models...")
        from src.models.prophet_model import ProphetMultiStore

        pm = ProphetMultiStore(forecast_horizon=horizon)
        pm.fit_all(subset, holidays_df=holidays_df)
        for key, model in pm.models.items():
            store, family = key
            safe_family = family.replace("/", "_").replace(" ", "_")
            model.save(str(checkpoint_dir / f"prophet_s{store}_{safe_family}.pkl"))
        logger.info("Prophet training complete: %d models", len(pm.models))

    if models_to_train in ("all", "xgboost"):
        logger.info("Training XGBoost models...")
        from src.models.xgboost_model import XGBoostMultiStore

        xm = XGBoostMultiStore()
        xm.fit_all(subset, target_col="sales", min_samples=100)
        for key, model in xm.models.items():
            store, family = key
            safe_family = family.replace("/", "_").replace(" ", "_")
            model.save(str(checkpoint_dir / f"xgboost_s{store}_{safe_family}"))
        logger.info("XGBoost training complete: %d models", len(xm.models))

    if models_to_train in ("all", "lstm"):
        logger.info("Training LSTM model...")
        from src.models.lstm_model import LSTMTrainer, TrainingConfig

        # Aggregate daily sales for a univariate LSTM
        daily = subset.groupby("date")["sales"].sum().sort_index().values.reshape(-1, 1)
        lstm_config = TrainingConfig(
            lookback=config.models["lstm_lookback"],
            horizon=horizon,
            epochs=50,
            patience=10,
        )
        trainer = LSTMTrainer(lstm_config)
        train_loader, val_loader = trainer.prepare_data(daily)
        trainer.train(train_loader, val_loader)
        trainer.save(str(checkpoint_dir / "lstm_aggregate.pt"))
        logger.info("LSTM training complete")


def run_detect_anomalies(config) -> None:
    """Run anomaly detection and save report."""
    from src.data.anomaly_detector import (
        AnomalyDetector,
        AnomalyHandler,
        AnomalyReportGenerator,
        HandlingStrategy,
    )

    logger.info("Loading sales data for anomaly detection...")
    raw_path = Path(config.data["raw_path"])
    df = pd.read_csv(raw_path / "train.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Run on a subset for speed
    subset = df[df["store_nbr"].isin([1, 2, 3])].copy()

    detector = AnomalyDetector(iqr_multiplier=1.5, contamination=0.01)
    result = detector.detect_anomalies(subset, target_col="sales")

    handler = AnomalyHandler(detector)
    handler.handle_anomalies(result, HandlingStrategy.INTERPOLATE, target_col="sales")

    report_path = "data/reports/anomaly_report.csv"
    reporter = AnomalyReportGenerator()
    reporter.generate_report(detector.anomaly_records, report_path)
    logger.info("Anomaly detection complete")


def run_backtest(config) -> None:
    """Run backtesting evaluation and save report."""
    from src.evaluation.backtester import Backtester

    logger.info("Loading data for backtesting...")
    raw_path = Path(config.data["raw_path"])
    df = pd.read_csv(raw_path / "train.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate daily sales for store 1, top family for speed
    store1 = df[df["store_nbr"] == 1].copy()
    top_family = store1.groupby("family")["sales"].sum().idxmax()
    subset = store1[store1["family"] == top_family][["date", "sales"]].copy()
    subset = subset.groupby("date")["sales"].sum().reset_index()
    subset = subset.sort_values("date")

    logger.info("Backtesting on store 1, family '%s': %d rows", top_family, len(subset))

    bt_config = config.backtesting
    backtester = Backtester(
        train_window_months=bt_config["train_window_months"],
        test_window_days=bt_config["test_window_days"],
        step_days=bt_config["step_days"],
    )

    def naive_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Repeat last known value."""
        last_val = train_df["sales"].iloc[-1]
        return np.full(len(test_df), last_val)

    def mean_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Predict rolling 28-day mean."""
        return np.full(len(test_df), train_df["sales"].iloc[-28:].mean())

    def prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """Fit Prophet on train and predict test period."""
        from src.models.prophet_model import ProphetForecaster

        pf = ProphetForecaster(forecast_horizon=len(test_df))
        pf.fit(train_df)
        result = pf.predict(periods=len(test_df))
        pred = result.forecast_df["yhat"].values[: len(test_df)]
        return np.clip(pred, 0, None)

    models = {
        "naive": naive_forecast,
        "mean_28d": mean_forecast,
        "prophet": prophet_forecast,
    }

    report = backtester.run(subset, models, target_col="sales", date_col="date")

    # Flatten to CSV
    rows = []
    for result in report.results:
        for model_name, metrics in result.metrics.items():
            rows.append(
                {
                    "window_id": result.window.window_id,
                    "test_start": result.window.test_start.strftime("%Y-%m-%d"),
                    "model": model_name,
                    **metrics,
                }
            )
    report_df = pd.DataFrame(rows)
    report_path = Path("data/reports/backtest_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    logger.info(
        "Backtest report saved: %d rows across %d windows",
        len(report_df),
        len(report.results),
    )

    # Log aggregate metrics
    for model_name, agg in report.aggregate_metrics.items():
        logger.info("  %s: %s", model_name, {k: f"{v:.2f}" for k, v in agg.items()})


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
    config = load_config()

    if args.command == "download":
        from src.data.downloader import KaggleDownloader

        downloader = KaggleDownloader(
            competition=config.data["kaggle_competition"],
            download_path=config.data["raw_path"],
        )
        downloader.download()

    elif args.command == "train":
        run_train(args, config)

    elif args.command == "detect-anomalies":
        run_detect_anomalies(config)

    elif args.command == "backtest":
        run_backtest(config)

    elif args.command == "dashboard":
        import subprocess

        subprocess.run(
            ["streamlit", "run", "src/dashboard/app.py"],
            check=True,
        )


if __name__ == "__main__":
    main()
