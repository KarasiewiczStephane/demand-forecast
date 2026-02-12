"""Report generation for backtesting results.

Converts BacktestReport data into CSV files for downstream
analysis and dashboard consumption.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.evaluation.backtester import BacktestReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate CSV reports from backtest results.

    Converts per-window, per-model metrics into a tidy DataFrame
    and writes it to disk.
    """

    @staticmethod
    def generate_csv_report(report: BacktestReport, output_path: str) -> pd.DataFrame:
        """Generate a CSV report of backtesting results.

        Each row represents one model evaluated on one backtest window.

        Args:
            report: BacktestReport with per-window results.
            output_path: File path to write the CSV.

        Returns:
            DataFrame with the report data.
        """
        rows: list[dict] = []

        for result in report.results:
            for model_name, metrics in result.metrics.items():
                row = {
                    "window_id": result.window.window_id,
                    "train_start": result.window.train_start,
                    "train_end": result.window.train_end,
                    "test_start": result.window.test_start,
                    "test_end": result.window.test_end,
                    "model": model_name,
                    **metrics,
                }
                rows.append(row)

        df = pd.DataFrame(rows)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Backtest report saved to %s", output_path)

        return df
