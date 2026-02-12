"""Automatic seasonality detection using Fourier analysis and ACF/PACF.

Identifies dominant seasonal periods in time series data through
frequency-domain analysis and autocorrelation measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)


@dataclass
class SeasonalityResult:
    """Results from seasonality analysis.

    Attributes:
        dominant_periods: Detected seasonal periods in days.
        weekly_strength: ACF value at lag 7.
        monthly_strength: ACF value at lag 30.
        yearly_strength: ACF value at lag 365.
        acf_values: Full autocorrelation function values.
        pacf_values: Partial autocorrelation function values.
    """

    dominant_periods: list[int]
    weekly_strength: float
    monthly_strength: float
    yearly_strength: float
    acf_values: np.ndarray
    pacf_values: np.ndarray


class SeasonalityDetector:
    """Detects seasonal patterns using Fourier and ACF/PACF analysis.

    Attributes:
        max_lags: Maximum number of lags for ACF/PACF computation.
    """

    def __init__(self, max_lags: int = 400) -> None:
        """Initialize the seasonality detector.

        Args:
            max_lags: Maximum number of lags for autocorrelation analysis.
        """
        self.max_lags = max_lags

    def detect_fourier(
        self, series: pd.Series, sampling_rate: float = 1.0
    ) -> list[tuple[int, float]]:
        """Detect dominant frequencies using Fast Fourier Transform.

        Args:
            series: Time series to analyze.
            sampling_rate: Sampling frequency (1.0 for daily data).

        Returns:
            List of (period, strength) tuples sorted by strength descending.
        """
        n = len(series)
        if n < 4:
            return []

        yf = fft(series.values - series.mean())
        xf = fftfreq(n, 1 / sampling_rate)

        positive_mask = xf > 0
        freqs = xf[positive_mask]
        magnitudes = np.abs(yf[positive_mask])

        if len(magnitudes) == 0:
            return []

        peaks, _ = signal.find_peaks(magnitudes, height=magnitudes.mean())

        dominant: list[tuple[int, float]] = []
        mag_max = magnitudes.max()
        for peak in peaks[:5]:
            if freqs[peak] > 0 and mag_max > 0:
                period = int(1 / freqs[peak])
                strength = float(magnitudes[peak] / mag_max)
                if period > 0:
                    dominant.append((period, strength))

        return sorted(dominant, key=lambda x: x[1], reverse=True)

    def compute_acf_pacf(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Compute autocorrelation and partial autocorrelation functions.

        Args:
            series: Time series to analyze (NaN values are dropped).

        Returns:
            Tuple of (acf_values, pacf_values) as numpy arrays.
        """
        clean = series.dropna()
        n = len(clean)
        acf_lags = min(self.max_lags, n // 2 - 1)
        pacf_lags = min(50, n // 4)

        if acf_lags < 1 or pacf_lags < 1:
            return np.array([1.0]), np.array([1.0])

        acf_vals = acf(clean, nlags=acf_lags)
        pacf_vals = pacf(clean, nlags=pacf_lags)
        return acf_vals, pacf_vals

    def analyze(self, series: pd.Series) -> SeasonalityResult:
        """Run full seasonality analysis on a time series.

        Args:
            series: Daily time series values.

        Returns:
            SeasonalityResult with detected periods and ACF/PACF values.
        """
        dominant = self.detect_fourier(series)
        acf_vals, pacf_vals = self.compute_acf_pacf(series)

        weekly = float(acf_vals[7]) if len(acf_vals) > 7 else 0.0
        monthly = float(acf_vals[30]) if len(acf_vals) > 30 else 0.0
        yearly = float(acf_vals[365]) if len(acf_vals) > 365 else 0.0

        logger.info(
            "Seasonality detected: periods=%s, weekly=%.3f, monthly=%.3f",
            [p[0] for p in dominant[:3]],
            weekly,
            monthly,
        )

        return SeasonalityResult(
            dominant_periods=[p[0] for p in dominant],
            weekly_strength=weekly,
            monthly_strength=monthly,
            yearly_strength=yearly,
            acf_values=acf_vals,
            pacf_values=pacf_vals,
        )
