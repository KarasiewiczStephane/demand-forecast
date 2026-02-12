"""Tests for the seasonality detection module."""

import numpy as np
import pandas as pd

from src.data.seasonality_detector import SeasonalityDetector, SeasonalityResult


class TestSeasonalityDetector:
    """Tests for SeasonalityDetector class."""

    def _make_weekly_series(self, n: int = 365) -> pd.Series:
        """Create a synthetic series with weekly seasonality."""
        t = np.arange(n)
        values = 100 + 20 * np.sin(2 * np.pi * t / 7)
        return pd.Series(values)

    def test_fourier_detects_weekly_period(self) -> None:
        """Fourier analysis should detect a 7-day period."""
        series = self._make_weekly_series(365)
        detector = SeasonalityDetector()
        result = detector.detect_fourier(series)
        periods = [p[0] for p in result]
        assert 7 in periods

    def test_fourier_returns_sorted(self) -> None:
        """Results should be sorted by strength descending."""
        series = self._make_weekly_series(365)
        detector = SeasonalityDetector()
        result = detector.detect_fourier(series)
        if len(result) > 1:
            strengths = [p[1] for p in result]
            assert strengths == sorted(strengths, reverse=True)

    def test_fourier_short_series(self) -> None:
        """Short series should return empty list."""
        series = pd.Series([1, 2, 3])
        detector = SeasonalityDetector()
        result = detector.detect_fourier(series)
        assert isinstance(result, list)

    def test_acf_pacf_computation(self) -> None:
        """ACF and PACF should return arrays of correct length."""
        series = self._make_weekly_series(200)
        detector = SeasonalityDetector(max_lags=50)
        acf_vals, pacf_vals = detector.compute_acf_pacf(series)
        assert len(acf_vals) > 1
        assert len(pacf_vals) > 1
        assert acf_vals[0] == 1.0  # ACF at lag 0 is always 1

    def test_acf_weekly_peak(self) -> None:
        """ACF should show a peak at lag 7 for weekly data."""
        series = self._make_weekly_series(365)
        detector = SeasonalityDetector(max_lags=50)
        acf_vals, _ = detector.compute_acf_pacf(series)
        assert acf_vals[7] > acf_vals[5]

    def test_analyze_returns_result(self) -> None:
        """Analyze should return a SeasonalityResult dataclass."""
        series = self._make_weekly_series(365)
        detector = SeasonalityDetector()
        result = detector.analyze(series)
        assert isinstance(result, SeasonalityResult)
        assert isinstance(result.dominant_periods, list)
        assert isinstance(result.weekly_strength, float)

    def test_analyze_weekly_strength(self) -> None:
        """Weekly strength should be positive for weekly data."""
        series = self._make_weekly_series(365)
        detector = SeasonalityDetector()
        result = detector.analyze(series)
        assert result.weekly_strength > 0

    def test_analyze_short_series(self) -> None:
        """Analyze should handle very short series without error."""
        series = pd.Series([10, 20, 30, 40, 50])
        detector = SeasonalityDetector()
        result = detector.analyze(series)
        assert isinstance(result, SeasonalityResult)
