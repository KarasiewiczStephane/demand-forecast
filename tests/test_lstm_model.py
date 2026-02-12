"""Tests for the PyTorch LSTM forecasting model."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.lstm_model import (
    LSTMForecaster,
    LSTMTrainer,
    TimeSeriesDataset,
    TrainingConfig,
)


def _make_data(n: int = 200, n_features: int = 3) -> np.ndarray:
    """Generate synthetic time series data."""
    np.random.seed(42)
    t = np.arange(n)
    target = 100 + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 2, n)
    features = np.random.normal(0, 1, (n, n_features - 1))
    return np.column_stack([target, features]).astype(np.float32)


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset class."""

    def test_length(self) -> None:
        """Dataset length should account for lookback and horizon."""
        data = np.random.rand(100, 3).astype(np.float32)
        ds = TimeSeriesDataset(data, lookback=10, horizon=5)
        assert len(ds) == 100 - 10 - 5 + 1

    def test_shapes(self) -> None:
        """Input and target should have correct shapes."""
        data = np.random.rand(50, 4).astype(np.float32)
        ds = TimeSeriesDataset(data, lookback=7, horizon=3)
        x, y = ds[0]
        assert x.shape == (7, 4)
        assert y.shape == (3,)

    def test_target_is_first_column(self) -> None:
        """Target should be extracted from the first column."""
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        ds = TimeSeriesDataset(data, lookback=3, horizon=2)
        _, y = ds[0]
        expected = data[3:5, 0]
        np.testing.assert_array_equal(y.numpy(), expected)


class TestLSTMForecaster:
    """Tests for LSTMForecaster module."""

    def test_forward_shape(self) -> None:
        """Forward pass should produce correct output shape."""
        model = LSTMForecaster(input_size=3, hidden_size=16, horizon=7)
        x = torch.randn(4, 10, 3)  # batch=4, seq=10, features=3
        output = model(x)
        assert output.shape == (4, 7)

    def test_no_nan_output(self) -> None:
        """Output should not contain NaN values."""
        model = LSTMForecaster(input_size=2, hidden_size=8, horizon=5)
        x = torch.randn(2, 7, 2)
        output = model(x)
        assert not torch.isnan(output).any()

    def test_single_layer(self) -> None:
        """Single-layer LSTM should work without dropout error."""
        model = LSTMForecaster(
            input_size=2, hidden_size=8, num_layers=1, dropout=0.5, horizon=3
        )
        x = torch.randn(1, 5, 2)
        output = model(x)
        assert output.shape == (1, 3)


class TestLSTMTrainer:
    """Tests for LSTMTrainer class."""

    def test_normalize_denormalize_roundtrip(self) -> None:
        """Normalization followed by denormalization should recover values."""
        trainer = LSTMTrainer()
        data = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        normalized = trainer._normalize(data)
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        recovered = trainer._denormalize(normalized[:, 0], col_idx=0)
        np.testing.assert_allclose(recovered, data[:, 0], atol=1e-5)

    def test_prepare_data(self) -> None:
        """prepare_data should return train and val DataLoaders."""
        config = TrainingConfig(lookback=5, horizon=3, batch_size=4)
        trainer = LSTMTrainer(config)
        data = _make_data(100, 3)
        train_loader, val_loader = trainer.prepare_data(data, val_ratio=0.2)
        assert len(train_loader) > 0
        assert len(val_loader) > 0

    def test_train_runs(self) -> None:
        """Training should run without errors and record losses."""
        config = TrainingConfig(
            lookback=7,
            horizon=3,
            hidden_size=8,
            num_layers=1,
            epochs=5,
            patience=3,
            batch_size=8,
        )
        trainer = LSTMTrainer(config)
        data = _make_data(100, 2)
        train_loader, val_loader = trainer.prepare_data(data)
        trainer.train(train_loader, val_loader)

        assert len(trainer.train_losses) > 0
        assert len(trainer.val_losses) > 0
        assert trainer.model is not None

    def test_early_stopping(self) -> None:
        """Training should stop before max epochs with patience."""
        config = TrainingConfig(
            lookback=5,
            horizon=2,
            hidden_size=4,
            num_layers=1,
            epochs=1000,
            patience=2,
            batch_size=16,
        )
        trainer = LSTMTrainer(config)
        data = _make_data(80, 2)
        train_loader, val_loader = trainer.prepare_data(data)
        trainer.train(train_loader, val_loader)
        assert len(trainer.train_losses) < 1000

    def test_predict(self) -> None:
        """Predictions should be numeric and finite."""
        config = TrainingConfig(
            lookback=7,
            horizon=3,
            hidden_size=8,
            num_layers=1,
            epochs=3,
            batch_size=8,
        )
        trainer = LSTMTrainer(config)
        data = _make_data(100, 2)
        train_loader, val_loader = trainer.prepare_data(data)
        trainer.train(train_loader, val_loader)

        input_seq = trainer._normalize(data[-7:])
        pred = trainer.predict(input_seq)
        assert pred.shape == (3,)
        assert np.all(np.isfinite(pred))

    def test_predict_without_train_raises(self) -> None:
        """Predicting without training should raise RuntimeError."""
        trainer = LSTMTrainer()
        with pytest.raises(RuntimeError):
            trainer.predict(np.zeros((7, 2)))

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded model should produce same predictions."""
        config = TrainingConfig(
            lookback=7,
            horizon=3,
            hidden_size=8,
            num_layers=1,
            epochs=3,
            batch_size=8,
        )
        trainer = LSTMTrainer(config)
        data = _make_data(100, 2)
        train_loader, val_loader = trainer.prepare_data(data)
        trainer.train(train_loader, val_loader)

        save_path = str(tmp_path / "lstm.pt")
        trainer.save(save_path)

        loaded = LSTMTrainer.load(save_path)
        # Use a fixed normalized input to avoid scaler_params drift
        input_seq = np.random.rand(7, 2).astype(np.float32)
        trainer.model.eval()
        loaded.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(input_seq).unsqueeze(0)
            pred1 = trainer.model(x).numpy()[0]
            pred2 = loaded.model(x).numpy()[0]
        np.testing.assert_allclose(pred1, pred2, atol=1e-5)

    def test_save_without_model_raises(self, tmp_path: Path) -> None:
        """Saving without a model should raise RuntimeError."""
        trainer = LSTMTrainer()
        with pytest.raises(RuntimeError):
            trainer.save(str(tmp_path / "nomodel.pt"))

    def test_training_config_defaults(self) -> None:
        """Default config should have expected values."""
        config = TrainingConfig()
        assert config.lookback == 28
        assert config.horizon == 28
        assert config.device == "cpu"
