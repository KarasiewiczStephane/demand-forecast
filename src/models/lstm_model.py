"""PyTorch LSTM model for time series forecasting.

Implements a sequence-to-one LSTM architecture with configurable
lookback window, hidden size, early stopping, and model checkpointing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for windowed time series sequences.

    Attributes:
        data: Tensor of shape (n_samples, n_features).
        lookback: Number of past timesteps per input window.
        horizon: Number of future timesteps to predict.
    """

    def __init__(self, data: np.ndarray, lookback: int, horizon: int) -> None:
        """Initialize the dataset.

        Args:
            data: 2D array of shape (n_timesteps, n_features).
            lookback: Input sequence length.
            horizon: Output prediction length.
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        """Return the number of available sequences."""
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an (input, target) pair.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of input tensor and target tensor.
        """
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0]
        return x, y


class LSTMForecaster(nn.Module):
    """LSTM neural network for time series forecasting.

    Attributes:
        hidden_size: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        horizon: Number of output timesteps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 28,
    ) -> None:
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate between layers.
            horizon: Number of output timesteps.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM and fully connected layers.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, horizon).
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


@dataclass
class TrainingConfig:
    """Configuration for LSTM training.

    Attributes:
        lookback: Input sequence length in days.
        horizon: Prediction horizon in days.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate.
        learning_rate: Optimizer learning rate.
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience in epochs.
        device: Compute device ('cpu' or 'cuda').
    """

    lookback: int = 28
    horizon: int = 28
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    device: str = "cpu"


class LSTMTrainer:
    """Training pipeline for the LSTM forecaster.

    Handles data normalization, training with early stopping,
    prediction, and model persistence.

    Attributes:
        config: Training hyperparameters.
        model: Trained LSTMForecaster instance.
        scaler_params: Min/max parameters for denormalization.
        train_losses: Per-epoch training loss history.
        val_losses: Per-epoch validation loss history.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration (defaults to TrainingConfig()).
        """
        self.config = config or TrainingConfig()
        self.model: LSTMForecaster | None = None
        self.scaler_params: dict[str, np.ndarray] = {}
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply min-max normalization.

        Args:
            data: Array to normalize.

        Returns:
            Normalized array with values in [0, 1].
        """
        self.scaler_params["min"] = data.min(axis=0)
        self.scaler_params["max"] = data.max(axis=0)
        denom = self.scaler_params["max"] - self.scaler_params["min"] + 1e-8
        return (data - self.scaler_params["min"]) / denom

    def _denormalize(self, data: np.ndarray, col_idx: int = 0) -> np.ndarray:
        """Reverse min-max normalization for a single column.

        Args:
            data: Normalized values to reverse.
            col_idx: Column index for min/max parameters.

        Returns:
            Denormalized array.
        """
        min_val = self.scaler_params["min"][col_idx]
        max_val = self.scaler_params["max"][col_idx]
        return data * (max_val - min_val) + min_val

    def prepare_data(
        self,
        data: np.ndarray,
        val_ratio: float = 0.2,
    ) -> tuple[DataLoader, DataLoader]:
        """Prepare train/validation DataLoaders from a numpy array.

        Args:
            data: 2D array with target as the first column.
            val_ratio: Fraction of data reserved for validation.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        normalized = self._normalize(data)

        split_idx = int(len(normalized) * (1 - val_ratio))
        train_data = normalized[:split_idx]
        val_data = normalized[split_idx:]

        train_ds = TimeSeriesDataset(
            train_data, self.config.lookback, self.config.horizon
        )
        val_ds = TimeSeriesDataset(val_data, self.config.lookback, self.config.horizon)

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False
        )

        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> LSTMTrainer:
        """Train the LSTM model with early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.

        Returns:
            Self for method chaining.
        """
        input_size = train_loader.dataset.data.shape[1]
        device = self.config.device

        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            horizon=self.config.horizon,
        ).to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = self.model(x)
                    val_loss += criterion(pred, y).item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        logger.info("Training complete. Best val loss: %.6f", best_val_loss)
        return self

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """Generate a forecast from a normalized input sequence.

        Args:
            input_sequence: 2D array of shape (lookback, n_features).

        Returns:
            Denormalized prediction array of shape (horizon,).

        Raises:
            RuntimeError: If model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.config.device)
            pred = self.model(x).cpu().numpy()[0]
            return self._denormalize(pred)

    def save(self, path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            path: File path for saving.
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "scaler_params": self.scaler_params,
                "input_size": self.model.lstm.input_size,
            },
            path,
        )
        logger.info("LSTM model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> LSTMTrainer:
        """Load a trained model from a checkpoint.

        Args:
            path: File path to load from.

        Returns:
            LSTMTrainer with restored model and parameters.
        """
        checkpoint = torch.load(path, weights_only=False)
        trainer = cls(checkpoint["config"])
        trainer.scaler_params = checkpoint["scaler_params"]

        trainer.model = LSTMForecaster(
            input_size=checkpoint["input_size"],
            hidden_size=trainer.config.hidden_size,
            num_layers=trainer.config.num_layers,
            dropout=trainer.config.dropout,
            horizon=trainer.config.horizon,
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        return trainer
