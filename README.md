# Demand Forecasting System

> Multi-model time series forecasting for retail demand prediction

## Overview

A production-ready demand forecasting system combining three ML models (Prophet, LSTM, XGBoost) with ensemble learning. Features automatic seasonality detection, anomaly handling, and an interactive Streamlit dashboard. Built on the Kaggle Store Sales Time Series Forecasting dataset.

### Key Features

- **Multi-model forecasting**: Prophet, PyTorch LSTM, XGBoost with optimized ensemble
- **Anomaly detection**: IQR + Isolation Forest with configurable handling strategies
- **Seasonality analysis**: Fourier transform and ACF/PACF for pattern detection
- **Feature engineering**: Lag features, rolling statistics, temporal indicators, holiday effects
- **Backtesting framework**: Rolling window cross-validation with Diebold-Mariano tests
- **Interactive dashboard**: Streamlit + Plotly visualizations with 5 pages
- **CI/CD**: GitHub Actions with linting, testing, and Docker build

## Architecture

```
+-----------------------------------------------------------+
|                   Streamlit Dashboard                      |
|  +----------+ +----------+ +---------+ +---------------+  |
|  | Forecast | |  Model   | | Anomaly | | Backtesting   |  |
|  |  View    | | Compare  | | Explorer| | Results       |  |
|  +----+-----+ +----+-----+ +----+----+ +------+--------+  |
+-------+------------+------------+-------------+-----------+
        |            |            |             |
+-------+------------+------------+-------------+-----------+
|                      Model Layer                          |
|  +---------+    +---------+    +---------+                |
|  | Prophet |    |  LSTM   |    | XGBoost |                |
|  +----+----+    +----+----+    +----+----+                |
|       +              |              +                     |
|                 +----+----+                               |
|                 | Ensemble|                               |
|                 +---------+                               |
+-----------------------------------------------------------+
                          |
+-----------------------------------------------------------+
|                     Data Layer                            |
|  +------------+ +--------------+ +--------------------+   |
|  |   DuckDB   | |   Feature    | |     Anomaly        |   |
|  | raw/cleaned| |  Engineering | |     Detection      |   |
|  +------------+ +--------------+ +--------------------+   |
+-----------------------------------------------------------+
```

## Quick Start

### Prerequisites

- Python 3.11+
- Kaggle account (for dataset download)

### Installation

```bash
# Clone repository
git clone https://github.com/KarasiewiczStephane/demand-forecast.git
cd demand-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

### Download Data & Train Models

```bash
# Download Store Sales dataset
python -m src.main download

# Run anomaly detection
python -m src.main detect-anomalies

# Train all models
python -m src.main train --models all

# Run backtesting
python -m src.main backtest
```

### Launch Dashboard

```bash
make dashboard
# or
streamlit run src/dashboard/app.py
```

Open http://localhost:8501 in your browser.

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# Or manually
make docker-build
make docker-run
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Forecast View** | Historical sales + ensemble forecast with confidence intervals, filterable by store/family |
| **Model Comparison** | Side-by-side metrics table, error distributions, rolling MAPE |
| **Anomaly Explorer** | Timeline of detected anomalies, filter by method/store/family |
| **Seasonality Patterns** | Seasonal decomposition, ACF/PACF analysis |
| **Backtesting Results** | Rolling accuracy, best-model-per-window, aggregate metrics |

## Project Structure

```
demand-forecast/
├── src/
│   ├── data/           # Data pipeline (download, validate, anomaly, features)
│   ├── models/         # Prophet, LSTM, XGBoost, Ensemble
│   ├── evaluation/     # Backtesting, metrics, report generation
│   ├── dashboard/      # Streamlit app with 5 pages
│   └── utils/          # Config loader, structured logging
├── tests/              # Unit tests (>80% coverage)
├── configs/            # YAML configuration
├── data/               # Raw, processed, and report data
├── models/             # Model checkpoints
├── .github/workflows/  # CI pipeline (lint, test, docker)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Configuration

All parameters are centralized in `configs/config.yaml`:

- **Forecast horizons**: 7, 14, 28 days
- **LSTM lookback**: 28 days
- **Backtesting**: 12-month train window, 28-day test, 7-day step
- **Anomaly thresholds**: IQR multiplier, Isolation Forest contamination

## Models

| Model | Description |
|-------|-------------|
| **Prophet** | Additive time series model with Ecuadorian holidays and changepoint detection |
| **LSTM** | PyTorch sequence model with configurable lookback, early stopping, gradient clipping |
| **XGBoost** | Gradient boosting on engineered features with per-store training |
| **Ensemble** | Weighted average with SLSQP-optimized weights minimizing RMSE |

## Testing

```bash
# Run all tests
make test

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run local CI checks
make ci-local
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make test` | Run tests with coverage |
| `make lint` | Run ruff linter and formatter |
| `make dashboard` | Launch Streamlit dashboard |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |
| `make ci-local` | Run full CI checks locally |
| `make clean` | Remove caches |

## License

MIT
