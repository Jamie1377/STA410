"""pytest configuration and fixtures for STA410 tests."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from stock_prediction.utils.config import Config, DataConfig, ModelConfig, TradingConfig, LoggingConfig, APIConfig


@pytest.fixture
def test_config():
    """Provide test configuration with safe defaults."""
    return Config(
        environment="testing",
        logging=LoggingConfig(level="ERROR"),  # Suppress logs during tests
        data=DataConfig(cache_dir="/tmp/test_cache", model_cache_dir="/tmp/test_models"),
        model=ModelConfig(random_seed=42),
        trading=TradingConfig(paper_trading=True),
        api=APIConfig(),
    )


@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.DataFrame({
        "Open": close_prices + np.random.randn(100) * 0.2,
        "High": close_prices + np.abs(np.random.randn(100) * 0.3),
        "Low": close_prices - np.abs(np.random.randn(100) * 0.3),
        "Close": close_prices,
        "Volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)
    
    # Ensure OHLC constraints
    data["High"] = data[["Open", "High", "Close"]].max(axis=1) + 0.1
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1) - 0.1
    
    return data


@pytest.fixture
def sample_features():
    """Provide sample feature data for model training."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
    
    return X, y


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
