"""Unit tests for configuration management."""

import pytest
import os
from stock_prediction.utils.config import Config, LoggingConfig, DataConfig, ModelConfig, TradingConfig
from stock_prediction.utils.exceptions import ConfigurationError


class TestLoggingConfig:
    """Test logging configuration."""
    
    @pytest.mark.unit
    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.log_dir == "logs"
        assert config.max_file_size == 10485760


class TestDataConfig:
    """Test data configuration."""
    
    @pytest.mark.unit
    def test_default_data_config(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.cache_dir == "data_cache"
        assert config.model_cache_dir == "model_cache"
        assert config.cache_ttl_hours == 24
        assert config.validate_on_load is True


class TestModelConfig:
    """Test model configuration."""
    
    @pytest.mark.unit
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert 0 < config.train_test_split < 1
        assert config.gd_learning_rate > 0
        assert config.gd_iterations > 0


class TestTradingConfig:
    """Test trading configuration."""
    
    @pytest.mark.unit
    def test_default_trading_config(self):
        """Test default trading configuration."""
        config = TradingConfig()
        assert 0 < config.max_portfolio_risk <= 1
        assert config.paper_trading is True  # Default to safe mode
    
    @pytest.mark.unit
    def test_risk_parameters_consistency(self):
        """Test that risk parameters are sensible."""
        config = TradingConfig()
        # Per-trade risk should be less than portfolio risk
        assert config.per_trade_risk < config.max_portfolio_risk


class TestMasterConfig:
    """Test master configuration."""
    
    @pytest.mark.unit
    def test_default_config(self, test_config):
        """Test default configuration."""
        assert test_config.environment == "testing"
        assert test_config.logging is not None
        assert test_config.data is not None
        assert test_config.model is not None
        assert test_config.trading is not None
    
    @pytest.mark.unit
    def test_config_to_dict(self, test_config):
        """Test configuration to dictionary conversion."""
        config_dict = test_config.to_dict()
        assert "environment" in config_dict
        assert "logging" in config_dict
        assert "data" in config_dict
        assert "model" in config_dict
        assert "trading" in config_dict
    
    @pytest.mark.unit
    def test_config_validation_invalid_split(self):
        """Test validation of train_test_split."""
        with pytest.raises(ValueError, match="train_test_split"):
            Config(model=ModelConfig(train_test_split=1.5))
    
    @pytest.mark.unit
    def test_config_validation_invalid_lr(self):
        """Test validation of learning rate."""
        with pytest.raises(ValueError, match="gd_learning_rate"):
            Config(model=ModelConfig(gd_learning_rate=-0.01))
    
    @pytest.mark.unit
    def test_config_validation_invalid_iterations(self):
        """Test validation of iterations."""
        with pytest.raises(ValueError, match="gd_iterations"):
            Config(model=ModelConfig(gd_iterations=-100))
    
    @pytest.mark.unit
    def test_config_validation_invalid_risk(self):
        """Test validation of risk parameters."""
        with pytest.raises(ValueError, match="max_portfolio_risk"):
            Config(trading=TradingConfig(max_portfolio_risk=1.5))
    
    @pytest.mark.unit
    def test_environment_variants(self):
        """Test configuration with different environments."""
        for env in ["development", "staging", "production"]:
            config = Config(environment=env)
            assert config.environment == env
