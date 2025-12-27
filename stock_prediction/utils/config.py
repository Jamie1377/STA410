"""
Configuration management for STA410 Stock Prediction System.

Supports environment-based configuration (development, staging, production)
and loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    max_file_size: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_SIZE", 10485760)))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", 10)))


@dataclass
class DataConfig:
    """Data loading and caching configuration."""
    cache_dir: str = field(default_factory=lambda: os.getenv("DATA_CACHE_DIR", "data_cache"))
    model_cache_dir: str = field(default_factory=lambda: os.getenv("MODEL_CACHE_DIR", "model_cache"))
    cache_ttl_hours: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_HOURS", 24)))
    validate_on_load: bool = field(default_factory=lambda: os.getenv("VALIDATE_DATA", "true").lower() == "true")


@dataclass
class ModelConfig:
    """Model training configuration."""
    train_test_split: float = field(default_factory=lambda: float(os.getenv("TRAIN_TEST_SPLIT", 0.8)))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", 42)))
    
    # GradientDescentRegressor defaults
    gd_learning_rate: float = field(default_factory=lambda: float(os.getenv("GD_LR", 0.05)))
    gd_iterations: int = field(default_factory=lambda: int(os.getenv("GD_ITERATIONS", 1000)))
    gd_momentum: float = field(default_factory=lambda: float(os.getenv("GD_MOMENTUM", 0.9)))
    gd_l2_reg: float = field(default_factory=lambda: float(os.getenv("GD_L2_REG", 0.01)))
    gd_l1_ratio: float = field(default_factory=lambda: float(os.getenv("GD_L1_RATIO", 0.01)))
    
    # Model optimization
    enable_optuna: bool = field(default_factory=lambda: os.getenv("ENABLE_OPTUNA", "true").lower() == "true")
    optuna_trials: int = field(default_factory=lambda: int(os.getenv("OPTUNA_TRIALS", 100)))


@dataclass
class TradingConfig:
    """Trading and risk management configuration."""
    max_portfolio_risk: float = field(default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_RISK", 0.05)))
    per_trade_risk: float = field(default_factory=lambda: float(os.getenv("PER_TRADE_RISK", 0.025)))
    stop_loss_pct: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", 0.03)))
    take_profit_pct: float = field(default_factory=lambda: float(os.getenv("TAKE_PROFIT_PCT", 0.003)))
    max_sector_exposure: float = field(default_factory=lambda: float(os.getenv("MAX_SECTOR_EXPOSURE", 0.4)))
    daily_loss_limit: float = field(default_factory=lambda: float(os.getenv("DAILY_LOSS_LIMIT", -0.03)))
    
    # Paper trading (default: True for safety)
    paper_trading: bool = field(default_factory=lambda: os.getenv("PAPER_TRADING", "true").lower() == "true")
    simulated_slippage: float = field(default_factory=lambda: float(os.getenv("SLIPPAGE_BPS", 5)) / 10000)  # 5 bps


@dataclass
class APIConfig:
    """External API configuration."""
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    
    yfinance_timeout: int = field(default_factory=lambda: int(os.getenv("YFINANCE_TIMEOUT", 30)))
    request_retries: int = field(default_factory=lambda: int(os.getenv("REQUEST_RETRIES", 3)))
    request_retry_delay: int = field(default_factory=lambda: int(os.getenv("REQUEST_RETRY_DELAY", 2)))


@dataclass
class Config:
    """Master configuration combining all subsystems."""
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.model.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        if self.model.gd_learning_rate <= 0:
            raise ValueError("gd_learning_rate must be positive")
        
        if self.model.gd_iterations <= 0:
            raise ValueError("gd_iterations must be positive")
        
        if not 0 < self.trading.max_portfolio_risk <= 1:
            raise ValueError("max_portfolio_risk must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "logging": self.logging.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "trading": self.trading.__dict__,
            "api": self.api.__dict__,
        }


# Load environment variables from .env file if it exists
def load_env_file(env_file: str = ".env") -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to .env file (default: .env in project root)
    """
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


# Global configuration instance
_config = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config: Singleton configuration instance
    """
    global _config
    if _config is None:
        load_env_file()  # Load .env file if available
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance (mainly for testing).
    
    Args:
        config: Configuration instance to use
    """
    global _config
    _config = config
