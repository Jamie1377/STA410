"""
Custom exception classes for STA410 Stock Prediction System.

These exceptions provide fine-grained error handling for production monitoring,
allowing callers to distinguish between different failure modes.
"""


class STA410Exception(Exception):
    """Base exception for all STA410 errors."""
    pass


class DataLoadingError(STA410Exception):
    """Raised when data loading from external sources fails."""
    pass


class DataValidationError(STA410Exception):
    """Raised when data fails validation checks."""
    pass


class FeatureEngineeringError(STA410Exception):
    """Raised when feature engineering calculations fail."""
    pass


class ModelTrainingError(STA410Exception):
    """Raised when model training fails."""
    pass


class ModelPredictionError(STA410Exception):
    """Raised when prediction generation fails."""
    pass


class ConfigurationError(STA410Exception):
    """Raised when configuration is invalid or missing."""
    pass


class SecretsManagementError(STA410Exception):
    """Raised when secrets/API keys cannot be loaded."""
    pass


class TradingExecutionError(STA410Exception):
    """Raised when trade execution fails."""
    pass


class CacheError(STA410Exception):
    """Raised when cache operations fail."""
    pass


class APIError(STA410Exception):
    """Raised when external API calls fail."""
    pass
