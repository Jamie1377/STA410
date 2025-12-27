"""
Data validation and quality checks for STA410 Stock Prediction System.

Provides validators for stock data, features, and predictions to catch
data quality issues early before they impact model training or trading.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from stock_prediction.utils.logger import setup_logger
from stock_prediction.utils.exceptions import DataValidationError

logger = setup_logger(__name__)


class DataValidator:
    """Validates stock market data and features."""
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame, symbol: str = "") -> bool:
        """
        Validate standard OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Stock symbol for logging
        
        Returns:
            True if valid
        
        Raises:
            DataValidationError: If validation fails
        """
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        provided_columns = set(df.columns)
        
        # Check required columns exist
        missing = required_columns - provided_columns
        if missing:
            raise DataValidationError(
                f"Missing required columns for {symbol}: {missing}"
            )
        
        # Check data types
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if not np.issubdtype(df[col].dtype, np.number):
                raise DataValidationError(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )
        
        # Check for NaN values (some allowed at start/end)
        nan_ratio = df[numeric_cols].isna().sum().sum() / (len(df) * len(numeric_cols))
        if nan_ratio > 0.1:  # More than 10% NaN
            logger.warning(
                f"{symbol}: High NaN ratio ({nan_ratio:.1%}) in OHLCV data. "
                "This may affect model training."
            )
        
        # Check prices are positive
        if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
            raise DataValidationError(f"{symbol}: Negative or zero prices found")
        
        # Check High >= Low
        if (df["High"] < df["Low"]).any():
            raise DataValidationError(f"{symbol}: High < Low (invalid OHLC data)")
        
        # Check Close is in [Low, High] range
        invalid_close = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
        if invalid_close.any():
            invalid_count = invalid_close.sum()
            logger.warning(
                f"{symbol}: {invalid_count} rows with Close outside [Low, High] range. "
                "Data quality may be compromised."
            )
        
        # Check volume is non-negative
        if (df["Volume"] < 0).any():
            raise DataValidationError(f"{symbol}: Negative volume found")
        
        logger.info(f"{symbol}: OHLCV validation passed ({len(df)} rows)")
        return True
    
    @staticmethod
    def validate_time_series_continuity(
        df: pd.DataFrame, symbol: str = "", expected_freq: str = "D"
    ) -> bool:
        """
        Check for gaps in time series data.
        
        Args:
            df: DataFrame with DatetimeIndex
            symbol: Stock symbol for logging
            expected_freq: Expected frequency (D=daily, H=hourly, etc.)
        
        Returns:
            True if continuous (within tolerance)
        
        Raises:
            DataValidationError: If significant gaps found
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Index must be DatetimeIndex")
        
        if len(df) < 2:
            logger.warning(f"{symbol}: Very small dataset ({len(df)} rows)")
            return True
        
        # Check for duplicate dates
        if df.index.duplicated().any():
            logger.warning(f"{symbol}: Duplicate dates found in index")
        
        # Check for gaps (business days)
        if expected_freq == "D":
            gaps = df.index.to_series().diff()
            expected_gap = pd.Timedelta(days=1)
            # Allow for weekends/holidays
            max_gap = pd.Timedelta(days=3)
            
            excessive_gaps = (gaps > max_gap).sum()
            if excessive_gaps > 0:
                logger.warning(
                    f"{symbol}: Found {excessive_gaps} gaps > 3 days. "
                    "This may indicate data quality issues or corporate actions."
                )
        
        logger.debug(f"{symbol}: Time series continuity check passed")
        return True
    
    @staticmethod
    def validate_features(df: pd.DataFrame, symbol: str = "") -> bool:
        """
        Validate feature columns for model training.
        
        Args:
            df: DataFrame with features
            symbol: Stock symbol for logging
        
        Returns:
            True if valid
        
        Raises:
            DataValidationError: If features are invalid
        """
        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            nan_counts = df[nan_cols].isna().sum()
            nan_ratio = (nan_counts / len(df)).mean()
            logger.warning(
                f"{symbol}: Features with NaN values: {nan_cols}. "
                f"Average NaN ratio: {nan_ratio:.1%}"
            )
        
        # Check for infinite values
        inf_cols = df.columns[np.isinf(df).any()].tolist()
        if inf_cols:
            raise DataValidationError(f"{symbol}: Infinite values in columns: {inf_cols}")
        
        # Check for constant features (no variance)
        constant_cols = df.columns[df.var() == 0].tolist()
        if constant_cols:
            logger.warning(f"{symbol}: Constant features (zero variance): {constant_cols}")
        
        # Check feature ranges (basic sanity check)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if (df[col] < -1e6).any() or (df[col] > 1e6).any():
                    logger.warning(
                        f"{symbol}: Feature '{col}' has extreme values. "
                        "Consider normalization."
                    )
        
        logger.debug(f"{symbol}: Feature validation passed ({len(df.columns)} features)")
        return True
    
    @staticmethod
    def validate_predictions(
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        threshold: float = 0.1
    ) -> bool:
        """
        Validate model predictions for reasonable values.
        
        Args:
            predictions: Array of predictions
            actuals: Optional array of actual values for relative checks
            threshold: Max deviation threshold as fraction
        
        Returns:
            True if predictions are reasonable
        
        Raises:
            DataValidationError: If predictions are invalid
        """
        # Check for NaN
        if np.isnan(predictions).any():
            raise DataValidationError(
                f"Predictions contain {np.isnan(predictions).sum()} NaN values"
            )
        
        # Check for infinite
        if np.isinf(predictions).any():
            raise DataValidationError(
                f"Predictions contain {np.isinf(predictions).sum()} infinite values"
            )
        
        # Check magnitude
        if np.abs(predictions).max() > 1e6:
            logger.warning(
                f"Predictions have extreme values (max: {np.abs(predictions).max():.2e}). "
                "This may indicate model instability."
            )
        
        # Relative checks if actuals provided
        if actuals is not None:
            actuals = np.asarray(actuals)
            
            # Check predictions are not constant when actuals vary
            if actuals.std() > 0 and predictions.std() == 0:
                logger.warning(
                    "Predictions are constant while actuals vary. "
                    "Model may not have learned features."
                )
            
            # Check for extreme deviation
            pct_errors = np.abs((predictions - actuals) / (np.abs(actuals) + 1e-8))
            extreme_errors = (pct_errors > threshold).sum()
            if extreme_errors / len(predictions) > 0.1:  # > 10% extreme errors
                logger.warning(
                    f"{extreme_errors} predictions ({extreme_errors/len(predictions):.1%}) "
                    f"deviate by > {threshold:.1%} from actuals"
                )
        
        logger.debug("Prediction validation passed")
        return True


def validate_data_for_training(
    X: np.ndarray, y: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of training data.
    
    Args:
        X: Feature array
        y: Target array
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check shapes match
    if len(X) != len(y):
        raise DataValidationError(
            f"X and y length mismatch: {len(X)} != {len(y)}"
        )
    
    if len(X) < 10:
        warnings.append(f"Very small training set: {len(X)} samples")
    
    # Check X has features
    if len(X.shape) < 2:
        raise DataValidationError("X must be 2D array")
    
    # Check for NaN/inf in X
    if np.isnan(X).any():
        warnings.append(f"X contains {np.isnan(X).sum()} NaN values")
    
    if np.isinf(X).any():
        raise DataValidationError(f"X contains infinite values")
    
    # Check for NaN/inf in y
    if np.isnan(y).any():
        warnings.append(f"y contains {np.isnan(y).sum()} NaN values")
    
    if np.isinf(y).any():
        raise DataValidationError("y contains infinite values")
    
    # Check feature scaling
    X_std = np.std(X, axis=0)
    if (X_std < 1e-10).any():
        warnings.append("Some features have very low variance")
    
    if (X_std > 1e6).any():
        warnings.append("Some features have very high variance - consider scaling")
    
    logger.info(f"Training data validation passed. {len(warnings)} warnings.")
    for w in warnings:
        logger.warning(f"  - {w}")
    
    return True, warnings
