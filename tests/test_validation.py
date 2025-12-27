"""Unit tests for data validation module."""

import pytest
import numpy as np
import pandas as pd
from stock_prediction.utils.validation import DataValidator, validate_data_for_training
from stock_prediction.utils.exceptions import DataValidationError


class TestOHLCVValidation:
    """Test OHLCV data validation."""
    
    @pytest.mark.unit
    def test_valid_ohlcv_data(self, sample_ohlcv_data):
        """Test that valid OHLCV data passes validation."""
        assert DataValidator.validate_ohlcv_data(sample_ohlcv_data, "TEST")
    
    @pytest.mark.unit
    def test_missing_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            # Missing "Low", "Close", "Volume"
        })
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            DataValidator.validate_ohlcv_data(df, "TEST")
    
    @pytest.mark.unit
    def test_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "Open": [100, -101],  # Negative price
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000000, 1000000],
        })
        
        with pytest.raises(DataValidationError, match="Negative or zero prices"):
            DataValidator.validate_ohlcv_data(df, "TEST")
    
    @pytest.mark.unit
    def test_high_less_than_low(self):
        """Test detection of High < Low."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [99, 103],  # High < Low
            "Low": [100, 102],
            "Close": [101, 102],
            "Volume": [1000000, 1000000],
        })
        
        with pytest.raises(DataValidationError, match="High < Low"):
            DataValidator.validate_ohlcv_data(df, "TEST")
    
    @pytest.mark.unit
    def test_negative_volume(self):
        """Test detection of negative volume."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000000, -1000000],  # Negative volume
        })
        
        with pytest.raises(DataValidationError, match="Negative volume"):
            DataValidator.validate_ohlcv_data(df, "TEST")
    
    @pytest.mark.unit
    def test_invalid_dtype(self):
        """Test detection of invalid data types."""
        df = pd.DataFrame({
            "Open": ["100", "101"],  # String instead of numeric
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Volume": [1000000, 1000000],
        })
        
        with pytest.raises(DataValidationError, match="must be numeric"):
            DataValidator.validate_ohlcv_data(df, "TEST")


class TestTimeSeriesValidation:
    """Test time series continuity validation."""
    
    @pytest.mark.unit
    def test_continuous_time_series(self, sample_ohlcv_data):
        """Test that continuous data passes validation."""
        assert DataValidator.validate_time_series_continuity(sample_ohlcv_data, "TEST")
    
    @pytest.mark.unit
    def test_duplicate_dates(self):
        """Test detection of duplicate dates."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D").tolist()
        dates.append(dates[-1])  # Duplicate last date
        
        df = pd.DataFrame({
            "Close": range(11),
        }, index=pd.DatetimeIndex(dates))
        
        # Should warn but not fail
        assert DataValidator.validate_time_series_continuity(df, "TEST")
    
    @pytest.mark.unit
    def test_non_datetime_index(self):
        """Test rejection of non-DatetimeIndex."""
        df = pd.DataFrame({
            "Close": [100, 101, 102],
        })
        
        with pytest.raises(DataValidationError, match="DatetimeIndex"):
            DataValidator.validate_time_series_continuity(df, "TEST")


class TestFeatureValidation:
    """Test feature column validation."""
    
    @pytest.mark.unit
    def test_valid_features(self):
        """Test that valid features pass validation."""
        df = pd.DataFrame({
            "MA_50": np.random.randn(100),
            "RSI": np.random.uniform(0, 100, 100),
            "Volume": np.random.randint(1000000, 5000000, 100),
        })
        
        assert DataValidator.validate_features(df, "TEST")
    
    @pytest.mark.unit
    def test_features_with_nan(self):
        """Test warning on NaN values."""
        df = pd.DataFrame({
            "MA_50": np.random.randn(100),
            "RSI": [np.nan] * 10 + list(np.random.uniform(0, 100, 90)),
        })
        
        # Should warn but pass
        assert DataValidator.validate_features(df, "TEST")
    
    @pytest.mark.unit
    def test_infinite_features(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            "MA_50": [np.inf] + list(np.random.randn(99)),
            "RSI": np.random.uniform(0, 100, 100),
        })
        
        with pytest.raises(DataValidationError, match="Infinite values"):
            DataValidator.validate_features(df, "TEST")


class TestPredictionValidation:
    """Test prediction validation."""
    
    @pytest.mark.unit
    def test_valid_predictions(self):
        """Test that valid predictions pass validation."""
        predictions = np.random.randn(100) * 10 + 100
        assert DataValidator.validate_predictions(predictions)
    
    @pytest.mark.unit
    def test_predictions_with_nan(self):
        """Test detection of NaN in predictions."""
        predictions = np.array([1, 2, np.nan, 4, 5])
        
        with pytest.raises(DataValidationError, match="NaN"):
            DataValidator.validate_predictions(predictions)
    
    @pytest.mark.unit
    def test_predictions_with_inf(self):
        """Test detection of infinite predictions."""
        predictions = np.array([1, 2, np.inf, 4, 5])
        
        with pytest.raises(DataValidationError, match="infinite"):
            DataValidator.validate_predictions(predictions)
    
    @pytest.mark.unit
    def test_extreme_predictions(self):
        """Test warning on extreme values."""
        predictions = np.array([1e7, 2e7, 3e7])  # Very large values
        
        # Should warn but pass
        assert DataValidator.validate_predictions(predictions)


class TestTrainingDataValidation:
    """Test comprehensive training data validation."""
    
    @pytest.mark.unit
    def test_valid_training_data(self, sample_features):
        """Test that valid training data passes."""
        X, y = sample_features
        is_valid, warnings = validate_data_for_training(X, y)
        assert is_valid
        assert len(warnings) == 0
    
    @pytest.mark.unit
    def test_mismatched_lengths(self):
        """Test detection of mismatched X and y lengths."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)
        
        with pytest.raises(DataValidationError, match="length mismatch"):
            validate_data_for_training(X, y)
    
    @pytest.mark.unit
    def test_small_dataset_warning(self):
        """Test warning for small datasets."""
        X = np.random.randn(5, 10)
        y = np.random.randn(5)
        
        is_valid, warnings = validate_data_for_training(X, y)
        assert is_valid
        assert any("small" in w.lower() for w in warnings)
    
    @pytest.mark.unit
    def test_nan_in_features_warning(self):
        """Test warning for NaN in features."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        
        is_valid, warnings = validate_data_for_training(X, y)
        assert is_valid
        assert any("nan" in w.lower() for w in warnings)
    
    @pytest.mark.unit
    def test_inf_in_data(self):
        """Test rejection of infinite values."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        y = np.random.randn(100)
        
        with pytest.raises(DataValidationError, match="infinite"):
            validate_data_for_training(X, y)
