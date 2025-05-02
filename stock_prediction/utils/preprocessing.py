from statsmodels.tsa.stattools import adfuller
import pandas_market_calendars as mcal
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import os
import random


def seed_everything(seed=42):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): Seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Add TensorFlow/PyTorch seeds if used


def find_d(series) -> int:
    """Determine differencing order for stationarity
    Args:
        series (pd.Series): Time series data
    Returns:
        int: Differencing order
    """
    # Check if the series is already stationary
    d = 0
    while adfuller(series)[1] > 0.05:
        series = series.diff().dropna()
        d += 1
    return d


def get_next_valid_date(current_date: str) -> pd.Timestamp:
    """
    Returns the next valid trading day using NYSE calendar.
    Args:
        current_date (str or pd.Timestamp): Current date
    Returns:
        pd.Timestamp: Next valid trading day
    """
    # Get NYSE calendar
    nyse = mcal.get_calendar("NYSE")

    # Convert input to pandas Timestamp if it isn't already
    current_date = pd.Timestamp(current_date)

    # Get valid trading days for a range (using 10 days to be safe)
    schedule = nyse.schedule(
        start_date=current_date, end_date=current_date + pd.Timedelta(days=10)
    )

    # Get the next valid trading day
    valid_days = schedule.index
    next_day = valid_days[valid_days > current_date][0]

    if next_day == pd.Timestamp("2025-01-09 00:00:00"):
        next_day += pd.Timedelta(days=1)
    return next_day


def get_mae(max_leaf_nodes, X_train, X_validation, y_train, y_validation):
    """
    Calculate Mean Absolute Error (MAE) for Decision Tree Regressor
    Args:
        max_leaf_nodes (int): Maximum number of leaf nodes
        X1_train (pd.DataFrame): Training features
        X1_validation (pd.DataFrame): Validation features
        Y1_train (pd.Series): Training target
        Y1_validation (pd.Series): Validation target
    Returns:
        float: Root Mean Squared Error (RMSE)
    """
    # Ensure the data is in the correct format
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_validation)
    rmse = root_mean_squared_error(y_validation, preds_val)
    return rmse