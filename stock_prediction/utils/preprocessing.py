from statsmodels.tsa.stattools import adfuller
import pandas_market_calendars as mcal
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn import utils
import numpy as np
import os
import random
# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     utils.check_random_state(seed)
#     # torch.manual_seed(seed)
#     # torch.cuda.manual_seed(seed)
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = True


# In utils.py or equivalent:
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Add TensorFlow/PyTorch seeds if used





def find_d(series):
    """Determine differencing order for stationarity"""
    d = 0
    while adfuller(series)[1] > 0.05:
        series = series.diff().dropna()
        d += 1
    return d


def get_next_valid_date(current_date) -> pd.Timestamp:
    """
    Returns the next valid trading day using NYSE calendar.
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


def get_mae(max_leaf_nodes, X1_train, X1_validation, Y1_train, Y1_validation):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X1_train, Y1_train)
    preds_val = model.predict(X1_validation)
    rmse = root_mean_squared_error(Y1_validation, preds_val)
    return rmse
