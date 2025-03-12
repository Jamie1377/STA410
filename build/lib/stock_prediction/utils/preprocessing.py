from statsmodels.tsa.stattools import adfuller
import pandas_market_calendars as mcal
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
def find_d(series):
    """Determine differencing order for stationarity"""
    d = 0
    while adfuller(series)[1] > 0.05:
        series = series.diff().dropna()
        d += 1
    return d


# def prepare_features(data, target="Close"):
#     """Create lagged features and technical indicators"""
#     df = data.copy()

#     # Lagged features
#     for lag in [1, 2, 3, 5, 8, 13]:
#         df[f"lag_{lag}"] = df[target].shift(lag)

#     # Rolling features
#     windows = [5, 20, 50]
#     for window in windows:
#         df[f"ma_{window}"] = df[target].rolling(window).mean()
#         df[f"vol_{window}"] = df[target].pct_change().rolling(window).std()

#     # Drop missing values
#     df.dropna(inplace=True)
#     return df


def get_next_valid_date(current_date):
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
