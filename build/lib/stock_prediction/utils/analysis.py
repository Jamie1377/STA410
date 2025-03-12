import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for features

    Parameters:
        data (pd.DataFrame): Input DataFrame with features

    Returns:
        pd.DataFrame: VIF scores for each feature
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)


def feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable

    Returns:
        pd.DataFrame: Feature importance scores
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    )
    return importance.sort_values(by="Importance", ascending=False)


def adf_test(series: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity

    Parameters:
        series (pd.Series): Time series data

    Returns:
        dict: Test results with keys 'Test Statistic', 'p-value', etc.
    """
    result = adfuller(series)
    return {
        "Test Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Critical Values": result[4],
    }


def interpret_acf_pacf(
    acf_values: np.ndarray, pacf_values: np.ndarray, significance_level: float = 0.05
) -> Tuple[int, int]:
    """
    Suggest ARIMA orders based on ACF/PACF analysis

    Parameters:
        acf_values (np.ndarray): ACF values
        pacf_values (np.ndarray): PACF values
        significance_level (float): Significance level

    Returns:
        Tuple[int, int]: Suggested (p, q) orders
    """
    conf = significance_level * np.sqrt(1 / len(acf_values))

    significant_acf = np.where(np.abs(acf_values) > conf)[0]
    significant_pacf = np.where(np.abs(pacf_values) > conf)[0]

    p = max(significant_pacf) if len(significant_pacf) > 0 else 0
    q = max(significant_acf) if len(significant_acf) > 0 else 0

    return p, q


def best_subset_selection(X: pd.DataFrame, y: pd.Series, max_features: int = 5) -> dict:
    """
    Perform best subset selection for feature selection

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        max_features (int): Maximum number of features to consider

    Returns:
        dict: Best subset results with keys 'features', 'r2', 'aic', 'bic'
    """
    from itertools import combinations
    from statsmodels.api import OLS

    results = []
    features = X.columns

    for k in range(1, min(max_features + 1, len(features) + 1)):
        for combo in combinations(features, k):
            X_subset = X[list(combo)]
            model = OLS(y, X_subset).fit()
            results.append(
                {
                    "features": combo,
                    "r2": model.rsquared,
                    "adj_r2": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )

    return sorted(results, key=lambda x: x["aic"])[0]
