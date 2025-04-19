# from stock_prediction.utils import seed_everything
# seed_everything(42)
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()


def optimize_lookback(X, y, model, min_window=50, max_window=None, step_size=20, n_splits=5):
    """
    Dynamically finds the optimal lookback window using walk-forward validation
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model: Sklearn-style model with .fit() and .predict()
        min_window (int): Minimum training window size
        step_size (int): Increment to test larger windows
        n_splits (int): CV splits
        
    Returns:
        (int): Optimal window size in samples
        (pd.DataFrame): Validation results
    """
    if max_window is None:
        max_window = round(0.75*len(X))    # Don't use more than 66% of data for training
    else:
        max_window = max_window
    results = {}
    
    for window in range(min_window, max_window, step_size):
        # tscv = TimeSeriesSplit(n_splits=n_splits, test_size=window//4)
        scores = []
        r2 = []
        
        # for train_idx, test_idx in tscv.split(X):
        #     X_train, X_test = X.iloc[train_idx[-window:]], X.iloc[test_idx]
        #     y_train, y_test = y.iloc[train_idx[-window:]], y.iloc[test_idx]
        #     X_train = Scaler.fit_transform(X_train)
        #     X_test = Scaler.transform(X_test)
        #     model.fit(X_train, y_train)
        #     preds = model.predict(X_test)
        #     scores.append(root_mean_squared_error(y_test, preds))
        #     # hand compute r2 score 
        #     r2.append(  
        #         1 - (np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        #     )  # R^2 = 1 - (SS_res / SS_tot)

        
        X_train, X_test, y_train, y_test = train_test_split(
                X.iloc[-window:,], y.iloc[-window:,], test_size=0.2, random_state=42
            )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        scores.append(root_mean_squared_error(y_test, preds))
        # hand compute r2 score
        r2.append(  
            1 - (np.sum((y_test - preds) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        )   



         
        print('Scores', scores)  
        results[window] = { 
            'rmse': np.mean(scores),
            'std': np.std(scores),
            'r2': np.mean(r2),

        }
        print(results)
    results_df = pd.DataFrame(results).T
    print(results_df)
    # optimal_window = results_df.loc[results_df['rmse'].idxmin(), 'window']
    optimal_window =results_df['r2'].idxmax()
    
    return optimal_window




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

def vizualize_correlation(data: pd.DataFrame):
    plt.figure(figsize=(30, 30))
    correlation_mat = data.corr()
    sns.heatmap(correlation_mat, annot = True, cmap='coolwarm')
    plt.show()


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
