import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from sklearn.model_selection import train_test_split
from .models import ARIMAXGBoost
from stock_prediction.utils import get_mae, get_next_valid_date

# Sample Dataset
stock_data = yf.download("AAPL", start="2024-02-20", end=date.today())
stock_data.columns = stock_data.columns.droplevel(1)
stock_data


class StockPredictor:
    """Stock price prediction pipeline

    Parameters:
        symbol (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        interval (str): Data interval (1d, 1h, etc)
    """

    def __init__(self, symbol, start_date, end_date=None, interval="1d"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else date.today()
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_params = {}
        self.data = None
        self.feature_sets = {
            "Close": {"target": "Close", "features": None},
            "Low": {"target": "Low", "features": None},
            # "Adj Close": {"target": "Adj Close", "features": None},
            "Daily Returns": {"target": "Daily Returns", "features": None},
            "Volatility": {"target": "Volatility", "features": None},
            "TNX": {"target": "TNX", "features": None},
            "Treasury_Yield": {"target": "Treasury_Yield", "features": None},
            "SP500": {"target": "SP500", "features": None},
            "USDCAD=X": {"target": "USDCAD=X", "features": None},
        }
        self.scalers = {}
        self.transformers = {}
        self.interval = interval

        # self.one_step_forward_forecast = {}

    def load_data(self):
        """Load and prepare stock data with features"""
        self.data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
        )
        self.data.columns = self.data.columns.get_level_values(0)
        self.data.ffill()
        self.data.dropna()
        # Add technical indicators
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()

        # self.data['RSI'] = ta.momentum.RSIIndicator(self.data['Close']).rsi()
        # Add Bollinger Bands
        # bb = ta.volatility.BollingerBands(self.data['Close'])
        # self.data['BB_High'] = bb.bollinger_hband()
        # self.data['BB_Low'] = bb.bollinger_lband()

        # Add rolling statistics
        self.data["rolling_std"] = self.data["Close"].rolling(window=50).std()
        self.data["rolling_min"] = self.data["Close"].rolling(window=50).min()
        # self.data['rolling_max'] = self.data['Close'].rolling(window=window).max()
        self.data["rolling_median"] = self.data["Close"].rolling(window=50).median()
        self.data["rolling_sum"] = self.data["Close"].rolling(window=50).sum()
        self.data["rolling_var"] = self.data["Close"].rolling(window=50).var()
        self.data["rolling_ema"] = (
            self.data["Close"].ewm(span=50, adjust=False).mean()
        )  # Exponential Moving Average

        # Add rolling quantiles (25th and 75th percentiles)
        self.data["rolling_25p"] = self.data["Close"].rolling(window=50).quantile(0.25)
        self.data["rolling_75p"] = self.data["Close"].rolling(window=50).quantile(0.75)

        # Drop rows with NaN values (due to rolling window)
        self.data.dropna(inplace=True)

        # Add MACD
        # macd = ta.trend.MACD(self.data['Close'])
        # self.data['MACD'] = macd.macd()
        # self.data['MACD_Signal'] = macd.macd_signal()

        # Add ATR
        # self.data['ATR'] = ta.volatility.AverageTrueRange(
        # high=self.data['High'], low=self.data['Low'], close=self.data['Close']
        # ).average_true_range()

        stock_data.index.name = "Date"  # Ensure the index is named "Date"

        # Fetch S&P 500 Index (GSPC) and Treasury Yield ETF (IEF) from Yahoo Finance
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)["Close"]
        tnx = yf.download(
            "^TNX", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        treasury_yield = yf.download(
            "IEF", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        exchange_rate = yf.download(
            "USDCAD=X", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        technology_sector = yf.download(
            "XLK", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        financials_sector = yf.download(
            "XLF", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        vix = yf.download(
            "^VIX", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]

        economic_data = (
            pd.concat(
                [
                    sp500,
                    tnx,
                    treasury_yield,
                    exchange_rate,
                    technology_sector,
                    financials_sector,
                    vix,
                ],
                axis=1,
                keys=[
                    "SP500",
                    "TNX",
                    "Treasury_Yield",
                    "USDCAD=X",
                    "Tech",
                    "Fin",
                    "VIX",
                ],
            )
            .reset_index()
            .rename(columns={"index": "Date"})
            .dropna()
        )
        economic_data.columns = economic_data.columns.get_level_values(0)

        economic_data["Date"] = pd.to_datetime(economic_data["Date"])
        economic_data.set_index("Date", inplace=True)

        nyse = mcal.get_calendar("NYSE")

        # Get the NYSE trading schedule
        schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
        economic_data["is_next_non_trading_day"] = economic_data.index.shift(
            -1, freq="1d"
        ).isin(schedule.index).astype(int) + economic_data.index.shift(
            1, freq="1d"
        ).isin(
            schedule.index
        ).astype(
            int
        )

        self.data = pd.merge(self.data, economic_data, on="Date", how="left")
        self.data["Daily Returns"] = self.data["Close"].pct_change()
        self.data["Volatility"] = self.data["Daily Returns"].rolling(window=20).std()

        # self.data['Total Assets'] = balance_sheet['Total Assets']
        # self.data['Debt/Equity'] = balance_sheet["Total Liabilities Net Minority Interest"] / balance_sheet["Total Equity Gross Minority Interest"]
        self.data = self.data.dropna()

        # Process each feature set
        for name, config in self.feature_sets.items():
            target = config["target"]
            X = self.data.drop(columns=[target]).values
            y = self.data[target].values

            X_df = self.data.drop(columns=[target])
            y_df = self.data[target]

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )

            self.X_train_df, self.X_test_df, self.y_train_df, self.y_test_df = (
                train_test_split(
                    X_df, y_df, test_size=0.2, shuffle=False, random_state=42
                )
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Polynomial features
            transformer = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = transformer.fit_transform(X_train_scaled)
            X_test_poly = transformer.transform(X_test_scaled)

            # Store transformed data
            self.feature_sets[name].update(
                {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "X_train_df": self.X_train_df,
                    "X_test_df": self.X_test_df,
                    "y_train_df": self.y_train_df,
                    "y_test_df": self.y_test_df,
                    "X_train_scaled": X_train_scaled,
                    "X_test_scaled": X_test_scaled,
                    "X_train_poly": X_train_poly,
                    "X_test_poly": X_test_poly,
                }
            )

            self.scalers[name] = scaler
            self.transformers[name] = transformer

        return self

    def train_sarima(self):
        """Train SARIMA model"""

        for feature_name, feature_set in self.feature_sets.items():
            self.models[feature_name] = {}
            self.forecasts[feature_name] = {}
            self.metrics[feature_name] = {}

            dtrain = feature_set["X_train_df"].diff()
            ddtrain = dtrain.diff()
            dddtrain = ddtrain.diff()
            logtrain = np.log(feature_set["X_train"])
            log2train = np.log(logtrain)
            log3train = np.log(log2train)
            log2train = np.log(
                np.log(feature_set["y_train"])
            )  # Double log transformation
            model = SARIMAX(log2train, order=(2, 1, 3), seasonal_order=(0, 0, 0, 0))
            self.models[feature_name]["sarima"] = model.fit()

            # Generate forecast
            forecast_log = self.models[feature_name]["sarima"].forecast(
                steps=len(feature_set["y_test"])
            )
            self.forecasts[feature_name]["sarima"] = np.exp(
                np.exp(forecast_log)
            )  # Transform back
            self.metrics[feature_name]["sarima"] = root_mean_squared_error(
                feature_set["y_test"][: len(forecast_log)],
                self.forecasts[feature_name]["sarima"],
            )
        return self

    def viz_sarima(self):
        "Vizualize Sarima model"
        dtrain = self.X_train.diff()
        ddtrain = dtrain.diff()
        dddtrain = ddtrain.diff()
        plt.plot(dtrain.index, dtrain)
        logtrain = np.log(self.X_train)
        log2train = np.log(logtrain)
        log3train = np.log(log2train)
        plt.plot(logtrain.index, logtrain)
        plt.plot(log2train.index, log2train)
        plt.plot(log3train.index, log3train)
        plt.plot(self.X_train.index, self.X_train)

    def train_ml_models(self):
        """Train multiple machine learning models"""
        base_models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeRegressor(random_state=42),
            "linear": LinearRegression(),
            "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            "knn": KNeighborsRegressor(n_neighbors=20, weights="distance"),
        }
        # n = len(self.X_train)
        # p = self.X_train.shape[1]  # Number of predictors

        # Train and evaluate each model

        for feature_name, feature_set in self.feature_sets.items():

            for model_name, model in base_models.items():
                fitted_model = model.fit(feature_set["X_train"], feature_set["y_train"])
                self.models[feature_name][model_name] = fitted_model

                # Calculate metrics
                y_pred = fitted_model.predict(feature_set["X_test"])
                self.forecasts[feature_name][model_name] = y_pred
                rmse = root_mean_squared_error(feature_set["y_test"], y_pred)
                self.metrics[feature_name][model_name] = rmse

        return self

    def arima_ml(self):

        for feature_name, feature_set in self.feature_sets.items():
            self.forecasts[feature_name]["arima_ml"] = {}
            target = feature_set["target"]
            y = self.data[target]
            # train, test = train_test_split( y, test_size=0.2, shuffle=False, random_state=42)
            # Assuming `y` is your time series data (e.g., stock prices)
            train_size = int(len(y) * 0.8)  # 80% for training
            train, test = y[:train_size], y[train_size:]

            # stock_prices = self.data["Close"]   #.iloc[:-horizon,]
            arima_model = ARIMA(
                train, order=(1, 1, 1)
            )  # Replace p, d, q with appropriate values
            arima_fit = arima_model.fit()
            train_size = len(train)
            test_size = len(test)
            # arima_predictions = arima_fit.predict(start=train_size, end=train_size + test_size - 1)
            arima_forecast = arima_fit.forecast(steps=test_size)
            arima_predictions = arima_fit.predict()

            # Step 2: Calculate Residuals
            residuals = train - arima_predictions

            # Step 3: Prepare Data for Machine Learning
            lagged_features = pd.concat(
                [residuals.shift(i) for i in range(1, 6)], axis=1
            )  # Lagged residuals
            lagged_features.columns = [f"lag_{i}" for i in range(1, 6)]
            lagged_features["residual"] = residuals
            lagged_features = lagged_features.dropna()

            X = lagged_features.drop("residual", axis=1)
            y = lagged_features["residual"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42
            )

            # Step 4: Train XGBoost on Residuals
            xgb_model = XGBRegressor()
            xgb_model.fit(X_train, y_train)

            # Step 5: Predict Residuals and Combine Predictions
            xgb_predictions = xgb_model.predict(X_test)
            final_predictions = arima_forecast[-len(X_test) :] + xgb_predictions

            # Step 6: Evaluate Performance
            test_rmse = root_mean_squared_error(y[-len(X_test) :], final_predictions)
            print(f"Test Root Mean Squared Error: {test_rmse}")
            self.forecasts[feature_name]["arima_ml"] = final_predictions

            plt.figure(figsize=(10, 6))
            plt.plot(
                self.data[target][-len(X_test) :].index,
                self.data[target][-len(X_test) :],
                label="Actual Prices",
            )
            plt.plot(
                self.data[target][-len(X_test) :].index,
                final_predictions,
                label="Predicted Prices",
                linestyle="--",
            )
            plt.legend()
            plt.title(f"{feature_name} Predictions by ARIMA + ML")
            plt.xlabel("Time")
            plt.ylabel("Price")

            # plt.xlim(left = pd.Timestamp('2024-01-01'), right = date.today()  + timedelta(days=15, hours=-5))
            plt.show()

        # return final_predictions

    # def fine_tune_tree_models(self):

    #     for feature_name, feature_set in self.feature_sets.items():

    #         self.models[feature_name]["decision_tree_max_leaf_node"] = {}
    #         self.forecasts[feature_name]["decision_tree_max_leaf_node"] = {}
    #         self.metrics[feature_name]["decision_tree_max_leaf_node"] = {}

    #         dt_rmse = root_mean_squared_error(
    #             feature_set["y_test"], self.forecasts[feature_name]["decision_tree"]
    #         )
    #         print(
    #             "Test RMSE for Decision Tree Model when not specifying max_leaf_nodes: {}".format(
    #                 dt_rmse
    #             )
    #         )
    #         candidate_max_leaf_nodes = [5, 25, 50, 230, 250, 300, 500]
    #         scores = {
    #             leaf_size: get_mae(
    #                 leaf_size,
    #                 feature_set["X_train"],
    #                 feature_set["X_test"],
    #                 feature_set["y_train"],
    #                 feature_set["y_test"],
    #             )
    #             for leaf_size in candidate_max_leaf_nodes
    #         }
    #         best_tree_size = min(scores, key=scores.get)
    #         n = len(feature_set["X_train"])
    #         p = feature_set["X_train"].shape[1]  # Number of predictors
    #         self.models[feature_name]["decision_tree_max_leaf_node"] = (
    #             DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
    #         )
    #         self.models[feature_name]["decision_tree_max_leaf_node"].fit(
    #             feature_set["X_train"],
    #             feature_set["y_train"],
    #         )
    #         self.forecasts[feature_name]["decision_tree_max_leaf_node"] = self.models[
    #             feature_name
    #         ]["decision_tree_max_leaf_node"].predict(feature_set["X_test"])
    #         # self.one_step_forward_forecast["decision_tree_max_leaf_node"] = self.models["decision_tree_max_leaf_node"].predict(X_test[-1,].reshape(1, -1))
    #         self.metrics[feature_name]["decision_tree_max_leaf_node"] = {
    #             f"rmse (decision_tree_max_leaf_node)": root_mean_squared_error(
    #                 feature_set["y_test"],
    #                 self.forecasts[feature_name]["decision_tree_max_leaf_node"],
    #             ),
    #             f"adjusted_r2 (decision_tree_max_leaf_node)": 1
    #             - (
    #                 1
    #                 - self.models[feature_name]["decision_tree_max_leaf_node"].score(
    #                     feature_set["X_train"], feature_set["y_train"]
    #                 )
    #             )
    #             * (n - 1)
    #             / (n - p - 1),
    #         }
    #         print(
    #             "Test RMSE for Decision Tree Model when specifying max_leaf_nodes: {}".format(
    #                 root_mean_squared_error(
    #                     feature_set["y_test"],
    #                     self.forecasts[feature_name]["decision_tree_max_leaf_node"],
    #                 )
    #             )
    #         )
    #         return self

    # def train_regularized_models(self):
    #     """Train Ridge and Lasso models with cross-validation"""
    #     # Test range of alpha values
    #     alphas = np.logspace(-4, 4, 50)

    #     for feature_name, feature_set in self.feature_sets.items():
    #         for model in ["ridge", "lasso"]:
    #             self.models[feature_name][model] = {}
    #             self.forecasts[feature_name][model] = {}
    #             self.metrics[feature_name][model] = {}

    #         # Ridge Regression
    #         ridge_cv_mse = []
    #         for alpha in alphas:
    #             ridge = Ridge(alpha=alpha, fit_intercept=True)
    #             mse = -cross_val_score(
    #                 ridge,
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 scoring="neg_mean_squared_error",
    #                 cv=5,
    #             ).mean()
    #             ridge_cv_mse.append(mse)

    #         # Find best Ridge alpha
    #         best_ridge_alpha = alphas[np.argmin(ridge_cv_mse)]
    #         self.best_params["ridge_alpha"] = best_ridge_alpha

    #         # Train Ridge with best alpha
    #         self.models[feature_name]["ridge"] = Ridge(
    #             alpha=best_ridge_alpha, fit_intercept=True
    #         )
    #         self.models[feature_name]["ridge"].fit(
    #             feature_set["X_train_scaled"], feature_set["y_train"]
    #         )
    #         self.forecasts[feature_name]["ridge"] = self.models[feature_name][
    #             "ridge"
    #         ].predict(feature_set["X_test_scaled"])
    #         # self.one_step_forward_forecast['ridge'] = self.models['ridge'].predict(X_test_scaled[-1,].reshape(1, -1))
    #         # Lasso Regression
    #         lasso_cv_mse = []
    #         for alpha in alphas:
    #             lasso = Lasso(alpha=alpha, fit_intercept=True)
    #             mse = -cross_val_score(
    #                 lasso,
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 scoring="neg_mean_squared_error",
    #                 cv=5,
    #             ).mean()
    #             lasso_cv_mse.append(mse)

    #         # Find best Lasso alpha
    #         best_lasso_alpha = alphas[np.argmin(lasso_cv_mse)]
    #         self.best_params["lasso_alpha"] = best_lasso_alpha

    #         # Train Lasso with best alpha
    #         self.models[feature_name]["lasso"] = Lasso(
    #             alpha=best_lasso_alpha, fit_intercept=True
    #         )
    #         self.models[feature_name]["lasso"].fit(
    #             feature_set["X_train_scaled"], feature_set["y_train"]
    #         )
    #         self.forecasts[feature_name]["lasso"] = self.models[feature_name][
    #             "lasso"
    #         ].predict(feature_set["X_test_scaled"])
    #         # self.one_step_forward_forecast['lasso'] = self.models['lasso'].predict(X_test_scaled[-1,].reshape(1, -1))

    #         # Calculate metrics
    #         self.metrics[feature_name]["ridge"] = {
    #             "rmse": root_mean_squared_error(
    #                 feature_set["y_test"], self.forecasts[feature_name]["ridge"]
    #             ),
    #             "cv_scores": cross_val_score(
    #                 self.models[feature_name]["ridge"],
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 cv=5,
    #                 scoring="r2",
    #             ),
    #             "cv_rmses": -cross_val_score(
    #                 self.models[feature_name]["ridge"],
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 cv=5,
    #                 scoring="neg_root_mean_squared_error",
    #             ),
    #         }

    #         self.metrics[feature_name]["lasso"] = {
    #             "rmse": root_mean_squared_error(
    #                 feature_set["y_test"], self.forecasts[feature_name]["lasso"]
    #             ),
    #             "cv_scores": cross_val_score(
    #                 self.models[feature_name]["lasso"],
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 cv=5,
    #                 scoring="r2",
    #             ),
    #             "cv_rmses": -cross_val_score(
    #                 self.models[feature_name]["lasso"],
    #                 feature_set["X_train_scaled"],
    #                 feature_set["y_train"],
    #                 cv=5,
    #                 scoring="neg_root_mean_squared_error",
    #             ),
    #         }

    #     return self

    # def print_regularization_metrics(self):
    #     """Print detailed metrics for regularized models"""
    #     print("\nRegularization Models Performance:")
    #     print("-" * 50)
    #     for model in ["ridge", "lasso"]:
    #         if model in self.metrics:
    #             print(f"\n{model.upper()} Regression:")
    #             print(f"Best alpha: {self.best_params[f'{model}_alpha']:.6f}")
    #             print(f"RMSE: {self.metrics[model]['rmse']:.2f}")
    #             print(
    #                 f"CV R² scores: {self.metrics[model]['cv_scores'].mean():.3f} "
    #                 f"(±{self.metrics[model]['cv_scores'].std()*2:.3f})"
    #             )
    #             if model == "ridge":
    #                 print(f"Feature coefficients:")
    #                 for feat, coef in zip(self.X.columns, self.models[model].coef_):
    #                     print(f"  {feat}: {coef:.4f}")
    #     print("-" * 50)

    # def train_polynomial(self, degree=2):
    #     """Train polynomial regression model"""

    #     for feature_name, feature_set in self.feature_sets.items():
    #         self.transformers[feature_name] = {}
    #         self.forecasts[feature_name]["polynomial"] = {}

    #         # transformer = PolynomialFeatures(degree=degree, include_bias=False)
    #         # X_train_poly = transformer.fit_transform(feature_set["X_train_scaled"])
    #         # X_test_poly = transformer.transform(feature_set["X_test_scaled"])

    #         X_train_poly = feature_set["X_train_poly"]
    #         X_test_poly = feature_set["X_test_poly"]
    #         y_train = feature_set["y_train"]
    #         y_test = feature_set["y_test"]

    #         model = LinearRegression()
    #         model.fit(X_train_poly, y_train)

    #         n = len(X_train_poly)
    #         p = X_train_poly.shape[1]  # Number of predictors

    #         self.models[feature_name]["polynomial"] = model
    #         self.forecasts[feature_name]["polynomial"] = model.predict(X_test_poly)
    #         # self.one_step_forward_forecast['polynomial'] = model.predict(X_test_poly[-1,].reshape(1, -1))
    #         self.metrics[feature_name]["polynomial"] = {
    #             "Polynomial Regression Model Test RMSE": root_mean_squared_error(
    #                 y_test, self.forecasts[feature_name]["polynomial"]
    #             ),
    #             "Polynomial Regression Model Adjusted_r2": 1
    #             - (1 - model.score(X_train_poly, y_train)) * (n - 1) / (n - p - 1),
    #         }
    #     return self

    # def predict_next_day_and_update_with_features(
    #     self, model, feature_cols, target_col, scaler=None
    # ):
    #     """
    #     Predict the next day's value using the last row of the dataset, update predictors, and append to the dataset.

    #     Parameters:
    #         model: The forecasting model (e.g., LSTM, ARIMA, etc.).
    #         data: DataFrame containing the dataset.
    #         feature_cols: List of feature column names.
    #         target_col: Name of the target column.
    #         scaler: Optional scaler for inverse transformation of predictions.

    #     Returns:
    #         updated_data: DataFrame updated with the predicted value.
    #         next_day_prediction: The forecasted value for the next day.
    #     """
    #     # Get the last row's features
    #     last_features = self.data.loc[self.data.index[-1], feature_cols].values.reshape(
    #         1, -1
    #     )

    #     # Perform prediction
    #     next_day_prediction = model.predict(last_features)

    #     # Inverse transform the prediction if a scaler is provided
    #     if scaler is not None:
    #         next_day_prediction = scaler.inverse_transform(
    #             next_day_prediction.reshape(-1, 1)
    #         ).flatten()

    #     # Prepare a new row with updated predictors
    #     new_row = {col: 0 for col in self.data.columns}  # Initialize with zeros

    #     # Update predictors (example: using last values or derived metrics)
    #     for col in feature_cols:
    #         if col == target_col:  # Skip the target column in features
    #             continue
    #         if "lag" in col:  # Example for lagged features
    #             lag_step = int(col.split("_")[-1])  # Extract lag step (e.g., 'lag_1')
    #             new_row[col] = self.data.loc[self.data.index[-lag_step], target_col]
    #         elif "ma" in col:  # Example for moving averages
    #             window_size = int(
    #                 col.split("_")[-1]
    #             )  # Extract window size (e.g., 'ma_5')
    #             new_row[col] = self.data[target_col].iloc[-window_size:].mean()
    #         else:
    #             new_row[col] = self.data.loc[
    #                 self.data.index[-1], col
    #             ]  # Static predictors

    #     # Add the predicted target value
    #     new_row[target_col] = next_day_prediction[0]

    #     # Append the new row to the dataset
    #     updated_data = self.data.append(new_row, ignore_index=True)

    #     return updated_data, next_day_prediction[0]

    def prepare_models(self, predictors: list[str], horizon, weight: False):
        """
        Prepare models for each predictor.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        target_column : str
            The target column to predict
        """
        self.models = {}
        self.scalers = {}
        self.transformers = {}

        for predictor in predictors:
            # Select features excluding the current predictor
            features = [col for col in predictors if col != predictor]

            # Prepare data
            X = self.data[features].iloc[:-horizon,]
            y = self.data[predictor].iloc[:-horizon,]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Polynomial features
            poly = PolynomialFeatures(degree=2)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)

            if weight is True:

                allowed_columns = list(self.data.iloc[:, 8:].columns)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                filtered_columns = [
                    col for col in X_train.columns if col in allowed_columns
                ]
                filtered_df = X_train.loc[:, filtered_columns]
                # filtered_df = X_train
                rf_model.fit(filtered_df, y_train)
                importances = rf_model.feature_importances_ * len(filtered_columns)

                importance_df = pd.DataFrame(
                    {"Feature": filtered_df.columns, "Importance": importances}
                )
                importance_df["Weight"] = (
                    importance_df["Importance"] / importance_df["Importance"].sum()
                )

                scaler_weight = StandardScaler()
                X_train[filtered_columns] = scaler_weight.fit_transform(
                    X_train[filtered_columns]
                )
                X_train[filtered_columns] *= importances

            # Perform random search
            # best_order, best_model = random_search_arima(y, p_range, d_range, q_range, n_iter=100, random_state=42)

            # print(f"Best ARIMA order: {best_order}")

            # Train models
            models = {
                "linear": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "polynomial": Ridge(alpha=1.0),
                "arimaxgb": ARIMAXGBoost(),
                # best_model
                # ARIMAXGBoost()  # q =5 is ok    q
            }

            # reshape

            # Fit models
            models["linear"].fit(X_train, y_train)
            models["ridge"].fit(X_train_scaled, y_train)
            models["polynomial"].fit(X_train_poly, y_train)
            models["arimaxgb"].fit(X_train, y_train)
            # if predictor == 'Close':
            #     models['arimaxgb'].fine_tune_boost(X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test)

            for name, model in models.items():

                if name == "linear":
                    y_pred = model.predict(X_test)
                    r2 = r2 = 1 - (1 - model.score(X_test, y_test))

                    # r2 = r2 = 1 - (1 -model.score(X_test, y_test))
                elif name == "ridge":
                    y_pred = model.predict(scaler.transform(X_test))
                    r2 = r2 = 1 - (1 - model.score(X_test_scaled, y_test))
                elif name == "polynomial":
                    y_pred = model.predict(poly.transform(scaler.transform(X_test)))
                    r2 = r2 = 1 - (1 - model.score(X_test_poly, y_test))

                # Compute metrics

                mse = root_mean_squared_error(y_test, y_pred)

                print(f"{predictor} - {name.capitalize()} Model:")
                print(f"  Mean Squared Error: {mse:.4f}")
                print(f"  R² Score: {r2:.4f}")

            # Store models, scalers, and transformers
            self.models[predictor] = models
            self.scalers[predictor] = scaler
            self.transformers[predictor] = poly

    def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
        """
        Perform one-step forward predictions for all predictors.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        model_type : str
            one of the
        horizon : int, optional
            Number of days to forecast (default: 20)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]z
            Forecasted data and backtest data
        """
        # Ensure models are prepared
        if not self.models:
            raise ValueError("Please run prepare_models() first")

        # Initialize prediction and backtest DataFrames
        prediction = self.data[predictors].copy().iloc[-horizon:,].dropna()
        backtest = self.data[predictors].copy().iloc[:-horizon,].dropna()
        observation = self.data[predictors].copy().dropna()
        

        # last_row = prediction.iloc[-1]
        # last_row1 = backtest.iloc[-1]

        # Dictionaries to store predictions for each predictor and model
        predictions = {
            predictor: {
                model: []
                for model in ["ridge", "linear", "polynomial", "arimaxgb", "combined"]
            }
            for predictor in predictors
        }

        for _ in range(horizon):
            # Predict each predictor using different models
            last_row = prediction.iloc[-horizon:,]
            last_row1 = backtest.iloc[-horizon:,]

            for predictor in predictors:
                # Select features excluding the current predictor
                features = [col for col in predictors if col != predictor]

                # Predictions using different models
                models = self.models[predictor]
                scaler = self.scalers[predictor]
                transformer = self.transformers[predictor]

                # Calculate rolling averages for MA_50 and MA_200 based on the Close price
                if predictor == "MA_50":
                    rolling_avg_50 = (
                        pd.Series(
                            np.concatenate(
                                (
                                    observation["Close"],
                                    predictions["Close"]["arimaxgb"][-2],
                                )
                            )
                        )
                        .rolling(window=50)
                        .mean()
                        .iloc[-1]
                    )
                    predictions[predictor]["arimaxgb"].append(rolling_avg_50)

                    rolling_avg_50_1 = (
                        pd.Series(
                            np.concatenate(
                                (
                                    backtest["Close"],
                                    predictions["Close"]["arimaxgb"][-1],
                                )
                            )
                        )
                        .rolling(window=50)
                        .mean()
                        .iloc[-1]
                    )
                    predictions[predictor]["arimaxgb"].append(rolling_avg_50_1)
                elif predictor == "MA_200":
                    rolling_avg_200 = (
                        pd.Series(
                            np.concatenate(
                                (
                                    observation["Close"],
                                    predictions["Close"]["arimaxgb"][-2],
                                )
                            )
                        )
                        .rolling(window=200)
                        .mean()
                        .iloc[-1]
                    )
                    predictions[predictor]["arimaxgb"].append(rolling_avg_200)

                    rolling_avg_200_1 = (
                        pd.Series(
                            np.concatenate(
                                (
                                    backtest["Close"],
                                    predictions["Close"]["arimaxgb"][-1],
                                )
                            )
                        )
                        .rolling(window=200)
                        .mean()
                        .iloc[-1]
                    )
                    predictions[predictor]["arimaxgb"].append(rolling_avg_200_1)

                else:

                    average_row = np.average(
                        last_row[features],
                        axis=0,
                        weights=[0.1 * i for i in range(1, len(last_row) + 1)],
                    )
                    average_row1 = np.average(
                        last_row1[features],
                        axis=0,
                        weights=[0.1 * i for i in range(1, len(last_row1) + 1)],
                    )

                    # Prepare input data
                    input_data = average_row
                    input_data1 = average_row1

                    # Make predictions

                    pred_arima_ml = models["arimaxgb"].predict(
                        input_data.reshape(1, -1)
                    )
                    # pred_arima_ml = models["arimaxgb"].predict(input_data.values.reshape(1, -1))
                    # pred_arima_ml = models["arimaxgb"].predict(input_data)

                    # pred_ridge = models['ridge'].predict(scaler.transform(input_data.values.reshape(1, -1)))[0]
                    # pred_linear = models['linear'].predict(input_data.values.reshape(1, -1))[0]
                    # pred_poly = models['polynomial'].predict(
                    #     transformer.transform(scaler.transform(input_data.values.reshape(1, -1)))
                    # )[0]
                    # if predictor not in ['Daily Returns', 'Volatility']:
                    #     pred_combined = models["arimaxgb"].predict(input_data.values.reshape(1, -1))
                    # else:
                    #     pred_combined =  models['linear'].predict(input_data.values.reshape(1, -1))[0]

                    pred_arima_ml1 = models["arimaxgb"].predict(
                        input_data1.reshape(1, -1)
                    )
                    # pred_arima_ml1 = models["arimaxgb"].predict(input_data1.values.reshape(1, -1))
                    # pred_arima_ml1 = models["arimaxgb"].predict(input_data1)

                    # pred_ridge1 = models['ridge'].predict(scaler.transform(input_data1.values.reshape(1, -1)))[0]
                    # pred_linear1 = models['linear'].predict(input_data1.values.reshape(1, -1))[0]
                    # pred_poly1 = models['polynomial'].predict(
                    #     transformer.transform(scaler.transform(input_data1.values.reshape(1, -1)))
                    # )[0]
                    # if predictor not in ['Daily Returns', 'Volatility']:
                    #     pred_combined1 = models["arimaxgb"].predict(input_data1.values.reshape(1, -1))
                    # else:
                    #     pred_combined1 =  models['linear'].predict(input_data1.values.reshape(1, -1))[0]

                    # Store predictions
                    # predictions[predictor]['ridge'].append(pred_ridge)
                    # predictions[predictor]['linear'].append(pred_linear)
                    # predictions[predictor]['polynomial'].append(pred_poly)
                    predictions[predictor]["arimaxgb"].append(pred_arima_ml)
                    # predictions[predictor]['combined'].append(pred_combined)

                    # predictions[predictor]['ridge'].append(pred_ridge1)
                    # predictions[predictor]['linear'].append(pred_linear1)
                    # predictions[predictor]['polynomial'].append(pred_poly1)
                    predictions[predictor]["arimaxgb"].append(pred_arima_ml1)
                # predictions[predictor]['combined'].append(pred_combined1)

            # Create new row with predictions
            next_day_date = get_next_valid_date(pd.Timestamp(last_row.iloc[-1].name))
            next_day_date1 = get_next_valid_date(pd.Timestamp(last_row1.iloc[-1].name))
            if "Low" in predictors:
                if predictor == "Close":
                    if (
                        predictions["Close"][model_type][-2]
                        > predictions["Low"][model_type][-2]
                    ):
                        predictions["Close"][model_type][-2] = predictions["Low"][
                            model_type
                        ][-2]
                    elif (
                        predictions["Close"][model_type][-1]
                        > predictions["Low"][model_type][-1]
                    ):
                        predictions["Close"][model_type][-1] = predictions["Low"][
                            model_type
                        ][-1]

            new_row = pd.DataFrame(
                {
                    predictor: predictions[predictor][model_type][-2]
                    for predictor in predictors
                },  # take the last one
                index=[next_day_date],
            )
            new_row1 = pd.DataFrame(
                {
                    predictor: predictions[predictor][model_type][-1]
                    for predictor in predictors
                },  # take the last one
                index=[next_day_date1],
            )

            # Update DataFrames
            prediction = pd.concat([prediction, new_row])
            observation = pd.concat([observation, new_row])
            last_row = prediction.iloc[-1]
            # last_row = prediction.iloc[-5:,]
            backtest = pd.concat([backtest, new_row1])
            last_row1 = backtest.iloc[-1]

            # last_row1 = backtest.iloc[-5:,]

        return prediction, backtest

    def plot_ridge_alpha_analysis(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Plot Ridge regression alpha vs RMSE analysis"""

        alphas = np.logspace(-6, 4, 600)
        test_rmse = []

        for alpha in alphas:
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_train_scaled, y_train)
            rmse = root_mean_squared_error(y_test, ridge.predict(X_test_scaled))
            test_rmse.append(rmse)

        plt.figure(figsize=(12, 6))
        plt.plot(
            alphas, test_rmse, label="Ridge regression", color="lime", linestyle="--"
        )

        # Find and plot minimum
        i = np.argmin(test_rmse)
        x_min = alphas[i]
        y_min = test_rmse[i]
        plt.plot(x_min, y_min, marker="o")

        plt.title("Ridge Regression: Lambda vs Test RMSE")
        plt.xlim(left=0, right=20)
        plt.xlabel("Lambda")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()

        return x_min, y_min

    def plot_predictions(self):
        """Plot actual vs predicted values for all models"""
        for feature_name, feature_set in self.feature_sets.items():
            y_test_df = feature_set["y_test_df"]
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_df.index, y_test_df, label="Actual", color="black")
            # self.forecasts[feature_name]['polynomial']
            for model, pred in self.forecasts[feature_name].items():
                if model != "sarima" and model != "knn":
                    plt.plot(
                        y_test_df.iloc[-len(pred) :,].index,
                        pred,
                        label=f"{model}",
                        linestyle="dashed",
                        alpha=0.7,
                    )

            plt.title(f"{feature_name} Stock Price Predictions")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.show()

    def print_metrics(self):
        """Print performance metrics for all models"""
        print("\nModel Performance Metrics:")
        print("-" * 50)
        for name, metric in self.metrics.items():
            if isinstance(metric, dict):
                print(f"{name.upper()}:")
                print(f"RMSE: {metric['rmse']:.2f}")
                print(f"R²: {metric['r2']:.2f}")
            else:
                print(f"{name.upper()}:")
                print(f"RMSE: {metric:.2f}")
            print("-" * 50)

    def full_workflow(
        start_date, end_date, predictors=None, companies=None, stock_settings={}
    ):
        """
        This function is used to output the prediction of the stock price for the future based on the stock price data from the start date to the end date.

        Args:
        start_date (str): The start date of the stock price data
        end_date (str): The end date of the stock price data
        predictors (list): The list of predictors used to predict the stock price
        companies (list): The list of company names of the stocks
        """
        default_horizons = [5, 7, 10]
        default_weight = False
        if companies is None:
            companies = ["AXP"]
        for company in companies:
            prediction_dataset = StockPredictor(
                company,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )
            prediction_dataset.load_data()
            if predictors is None:
                predictors = [
                    "Close",
                    "MA_50",
                    "MA_200",
                    "SP500",
                    "TNX",
                    "USDCAD=X",
                    "Tech",
                    "Fin",
                    "VIX",
                ] + [
                    "rolling_min",
                    "rolling_median",
                    "rolling_sum",
                    "rolling_ema",
                    "rolling_25p",
                    "rolling_75p",
                ]
            predictors = predictors

            predictor = prediction_dataset
            if len(stock_settings) != 0 or company in stock_settings:
                # Use custom settings for the stock
                settings = stock_settings[company]
                horizons = settings["horizons"]
                weight = settings["weight"]
            else:
                # Use default settings for other stocks
                horizons = default_horizons
                weight = default_weight
            for horizon in horizons:
                prediction_dataset.prepare_models(
                    predictors, horizon=horizon, weight=weight
                )
                # prediction_dataset._evaluate_models('Close')
                prediction, backtest = (
                    predictor.one_step_forward_forecast(
                        predictors, model_type="arimaxgb", horizon=horizon
                    )
                )
                # print(prediction)
                # print(backtest)

                # Data Viz (Not that key)
                plt.figure(figsize=(12, 6))

                first_day = pd.to_datetime(end_date - timedelta(days=5 + horizon))

                plt.plot(
                    prediction[
                        prediction.index >= prediction_dataset.data.iloc[-1].name
                    ].index,
                    prediction[
                        prediction.index >= prediction_dataset.data.iloc[-1].name
                    ].Close,
                    label="Prediction",
                    color="blue",
                )
                plt.plot(
                    backtest[backtest.index > first_day].index,
                    backtest[backtest.index > first_day].Close,
                    label="Backtest",
                    color="red",
                )
                plt.plot(
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index > first_day
                    ].index,
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index > first_day
                    ],
                    label="Actual",
                    color="black",
                )
                # cursor(hover=True)
                plt.title(
                    f"Price Prediction ({prediction_dataset.symbol}) (horizon = {horizon}) (weight = {weight})"
                )
                plt.axvline(
                    x=backtest.index[-1],
                    color="g",
                    linestyle="--",
                    label="Reference Line (Last Real Data Point)",
                )
                plt.text(
                    backtest.index[-1],
                    backtest.Close[-1],
                    f"x={str(backtest.index[-1].date())}",
                    ha="right",
                    va="bottom",
                )

                plt.xlabel("Date")
                plt.ylabel("Stock Price")
                plt.legend()
                plt.show()


# Example usage
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", start_date="2020-01-01")