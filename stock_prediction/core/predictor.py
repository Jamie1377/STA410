import yfinance as yf
import pandas as pd
import numpy as np
from numpy.polynomial import polynomial 
from datetime import date, timedelta
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from sklearn.model_selection import train_test_split
from stock_prediction.core import ARIMAXGBoost
from stock_prediction.utils import get_next_valid_date
from hmmlearn.hmm import GaussianHMM

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
        self.history = []  # New attribute for error correction

    def _compute_rsi(self, window=14):
        """Custom RSI implementation"""
        delta = self.data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        return 100 - (
            100 / (1 + (gain.rolling(window).mean() / loss.rolling(window).mean()))
        )

    def _compute_atr(self, window=14):
        """Average True Range"""
        high_low = self.data["High"] - self.data["Low"]
        high_close = (self.data["High"] - self.data["Close"].shift()).abs()
        low_close = (self.data["Low"] - self.data["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def load_data(self):
        """Load and prepare stock data with features"""
            
        # Add momentum-specific features
        window = 15  # Standard momentum window
        self.data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
        )
        self.data.columns = self.data.columns.get_level_values(0)  # Remove multi-index
        self.data.ffill()
        self.data.dropna()
        # Add technical indicators
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()
        self.data["MA_7"] = self.data["Close"].rolling(window=7).mean()
        self.data["MA_21"] = self.data["Close"].rolling(window=21).mean()


        # Fourier transform
        data_FT = self.data.copy().reset_index()[["Date",'Close']]
        close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))

        self.data["FT_real"] = np.real(close_fft)
        self.data["FT_img"] = np.imag(close_fft)
        # fft_df = pd.DataFrame({'fft': close_fft})
        # fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        # fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        # fft_list = np.asarray(fft_df['fft'].tolist())
        # for num_ in [3, 6, 9, 100]:
        #     fft_list_m10 = np.copy(fft_list)
        #     fft_list_m10[num_:-num_] = 0  
        #     complex_num = np.fft.ifft(fft_list_m10)
        #     self.data[f'Fourier_trans_{num_}_comp_real'] = np.real(complex_num)
        #     self.data[f'Fourier_trans_{num_}_comp_img'] = np.imag(complex_num)


        from sklearn.decomposition import PCA

        X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
        pca = PCA(n_components=2)  # Keep top 2 components
        X_pca = pca.fit_transform(X_fft)

        for i in range(X_pca.shape[1]):
            self.data[f'Fourier_PCA_{i}'] = X_pca[:, i]

                    



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
        stock_data.index.name = "Date"  # Ensure the index is named "Date"

        # Advanced Momentum
        self.data["RSI"] = self._compute_rsi(window=14)
        self.data["MACD"] = (
            self.data["Close"].ewm(span=12).mean()
            - self.data["Close"].ewm(span=26).mean()
        )
        # 2. Williams %R
        high_max = self.data['High'].rolling(window).max()
        low_min = self.data['Low'].rolling(window).min()
        self.data['Williams_%R'] = ((high_max - self.data['Close']) / 
                                (high_max - low_min)) * -100

             
        # 3. Stochastic Oscillator
        self.data['Stochastic_%K'] = ((self.data['Close'] - low_min) / 
                                    (high_max - low_min)) * 100
        self.data['Stochastic_%D'] = self.data['Stochastic_%K'].rolling(3).mean()

        # 4. Momentum Divergence Detection
        self.data['Price_Change'] = self.data['Close'].diff()
        self.data['Momentum_Divergence'] = (self.data['Price_Change'] * 
                                        self.data['MACD'].diff()).rolling(5).sum()
        
       

        # Volatility-adjusted Channels
        self.data["ATR"] = self._compute_atr(window=14)
        self.data["Upper_Bollinger"] = (
            self.data["MA_21"] + 2 * self.data["Close"].rolling(50).std()
        )
        self.data["Lower_Bollinger"] = (
            self.data["MA_21"] - 2 * self.data["Close"].rolling(50).std()
        )

        # Volume-based Features
        # self.data['OBV'] = self._compute_obv()
        self.data["VWAP"] = (
            self.data["Volume"]
            * (self.data["High"] + self.data["Low"] + self.data["Close"])
            / 3
        ).cumsum() / self.data["Volume"].cumsum()
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)["Close"]
        # Economic Indicators
        # Fetch S&P 500 Index (GSPC) and Treasury Yield ETF (IEF) from Yahoo Finance
        sp500 = sp500 - sp500.mean()
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

        # Get the NYSE trading schedule
        nyse = mcal.get_calendar("NYSE")
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
        # self.data["Daily Returns"] = self.data["Close"].pct_change()
        self.data["Daily Returns"] = self.data["Close"].pct_change(window) * 100 # Percentage change in the standard window for the momentum
        self.data["Volatility"] = self.data["Daily Returns"].rolling(window=20).std()
         # 5. Adaptive Momentum Score
        vol_weight = self.data['Volatility'] * 100
        self.data['Momentum_Score'] = (self.data['RSI'] * 0.4 + 
                                    self.data['Daily Returns'] * 0.3 + 
                                    self.data['Williams_%R'] * 0.3) / (1 + vol_weight)
        # Drop rows with NaN values
        self.data['Momentum_Interaction'] = self.data['RSI'] * self.data['Daily Returns']
        self.data['Volatility_Adj_Momentum'] = self.data['Momentum_Score'] / (1 + self.data['Volatility'])
        self.data['Volatility_Adj_Momentum'] = self.data['Volatility_Adj_Momentum'].clip(lower=0.1)
        self.data['Volatility_Adj_Momentum'] = self.data['Volatility_Adj_Momentum'].clip(upper=10.0)
        self.data['Volatility_Adj_Momentum'] = self.data['Volatility_Adj_Momentum'].fillna(0.0)

       

        # Prepare features for HMM
        
        hmm = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
        hmm.fit(self.data["Close"].pct_change().dropna().values.reshape(-1,1))
        print(self.data["Close"].pct_change().dropna().values.reshape(-1,1))

        # Predict hidden states
        self.data['Market_State'] = 1
        self.data['Market_State'].iloc[1:] = hmm.predict(self.data["Close"].pct_change().dropna().values.reshape(-1,1))

        




        self.data = self.data.dropna()

        # Process each feature set
        # for name, config in self.feature_sets.items():
        #     target = config["target"]
        #     X = self.data.drop(columns=[target]).values
        #     y = self.data[target].values

        #     X_df = self.data.drop(columns=[target])
        #     y_df = self.data[target]

        #     # Train test split
        #     X_train, X_test, y_train, y_test = train_test_split(
        #         X, y, test_size=0.2, shuffle=False, random_state=42
        #     )

        #     self.X_train_df, self.X_test_df, self.y_train_df, self.y_test_df = (
        #         train_test_split(
        #             X_df, y_df, test_size=0.2, shuffle=False, random_state=42
        #         )
        #     )

        #     # Scale features
        #     scaler = StandardScaler()
        #     X_train_scaled = scaler.fit_transform(X_train)
        #     X_test_scaled = scaler.transform(X_test)

        #     # Polynomial features
        #     transformer = PolynomialFeatures(degree=2, include_bias=False)
        #     X_train_poly = transformer.fit_transform(X_train_scaled)
        #     X_test_poly = transformer.transform(X_test_scaled)

        #     # Store transformed data
        #     self.feature_sets[name].update(
        #         {
        #             "X_train": X_train,
        #             "X_test": X_test,
        #             "y_train": y_train,
        #             "y_test": y_test,
        #             "X_train_df": self.X_train_df,
        #             "X_test_df": self.X_test_df,
        #             "y_train_df": self.y_train_df,
        #             "y_test_df": self.y_test_df,
        #             "X_train_scaled": X_train_scaled,
        #             "X_test_scaled": X_test_scaled,
        #             "X_train_poly": X_train_poly,
        #             "X_test_poly": X_test_poly,
        #         }
        #     )
        #     self.scalers[name] = scaler
        #     self.transformers[name] = transformer

        return self

    def prepare_models(
        self, predictors: list[str], horizon, weight: bool = False, refit: bool = True
    ):
        """
        Prepare models for each predictor.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        horizon : int
            Number of days to forecast
        weight : bool
            Whether to apply feature weighting
        refit : bool
            Whether to refit models on full data
        """
        self.models = {}
        self.scalers = {}
        self.transformers = {}
        self.feature_importances = {}

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
            # X_train_poly = poly.fit_transform(X_train_scaled)
            # degree = 2
            # coefs = polynomial.polyfit(X_train, y_train, deg=degree)
            # X_train_poly = np.column_stack([X_train**i for i in range(degree + 1)])


            degree = 2
            X_train_poly = np.zeros_like(X_train)  # Preserve the shape
            X_test_poly = np.zeros_like(X_test)
            for i in range(X_train.shape[1]):  # Loop over each feature
                coef = np.polynomial.polynomial.polyfit(X_train.iloc[:, i], y_train, degree)  # Fit polynomial
                X_train_poly[:, i] = np.polynomial.polynomial.polyval(X_train.iloc[:, i], coef)  # Transform X
                X_test_poly[:, i] = np.polynomial.polynomial.polyval(X_test.iloc[:, i], coef)    # Transform X_test



           




            # X_test_poly = poly.transform(X_test_scaled)


            # Train models
            models = {
                "linear": LinearRegression(),
                "ridge": Ridge(alpha=1.0),
                "polynomial": LinearRegression(),
                "arimaxgb": ARIMAXGBoost(),
            }

            # Feature importance
            feature_weights = np.ones(len(features))
            # if weight is True:
            #     allowed_columns = list(self.data.iloc[:, 8:].columns)
            #     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            #     filtered_columns = [
            #         col for col in X_train.columns if col in allowed_columns
            #     ]
            #     filtered_df = X_train.loc[:, filtered_columns]
            #     # filtered_df = X_train
            #     rf_model.fit(filtered_df, y_train)
            #     importances = rf_model.feature_importances_ * len(filtered_columns)

            #     importance_df = pd.DataFrame(
            #         {"Feature": filtered_df.columns, "Importance": importances}
            #     )
            #     importance_df["Weight"] = (
            #         importance_df["Importance"] / importance_df["Importance"].sum()
            #     )

            #     scaler_weight = StandardScaler()
            #     X_train[filtered_columns] = scaler_weight.fit_transform(
            #         X_train[filtered_columns]
            #     )
            # X_train[filtered_columns] *= importances

            


            





            from sklearn.ensemble import (
                RandomForestRegressor,
                GradientBoostingRegressor,
            )
            from sklearn.model_selection import cross_val_score

            # def advanced_feature_weighting(X, y):
            #     # Multiple models for feature importance
            #     models = [
            #         RandomForestRegressor(n_estimators=100),
            #         GradientBoostingRegressor(n_estimators=100),
            #         ExtraTreesRegressor(n_estimators=100),
            #     ]

            #     # Collect feature importances from multiple models
            #     all_importances = []
            #     for model in models:
            #         model.fit(X, y)
            #         all_importances.append(model.feature_importances_)

            #     # Aggregate importances
            #     avg_importances = np.mean(all_importances, axis=0)

            #     # Normalize and apply cross-validation performance weighting
            #     cv_scores = []
            #     for model in models:
            #         scores = cross_val_score(
            #             model, X, y, cv=5, scoring="neg_mean_squared_error"
            #         )
            #         cv_scores.append(-np.mean(scores))

            #     # Weight importances by model performance
            #     model_weights = 1 / np.array(cv_scores)
            #     model_weights /= np.sum(model_weights)

            #     # Final feature weights
            #     weighted_importances = np.zeros_like(avg_importances)
            #     for i, (imp, mw) in enumerate(zip(all_importances, model_weights)):
            #         weighted_importances += imp * mw

            #     return weighted_importances / np.sum(weighted_importances)
            def advanced_feature_weighting(X, y):
                """Modified to ensure stable weights"""
                models = [
                    RandomForestRegressor(n_estimators=100),
                    GradientBoostingRegressor(n_estimators=100),
                    ExtraTreesRegressor(n_estimators=100)
                ]
                
                # Calculate normalized importances
                all_importances = []
                for model in models:
                    model.fit(X, y)
                    if hasattr(model, "feature_importances_"):
                        imp = model.feature_importances_
                    else:
                        imp = np.abs(model.coef_)  # For linear models
                    all_importances.append(imp / np.sum(imp))  # Normalized
                
                # Geometric mean instead of average
                avg_importances = np.exp(np.mean(np.log(all_importances + 1e-8), axis=0))
                
                # Cross-validation weighting
                cv_scores = []
                for model in models:
                    scores = cross_val_score(model, X, y, cv=5, 
                                        scoring='neg_mean_squared_error')
                    cv_scores.append(-np.mean(scores))
                
                # Softmax weighting
                model_weights = np.exp(cv_scores) / np.sum(np.exp(cv_scores))
                
                # Final weights
                weighted_importances = np.zeros_like(avg_importances)
                for i, (imp, mw) in enumerate(zip(all_importances, model_weights)):
                    weighted_importances += imp * mw
                
                return weighted_importances / np.sum(weighted_importances)

            if weight is True:

                # # Attempt to get feature importances using Random Forest
                # rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                # rf_model.fit(X_train, y_train)

                # # Get feature importances
                # importances = rf_model.feature_importances_

                # # Normalize importances
                # feature_weights = importances / np.sum(importances)
                feature_weights = advanced_feature_weighting(X_train, y_train)

                # 1. Ensure positive normalized weights
                feature_weights = np.abs(feature_weights)  # Force non-negative
                feature_weights += 1e-8  # Prevent zero division
                feature_weights /= np.sum(feature_weights)  # Normalize to sum=1

                # Debug: Print feature weights
                print(f"\nFeature weights for {predictor}:")
                for feat, w in zip(features, feature_weights):
                    print(f"{feat}: {w:.4f}")

                # 2. Apply weights BEFORE scaling
                X_train_weighted = X_train.copy()
                for i, feat in enumerate(features):
                    X_train_weighted[feat] *= feature_weights[i] * 100  # Scale to preserve magnitude




                 # 3. Use same scaler for train/test
                scaler_1 = StandardScaler()
                X_train_scaled_weighted = scaler_1.fit_transform(X_train_weighted)
                # X_test_scaled = scaler_1.transform(X_test[features])  # Use original test features
                
        
                # # Store feature importances for reference
                # self.feature_importances[predictor] = dict(
                #     zip(features, feature_weights)
                # )

                # # Apply weights to training data
                # X_train_weighted = X_train.copy()
                # for i, feat in enumerate(features):
                #     X_train_weighted[feat] *= feature_weights[i]

                # X_train_scaled_weighted = scaler.transform(X_train_weighted)
                X_train_poly_weighted = poly.transform(X_train_scaled_weighted)

                # Weighted fitting for applicable models
                models["linear"].fit(X_train_weighted, y_train)
                models["ridge"].fit(X_train_scaled_weighted, y_train)
                models["polynomial"].fit(X_train_poly_weighted, y_train)
                models["arimaxgb"].fit(X_train_weighted, y_train)

            else:

                # Fit models
                models["linear"].fit(X_train, y_train)
                models["ridge"].fit(X_train_scaled, y_train)
                models["polynomial"].fit(X_train_poly, y_train)
                models["arimaxgb"].fit(X_train, y_train)

            for name, model in models.items():

                if name == "linear":
                    y_pred = model.predict(X_test)
                       # 1 - (1 - model.score(X_test, y_test))
                elif name == "ridge":
                    y_pred = model.predict(scaler.transform(X_test))
                       # 1 - (1 - model.score(X_test_scaled, y_test))
                elif name == "polynomial":
                    # y_pred = model.predict(poly.transform(scaler.transform(X_test)))

                    
                    degree = 2
                    # X_test_poly = np.column_stack([X_test**i for i in range(degree + 1)])
                    y_pred = model.predict(X_test_poly)
                       # 1 - (1 - model.score(X_test_poly, y_test))
                elif name == "arimaxgb":
                    y_pred = model.predict(X_test)
                
                # Compute adjusted R^2  # original one r2_score(y_test, y_pred)
                r2 = r2_score(y_true=y_test, y_pred=y_pred) 
                adj_r2 = 1 - ( 1 - r2_score(y_true=y_test, y_pred=y_pred) ) * ( X_test.shape[0] - 1 ) / (  X_test.shape[0] - X_test.shape[1] - 1 ) 

                # Compute metrics
                rmse = root_mean_squared_error(y_test, y_pred)

                print(f"{predictor} - {name.capitalize()} Model:")
                print(f"  Test Mean Squared Error: {rmse:.4f}")
                print(f"  R² Score: {r2:.4f}")
            print(
                "-----------------------------------------------------------------------------------------"
            )

            # Store models, scalers, and transformers
            self.models[predictor] = models
            self.scalers[predictor] = scaler
            self.transformers[predictor] = poly

            if refit is True:
                # Refit models on full data
                refit_models = {
                    "linear": LinearRegression(),
                    "ridge": Ridge(alpha=1.0),
                    "polynomial": LinearRegression(), #Ridge(alpha=1.0),
                    "arimaxgb": ARIMAXGBoost(),
                }
                refit_models["linear"].fit(X, y)
                refit_models["ridge"].fit(scaler.transform(X), y)
                # refit_models["polynomial"].fit(poly.transform(scaler.transform(X)), y)
                degree = 2
                X_poly = np.zeros_like(X)  # Preserve the shape
                for i in range(X.shape[1]):  # Loop over each feature
                    coef = np.polynomial.polynomial.polyfit(X.iloc[:, i], y, degree)  # Fit polynomial
                    X_poly[:, i] = np.polynomial.polynomial.polyval(X.iloc[:, i], coef)  # Transform X
                refit_models["polynomial"].fit(X_poly,y)
                refit_models["arimaxgb"].fit(X, y)
                self.models[predictor] = refit_models

    def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
        """
        Perform one-step forward predictions for all predictors with enhanced methods.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        model_type : str
            one of the model types
        horizon : int
            Number of days to forecast

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Forecasted data and backtest data
        """
        # Ensure models are prepared
        if not self.models:
            raise ValueError("Please run prepare_models() first")

        # Initialize prediction and backtest DataFrames
        prediction = self.data[predictors].copy().iloc[-horizon:].dropna()
        backtest = self.data[predictors].copy().iloc[:-horizon].dropna()
        observation = self.data[predictors].copy().dropna()

        # Initialize arrays for storing predictions
        pred_array = np.zeros((horizon, len(predictors)))
        raw_pred_array = np.zeros((horizon, len(predictors)))
        backtest_array = np.zeros((horizon, len(predictors)))
        raw_backtest_array = np.zeros((horizon, len(predictors)))

        # Create maps for quick lookup
        pred_dates = []
        backtest_dates = []
        predictor_indices = {p: i for i, p in enumerate(predictors)}

        # Initialize error correction mechanisms
        # 1. Base correction factors
        error_correction = {predictor: 1.0 for predictor in predictors}

        # 2. Feature-specific correction bounds
        price_vars = ["Open", "High", "Low", "Close"]
        bounds = {}
        for p in predictors:
            if p in price_vars:
                bounds[p] = (0.95, 1.05)  # Tighter bounds for prices
            elif p.startswith("MA_"):
                bounds[p] = (0.97, 1.03)  # Even tighter for moving averages
            else:
                bounds[p] = (0.6, 1.4)  # Wider for other indicators

        # 3. Initialize regime detection
        regime = "normal"  # Default regime
        price_changes = []

        # 4. Initialize Kalman filter parameters (simplified)
        kalman_gain = {p: 0.2 for p in predictors}
        error_variance = {p: 1.0 for p in predictors}

        # 5. Create ensembles of correction factors
        ensemble_corrections = {p: [0.935, 1.0, 1.035] for p in predictors}
        ensemble_weights = {p: np.array([1 / 3, 1 / 3, 1 / 3]) for p in predictors}

        # Calculate initial volatility (if Close is in predictors)
        if "Close" in predictors:
            close_history = observation["Close"].tail(20)
            returns = close_history.pct_change().dropna()
            current_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            current_volatility = 0.2  # Default volatility assumption

        # Helper functions
        def update_regime(prev_values, new_value):
            """Update market regime based on recent price action"""
            if len(prev_values) < 2:
                return "normal"

            # Calculate recent returns
            recent_returns = np.diff(prev_values) / prev_values[:-1]

            # Calculate volatility
            vol = np.std(recent_returns) * np.sqrt(252)

            # Detect trend
            trend = sum(1 if r > 0 else -1 for r in recent_returns)

            if vol > 0.4:  # High volatility threshold
                return "volatile"
            elif abs(trend) > len(recent_returns) * 0.7:  # Strong trend
                return "trending"
            else:
                return "mean_reverting"

        def adaptive_bounds(predictor, volatility, regime):
            """Calculate adaptive bounds based on volatility and regime"""
            base_lower, base_upper = bounds[predictor]

            # Adjust bounds based on regime
            if regime == "volatile":
                # Wider bounds during volatility
                lower = base_lower - 0.1
                upper = base_upper + 0.1
            elif regime == "trending":
                # Asymmetric bounds for trending markets
                if predictor in price_vars:
                    recent_trend = (
                        np.mean(price_changes[-5:]) if len(price_changes) >= 5 else 0
                    )
                    if recent_trend > 0:
                        # Uptrend - allow more upside correction
                        lower = base_lower
                        upper = base_upper + 0.1
                    else:
                        # Downtrend - allow more downside correction
                        lower = base_lower - 0.1
                        upper = base_upper
                else:
                    lower, upper = base_lower, base_upper
            else:
                # Default bounds
                lower, upper = base_lower, base_upper

            # Further adjust based on volatility
            vol_factor = min(1.0, volatility / 0.2)  # Normalize volatility
            lower -= 0.05 * vol_factor
            upper += 0.05 * vol_factor

            return max(0.5, lower), min(2.0, upper)  # Hard limits

        def apply_kalman_update(predictor, predicted, actual, step):
            """Apply Kalman filter update to correction factor"""
            # global kalman_gain, error_variance

            # Skip if we don't have actual to compare
            if actual is None:
                return error_correction[predictor]

            # Calculate prediction error
            pred_error = (actual - predicted) / actual if predicted != 0 else 0

            # Update error variance estimate (simplified)
            error_variance[predictor] = 0.7 * error_variance[predictor] + 0.3 * (
                pred_error**2
            )

            # Update Kalman gain
            k_gain = error_variance[predictor] / (error_variance[predictor] + 0.1)
            kalman_gain[predictor] = min(0.5, max(0.05, k_gain))  # Bounded gain

            # Exponentially reduce gain with forecast horizon
            horizon_factor = np.exp(-0.1 * step)
            effective_gain = kalman_gain[predictor] * horizon_factor

            # Calculate correction factor
            correction = 1.0 + effective_gain * pred_error

            return correction

        def enforce_constraints(pred_values, step):
            """Enforce cross-variable constraints"""
            if all(p in predictors for p in ["Open", "High", "Low", "Close"]):
                # Get indices
                o_idx = predictor_indices["Open"]
                h_idx = predictor_indices["High"]
                l_idx = predictor_indices["Low"]
                c_idx = predictor_indices["Close"]

                # Ensure High is highest
                highest = max(
                    pred_values[step, o_idx],
                    pred_values[step, c_idx],
                    pred_values[step, h_idx],
                )
                pred_values[step, h_idx] = highest

                # Ensure Low is lowest
                lowest = min(
                    pred_values[step, o_idx],
                    pred_values[step, c_idx],
                    pred_values[step, l_idx],
                )
                pred_values[step, l_idx] = lowest

            return pred_values

        # Main forecasting loop
        for step in range(horizon):
            # Get last known dates
            if step == 0:
                # last_pred_row = prediction.iloc[-1]
                # last_backtest_row = backtest.iloc[-1]
                # last_pred_date = last_pred_row.name
                # last_backtest_date = last_backtest_row.name

                last_pred_row = (
                    prediction.iloc[-horizon:].mean(axis=0)
                    if len(prediction) >= horizon
                    else prediction.iloc[-1]
                )
                last_backtest_row = (
                    backtest.iloc[-horizon:].mean(axis=0)
                    if len(backtest) >= horizon
                    else backtest.iloc[-1]
                )
                last_pred_date = prediction.iloc[-1].name
                last_backtest_date = backtest.iloc[-1].name

                # last_pred_row = prediction.iloc[-horizon:,].mean( axis=0)
                # last_backtest_row = backtest.iloc[-horizon:,].mean(axis=0)
            else:
                last_pred_date = pred_dates[-1]
                last_backtest_date = backtest_dates[-1]

            # Calculate next dates
            next_pred_date = get_next_valid_date(pd.Timestamp(last_pred_date))
            next_backtest_date = get_next_valid_date(pd.Timestamp(last_backtest_date))
            pred_dates.append(next_pred_date)
            backtest_dates.append(next_backtest_date)

            # # Step 1: Update market regime if we have Close
            if "Close" in predictors and step > 0:
                # Get recent close values
                close_idx = predictor_indices["Close"]
                if step > 1:
                    recent_close_vals = pred_array[:step, close_idx]
                    regime = update_regime(recent_close_vals, None)

                    # Also track price changes for trending analysis
                    if step > 1:
                        price_changes.append(
                            pred_array[step - 1, close_idx]
                            - pred_array[step - 2, close_idx]
                        )

            # Step 2: First handle Close price prediction (which others depend on)
            if "Close" in predictors:
                close_idx = predictor_indices["Close"]
                close_features = [col for col in predictors if col != "Close"]

                # Prepare input data - use last available information
                if step == 0:
                    # pred_input = last_pred_row[close_features].values
                    # backtest_input = last_backtest_row[close_features].values
                    # raw_backtest_input = last_backtest_row[close_features].values

                    # Use averaged input from last 'horizon' rows
                    if len(prediction) >= horizon:
                        pred_input = (
                            prediction[close_features]
                            .iloc[-horizon:]
                            .mean(axis=0)
                            .values
                        )
                    else:
                        pred_input = last_pred_row[close_features].values

                    if len(backtest) >= horizon:
                        backtest_input = (
                            backtest[close_features].iloc[-horizon:].mean(axis=0).values
                        )
                        raw_backtest_input = (
                            backtest[close_features].iloc[-horizon:].mean(axis=0).values
                        )
                    else:
                        backtest_input = last_backtest_row[close_features].values
                        raw_backtest_input = last_backtest_row[close_features].values
                # else:
                # # Construct from previous predictions
                # pred_input = np.array([
                #     pred_array[step-1, predictor_indices[feat]]
                #     for feat in close_features
                # ])
                # backtest_input = np.array([
                #     backtest_array[step-1, predictor_indices[feat]]
                #     for feat in close_features
                # ])
                # raw_backtest_input = np.array([
                #     raw_backtest_array[step-1, predictor_indices[feat]]
                #     for feat in close_features
                # ])
                else:
                    # For subsequent steps, if we have enough predicted values, use their average
                    if step >= horizon:
                        # Use average of last 'horizon' predictions
                        pred_input = np.array(
                            [
                                np.mean(
                                    pred_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        raw_pred_input = np.array(
                            [
                                np.mean(
                                    raw_pred_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        backtest_input = np.array(
                            [
                                np.mean(
                                    backtest_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        raw_backtest_input = np.array(
                            [
                                np.mean(
                                    raw_backtest_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                    else:
                        # If we don't have enough predictions yet, combine historical and predicted
                        pred_inputs = []
                        raw_pred_inputs = []
                        backtest_inputs = []
                        raw_backtest_inputs = []

                        for feat in close_features:
                            feat_idx = predictor_indices[feat]

                            # Get predicted values so far
                            pred_vals = (
                                pred_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            raw_pred_vals = (
                                raw_pred_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            backtest_vals = (
                                backtest_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            raw_backtest_vals = (
                                raw_backtest_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )

                            # Calculate how many historical values we need
                            hist_needed = horizon - len(pred_vals)

                            if hist_needed > 0:
                                # Combine historical and predicted values
                                if feat in prediction.columns:
                                    pred_hist = (
                                        prediction[feat].iloc[-hist_needed:].values
                                    )
                                    all_pred_vals = np.concatenate(
                                        [pred_hist, pred_vals]
                                    )
                                    raw_all_pred_vals = np.concatenate(
                                        [pred_hist, raw_pred_vals]
                                    )
                                    pred_inputs.append(np.mean(all_pred_vals))
                                    raw_pred_inputs.append(np.mean(raw_all_pred_vals))
                                else:
                                    pred_inputs.append(0)  # Fallback

                                if feat in backtest.columns:
                                    backtest_hist = (
                                        backtest[feat].iloc[-hist_needed:].values
                                    )
                                    all_backtest_vals = np.concatenate(
                                        [backtest_hist, backtest_vals]
                                    )
                                    backtest_inputs.append(np.mean(all_backtest_vals))

                                    raw_backtest_hist = (
                                        backtest[feat].iloc[-hist_needed:].values
                                    )
                                    all_raw_backtest_vals = np.concatenate(
                                        [raw_backtest_hist, raw_backtest_vals]
                                    )
                                    raw_backtest_inputs.append(
                                        np.mean(all_raw_backtest_vals)
                                    )
                                else:
                                    backtest_inputs.append(0)  # Fallback
                                    raw_backtest_inputs.append(0)  # Fallback
                            else:
                                # We have enough predicted values already
                                pred_inputs.append(np.mean(pred_vals[-horizon:]))
                                backtest_inputs.append(
                                    np.mean(backtest_vals[-horizon:])
                                )
                                raw_backtest_inputs.append(
                                    np.mean(raw_backtest_vals[-horizon:])
                                )

                        pred_input = np.array(pred_inputs)
                        backtest_input = np.array(backtest_inputs)
                        raw_backtest_input = np.array(raw_backtest_inputs)

                # Apply model for Close price
                close_model = self.models["Close"][model_type]


                # Vector prediction for both datasets
                raw_pred_close = close_model.predict(pred_input.reshape(1, -1))[0]
                raw_backtest_close = close_model.predict(backtest_input.reshape(1, -1))[
                    0
                ]
                raw_backtest_raw_close = close_model.predict(
                    raw_backtest_input.reshape(1, -1)
                )[0]



                # Apply ensemble correction - weighted average of multiple correction factors
                ensemble_pred = 0
                ensemble_backtest = 0
                for i, corr in enumerate(ensemble_corrections["Close"]):
                    ensemble_pred += (
                        raw_pred_close * corr * ensemble_weights["Close"][i]
                    )
                    ensemble_backtest += (
                        raw_backtest_close * corr * ensemble_weights["Close"][i]
                    )

                # Apply the main error correction with adaptive bounds
                lower_bound, upper_bound = adaptive_bounds(
                    "Close", current_volatility, regime
                )
                close_correction = max(
                    lower_bound, min(upper_bound, error_correction["Close"])
                )

                pred_close = ensemble_pred * close_correction
                backtest_close = ensemble_backtest * close_correction

                # test if first prediction is way off
                if (
                    step == 0
                    and abs(
                        1 - backtest_close / self.data.copy().iloc[-horizon]["Close"]
                    )
                    >= 0.075
                ):
                    pred_close = 0.5 * (self.data.copy().iloc[-1]["Close"] + pred_close)
                    backtest_close = 0.5 * (
                        self.data.copy().iloc[-horizon]["Close"] + backtest_close
                    )
                # Store predictions
                pred_array[step, close_idx] = pred_close
                raw_pred_array[step, close_idx] = raw_pred_close
                backtest_array[step, close_idx] = backtest_close
                raw_backtest_array[step, close_idx] = raw_backtest_raw_close

                # Store predictions v2 mirror original code
                pred_array[step, close_idx] = raw_pred_close
                raw_pred_array[step, close_idx] = raw_pred_close
                backtest_array[step, close_idx] = raw_backtest_close
                raw_backtest_array[step, close_idx] = raw_backtest_raw_close

                # Update volatility estimate
                if step > 0:
                    prev_close = pred_array[step - 1, close_idx]
                    returns = (pred_close / prev_close) - 1
                    current_volatility = 0.94 * current_volatility + 0.06 * abs(
                        returns
                    ) * np.sqrt(252)

            # Step 3: Now handle other predictors
            for predictor in predictors:

                if predictor == "Close":
                    continue  # Already handled

                pred_idx = predictor_indices[predictor]

                # Special handling for MA calculations - direct calculation rather than model
                if predictor == "MA_50" and "Close" in predictors:
                    close_idx = predictor_indices["Close"]

                    # Get recent Close values to calculate MA
                    if step == 0:
                        # Use historical data for initial MA calculation
                        hist_close_pred = observation["Close"].values[-49:]
                        hist_close_backtest = backtest["Close"].values[-49:]
                        hist_close_raw_backtest = backtest["Close"].values[-49:]
                    else:
                        # Combine historical with predicted for later steps
                        pred_close_history = pred_array[:step, close_idx]
                        raw_pred_close_history = raw_pred_array[:step, close_idx]
                        backtest_close_history = backtest_array[:step, close_idx]
                        raw_backtest_close_history = raw_backtest_array[
                            :step, close_idx
                        ]

                        # Concatenate with appropriate historical data
                        if len(pred_close_history) < 49:
                            hist_close_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(49 - len(pred_close_history)) :
                                    ],
                                    pred_close_history,
                                ]
                            )
                            hist_close_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(49 - len(backtest_close_history)) :
                                    ],
                                    backtest_close_history,
                                ]
                            )
                            hist_close_raw_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(49 - len(raw_backtest_close_history)) :
                                    ],
                                    raw_backtest_close_history,
                                ]
                            )
                        else:
                            hist_close_pred = pred_close_history[-49:]
                            hist_close_backtest = backtest_close_history[-49:]
                            hist_close_raw_backtest = raw_backtest_close_history[-49:]

                    # Get current Close predictions
                    current_pred_close = pred_array[step, close_idx]
                    current_raw_pred_close = raw_pred_array[step, close_idx]
                    current_backtest_close = backtest_array[step, close_idx]
                    current_raw_close = raw_backtest_array[step, close_idx]

                    # Calculate MA_50 (vectorized)
                    ma50_pred = np.mean(np.append(hist_close_pred, current_pred_close))
                    ma50_raw_pred = np.mean(
                        np.append(hist_close_pred, current_raw_pred_close)
                    )
                    ma50_backtest = np.mean(
                        np.append(hist_close_backtest, current_backtest_close)
                    )
                    ma50_raw_backtest = np.mean(
                        np.append(hist_close_raw_backtest, current_raw_close)
                    )

                    # Store MA_50 values
                    pred_array[step, pred_idx] = ma50_pred
                    raw_pred_array[step, pred_idx] = ma50_raw_pred
                    backtest_array[step, pred_idx] = ma50_backtest
                    raw_backtest_array[step, pred_idx] = ma50_raw_backtest

                elif predictor == "MA_200" and "Close" in predictors:
                    close_idx = predictor_indices["Close"]

                    # Similar approach for MA_200
                    if step == 0:
                        hist_close_pred = observation["Close"].values[-199:]
                        hist_close_raw_pred = observation["Close"].values[-199:]
                        hist_close_backtest = backtest["Close"].values[-199:]
                        hist_close_raw_backtest = backtest["Close"].values[-199:]
                    else:
                        pred_close_history = pred_array[:step, close_idx]
                        raw_pred_close_history = raw_pred_array[:step, close_idx]
                        backtest_close_history = backtest_array[:step, close_idx]
                        raw_backtest_close_history = raw_backtest_array[
                            :step, close_idx
                        ]

                        if len(pred_close_history) < 199:
                            hist_close_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(199 - len(pred_close_history)) :
                                    ],
                                    pred_close_history,
                                ]
                            )
                            hist_close_raw_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(199 - len(raw_pred_close_history)) :
                                    ],
                                    raw_pred_close_history,
                                ]
                            )
                            hist_close_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(199 - len(backtest_close_history)) :
                                    ],
                                    backtest_close_history,
                                ]
                            )
                            hist_close_raw_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(199 - len(raw_backtest_close_history)) :
                                    ],
                                    raw_backtest_close_history,
                                ]
                            )
                        else:
                            hist_close_pred = pred_close_history[-199:]
                            hist_close_raw_pred = raw_pred_close_history[-199:]
                            hist_close_backtest = backtest_close_history[-199:]
                            hist_close_raw_backtest = raw_backtest_close_history[-199:]

                    current_pred_close = pred_array[step, close_idx]
                    current_raw_pred_close = raw_pred_array[step, close_idx]
                    current_backtest_close = backtest_array[step, close_idx]
                    current_raw_backtest_close = raw_backtest_array[step, close_idx]

                    ma200_pred = np.mean(np.append(hist_close_pred, current_pred_close))
                    ma200_raw_pred = np.mean(
                        np.append(hist_close_raw_pred, current_raw_pred_close)
                    )
                    ma200_backtest = np.mean(
                        np.append(hist_close_backtest, current_backtest_close)
                    )
                    ma200_raw_backtest = np.mean(
                        np.append(hist_close_raw_backtest, current_raw_backtest_close)
                    )

                    pred_array[step, pred_idx] = ma200_pred
                    raw_pred_array[step, pred_idx] = ma200_raw_pred
                    backtest_array[step, pred_idx] = ma200_backtest
                    raw_backtest_array[step, pred_idx] = ma200_raw_backtest

                # elif predictor == "Daily Returns" and "Close" in predictors:
                #     close_idx = predictor_indices["Close"]

                #     # Calculate daily returns from consecutive Close prices
                #     if step == 0:
                #         # Get last actual Close from data
                #         prev_close_pred = observation["Close"].values[-1]
                #         prev_close_raw_backtest = observation["Close"].values[-1]
                #         prev_close_backtest = backtest["Close"].values[-1]
                #         prev_close_raw_backtest = backtest["Close"].values[-1]
                #     else:
                #         # Use previous predicted Close
                #         prev_close_pred = pred_array[step - 1, close_idx]
                #         prev_close_raw_pred = raw_pred_array[step - 1, close_idx]
                #         prev_close_backtest = backtest_array[step - 1, close_idx]
                #         prev_close_raw_backtest = raw_backtest_array[
                #             step - 1, close_idx
                #         ]

                #     # Get current predicted Close
                #     current_close_pred = pred_array[step, close_idx]
                #     current_close_raw_pred = raw_pred_array[step, close_idx]
                #     current_close_backtest = backtest_array[step, close_idx]
                #     current_close_raw_backtest = raw_backtest_array[step, close_idx]

                #     # Calculate returns (handle division by zero)
                #     if prev_close_pred != 0:
                #         daily_return_pred = (current_close_pred / prev_close_pred) - 1
                #     else:
                #         daily_return_pred = 0

                #     if prev_close_backtest != 0:
                #         daily_return_backtest = (
                #             current_close_backtest / prev_close_backtest
                #         ) - 1
                #     else:
                #         daily_return_backtest = 0

                #     if prev_close_raw_backtest != 0:
                #         daily_return_raw_backtest = (
                #             current_close_raw_backtest / prev_close_raw_backtest
                #         ) - 1
                #     else:
                #         daily_return_raw_backtest = 0

                #     if prev_close_raw_pred != 0:
                #         daily_return_raw_pred = (
                #             current_close_raw_pred / prev_close_raw_pred
                #         ) - 1
                #     else:
                #         daily_return_raw_pred = 0

                #     # Store daily returns
                #     pred_array[step, pred_idx] = daily_return_pred
                #     raw_pred_array[step, pred_idx] = daily_return_raw_pred
                #     backtest_array[step, pred_idx] = daily_return_backtest
                #     raw_backtest_array[step, pred_idx] = daily_return_raw_backtest

                elif predictor == "VIX" and "Close" in predictors:
                    # Use current volatility estimate directly
                    pred_array[step, pred_idx] = current_volatility
                    raw_pred_array[step, pred_idx] = current_volatility
                    backtest_array[step, pred_idx] = current_volatility
                    raw_backtest_array[step, pred_idx] = current_volatility

                else:
                    # Regular predictor - use model
                    features = [col for col in predictors if col != predictor]

                    # Prepare input data using moving average approach
                    if step == 0:
                        # Use averaged input from last 'horizon' rows
                        if len(prediction) >= horizon:
                            pred_input = (
                                prediction[features].iloc[-horizon:].mean(axis=0).values
                            )
                        else:
                            pred_input = last_pred_row[features].values

                        if len(backtest) >= horizon:
                            backtest_input = (
                                backtest[features].iloc[-horizon:].mean(axis=0).values
                            )
                        else:
                            backtest_input = last_backtest_row[features].values
                    else:
                        # For subsequent steps, similar approach as Close prediction
                        if step >= horizon:
                            # Use average of last 'horizon' predictions
                            pred_input = np.array(
                                [
                                    np.mean(
                                        pred_array[
                                            max(0, step - horizon) : step,
                                            predictor_indices[feat],
                                        ]
                                    )
                                    for feat in features
                                ]
                            )
                            backtest_input = np.array(
                                [
                                    np.mean(
                                        backtest_array[
                                            max(0, step - horizon) : step,
                                            predictor_indices[feat],
                                        ]
                                    )
                                    for feat in features
                                ]
                            )
                        else:
                            # If we don't have enough predictions yet, combine historical and predicted
                            pred_inputs = []
                            backtest_inputs = []

                            for feat in features:
                                feat_idx = predictor_indices[feat]

                                # Get predicted values so far
                                pred_vals = (
                                    pred_array[:step, feat_idx]
                                    if step > 0
                                    else np.array([])
                                )
                                backtest_vals = (
                                    backtest_array[:step, feat_idx]
                                    if step > 0
                                    else np.array([])
                                )

                                # Calculate how many historical values we need
                                hist_needed = horizon - len(pred_vals)

                                if hist_needed > 0:
                                    # Combine historical and predicted values
                                    if feat in prediction.columns:
                                        pred_hist = (
                                            prediction[feat].iloc[-hist_needed:].values
                                        )
                                        all_pred_vals = np.concatenate(
                                            [pred_hist, pred_vals]
                                        )
                                        pred_inputs.append(np.mean(all_pred_vals))
                                    else:
                                        pred_inputs.append(0)  # Fallback

                                    if feat in backtest.columns:
                                        backtest_hist = (
                                            backtest[feat].iloc[-hist_needed:].values
                                        )
                                        all_backtest_vals = np.concatenate(
                                            [backtest_hist, backtest_vals]
                                        )
                                        backtest_inputs.append(
                                            np.mean(all_backtest_vals)
                                        )
                                    else:
                                        backtest_inputs.append(0)  # Fallback
                                else:
                                    # We have enough predicted values already
                                    pred_inputs.append(np.mean(pred_vals[-horizon:]))
                                    backtest_inputs.append(
                                        np.mean(backtest_vals[-horizon:])
                                    )

                            pred_input = np.array(pred_inputs)
                            backtest_input = np.array(backtest_inputs)

                    # Get model predictions
                    model = self.models[predictor][model_type]


                    
                    raw_pred = model.predict(pred_input.reshape(1, -1))[0]
                    raw_backtest = model.predict(backtest_input.reshape(1, -1))[0]

                    # Apply adaptive correction
                    lower_bound, upper_bound = adaptive_bounds(
                        predictor, current_volatility, regime
                    )
                    predictor_correction = max(
                        lower_bound, min(upper_bound, error_correction[predictor])
                    )

                    # Apply Kalman filter update for backtest
                    # (we can compare backtest with actual historical data)
                    actual_value = None
                    if (
                        next_backtest_date in self.data.index
                        and predictor in self.data.columns
                    ):
                        # actual_value = self.data.loc[next_backtest_date, predictor]
                        actual_value = self.data[self.data.index == next_backtest_date][
                            predictor
                        ].values[0]
                        kalman_correction = apply_kalman_update(
                            predictor, raw_backtest, actual_value, step
                        )
                        # Update the main correction factor with the Kalman result
                        error_correction[predictor] = (
                            0.7 * error_correction[predictor] + 0.3 * kalman_correction
                        )

                    # Apply correction
                    pred_value = raw_pred * predictor_correction
                    backtest_value = raw_backtest * predictor_correction
                    raw_backtest_value = raw_backtest
                    raw_pred_value = raw_pred

                    # # Store predictions
                    pred_array[step, pred_idx] = pred_value
                    raw_pred_array[step, pred_idx] = raw_pred_value
                    backtest_array[step, pred_idx] = backtest_value
                    raw_backtest_array[step, pred_idx] = raw_backtest_value

                    # Store predictions v2 mirror original code
                    # pred_array[step, pred_idx] = raw_pred
                    # backtest_array[step, pred_idx] = raw_backtest
                    # raw_backtest_array[step, pred_idx] = raw_backtest

            # Step 4: Apply cross-variable constraints
            pred_array = enforce_constraints(pred_array, step)
            backtest_array = enforce_constraints(backtest_array, step)

            # # Step 5: Update ensemble weights based on performance (for backtest)
            if step > 0 and step % 5 == 0:
                for predictor in predictors:
                    # Skip if we don't have enough data
                    if len(pred_dates) < 5:
                        continue

                    pred_idx = predictor_indices[predictor]

                    # Check if we have actual data to compare with backtest
                    actual_values = []
                    for date in backtest_dates[-5:]:
                        if date in self.data.index and predictor in self.data.columns:
                            actual_values.append(self.data.loc[date, predictor])

                    if len(actual_values) >= 3:  # Need enough data points
                        # Calculate errors for each ensemble member
                        errors = []
                        for i, corr in enumerate(ensemble_corrections[predictor]):
                            # Get predictions with this correction factor
                            corrected_preds = (
                                backtest_array[-len(actual_values) :, pred_idx] * corr
                            )

                            # Calculate mean squared error
                            mse = np.mean((corrected_preds - actual_values) ** 2)
                            errors.append(mse)

                        # Convert errors to weights (smaller error -> higher weight)
                        if max(errors) > min(errors):  # Avoid division by zero
                            inv_errors = 1.0 / (np.array(errors) + 1e-10)
                            new_weights = inv_errors / sum(inv_errors)

                            # Update weights with smoothing
                            ensemble_weights[predictor] = (
                                0.7 * ensemble_weights[predictor] + 0.3 * new_weights
                            )

        # Convert arrays to DataFrames
        prediction_df = pd.DataFrame(pred_array, columns=predictors, index=pred_dates)

        backtest_df = pd.DataFrame(
            backtest_array, columns=predictors, index=backtest_dates
        )

        raw_backtest_df = pd.DataFrame(
            raw_backtest_array, columns=predictors, index=backtest_dates
        )

        raw_prediction_df = pd.DataFrame(
            raw_pred_array, columns=predictors, index=pred_dates
        )

        # Concatenate with original data to include history
        final_prediction = pd.concat([prediction, prediction_df])
        final_raw_prediction = pd.concat([prediction, raw_prediction_df])
        final_backtest = pd.concat([backtest, backtest_df])
        final_raw_backtest = pd.concat([backtest, raw_backtest_df])

        return (
            final_prediction,
            final_backtest,
            final_raw_prediction,
            final_raw_backtest,
        )

    # CLD V1
    # def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
    #     """
    #     Perform one-step forward predictions for all predictors.

    #     Parameters:
    #     -----------
    #     predictors : List[str]
    #         List of predictor column names
    #     model_type : str
    #         one of the model types
    #     horizon : int
    #         Number of days to forecast

    #     Returns:
    #     --------
    #     Tuple[pd.DataFrame, pd.DataFrame]
    #         Forecasted data and backtest data
    #     """
    #     # Ensure models are prepared
    #     if not self.models:
    #         raise ValueError("Please run prepare_models() first")

    #     # Initialize prediction and backtest DataFrames
    #     prediction = self.data[predictors].copy().iloc[-horizon:].dropna()
    #     backtest = self.data[predictors].copy().iloc[:-horizon].dropna()
    #     observation = self.data[predictors].copy().dropna()

    #     # Initialize numpy arrays for vectorized operations
    #     pred_values = np.zeros((horizon, len(predictors)))
    #     pred_dates = []
    #     backtest_values = np.zeros((horizon, len(predictors)))
    #     backtest_dates = []

    #     # Error correction factors (initialized to 1.0 - no correction)
    #     error_correction = {predictor: 1.0 for predictor in predictors}

    #     # Create mapping from predictor names to column indices
    #     predictor_to_idx = {predictor: i for i, predictor in enumerate(predictors)}

    #     for step in range(horizon):
    #         # Get the latest available data
    #         if step == 0:
    #             last_pred_row = prediction.iloc[-1]
    #             last_backtest_row = backtest.iloc[-1]
    #             last_pred_date = last_pred_row.name
    #             last_backtest_date = last_backtest_row.name
    #         else:
    #             last_pred_date = pred_dates[-1]
    #             last_backtest_date = backtest_dates[-1]

    #             # Create temporary DataFrames for previous predictions
    #             temp_pred_df = pd.DataFrame(
    #                 pred_values[:step],
    #                 columns=predictors,
    #                 index=pred_dates
    #             )
    #             temp_backtest_df = pd.DataFrame(
    #                 backtest_values[:step],
    #                 columns=predictors,
    #                 index=backtest_dates
    #             )

    #             # Combine with original data
    #             combined_pred = pd.concat([prediction, temp_pred_df])
    #             combined_backtest = pd.concat([backtest, temp_backtest_df])

    #             # Get the latest rows
    #             last_pred_row = combined_pred.iloc[-1]
    #             last_backtest_row = combined_backtest.iloc[-1]

    #         # Calculate next dates
    #         next_pred_date = get_next_valid_date(pd.Timestamp(last_pred_date))
    #         next_backtest_date = get_next_valid_date(pd.Timestamp(last_backtest_date))
    #         pred_dates.append(next_pred_date)
    #         backtest_dates.append(next_backtest_date)

    #         # First predict Close (if it's in the predictors list)
    #         if "Close" in predictors:
    #             close_idx = predictor_to_idx["Close"]
    #             close_features = [col for col in predictors if col != "Close"]

    #             # Prepare input data for Close prediction
    #             if step == 0:
    #                 # Use original data
    #                 pred_features = last_pred_row[close_features].values
    #                 backtest_features = last_backtest_row[close_features].values
    #             else:
    #                 # Use combination of original and predicted features
    #                 pred_features = np.array([
    #                     pred_values[step-1, predictor_to_idx[feat]] if step > 0 else last_pred_row[feat]
    #                     for feat in close_features
    #                 ])
    #                 backtest_features = np.array([
    #                     backtest_values[step-1, predictor_to_idx[feat]] if step > 0 else last_backtest_row[feat]
    #                     for feat in close_features
    #                 ])

    #             # Make predictions for Close
    #             close_model = self.models["Close"][model_type]
    #             pred_close = close_model.predict(pred_features.reshape(1, -1))[0]
    #             backtest_close = close_model.predict(backtest_features.reshape(1, -1))[0]

    #             # Apply error correction
    #             pred_close *= error_correction["Close"]
    #             backtest_close *= error_correction["Close"]

    #             # Store Close predictions
    #             pred_values[step, close_idx] = pred_close
    #             backtest_values[step, close_idx] = backtest_close

    #         # Then predict other variables
    #         for predictor in predictors:
    #             if predictor == "Close":
    #                 continue  # Already handled

    #             predictor_idx = predictor_to_idx[predictor]

    #             # Special handling for MA_50 and MA_200
    #             if predictor == "MA_50" and "Close" in predictors:
    #                 close_idx = predictor_to_idx["Close"]

    #                 # Get historical Close values
    #                 if step == 0:
    #                     # Use only historical data
    #                     hist_close_pred = observation["Close"].values[-49:]
    #                     hist_close_backtest = backtest["Close"].values[-49:]
    #                 else:
    #                     # Combine historical with predicted values
    #                     pred_close_values = pred_values[:step, close_idx]
    #                     backtest_close_values = backtest_values[:step, close_idx]

    #                     hist_close_pred = np.concatenate([
    #                         observation["Close"].values[-(50-step):] if 50-step > 0 else [],
    #                         pred_close_values
    #                     ])
    #                     hist_close_backtest = np.concatenate([
    #                         backtest["Close"].values[-(50-step):] if 50-step > 0 else [],
    #                         backtest_close_values
    #                     ])

    #                 # Add current predicted Close
    #                 pred_close = pred_values[step, close_idx]
    #                 backtest_close = backtest_values[step, close_idx]

    #                 # Calculate MA_50 with vectorized operations
    #                 ma50_pred = np.mean(np.append(hist_close_pred, pred_close)[-50:])
    #                 ma50_backtest = np.mean(np.append(hist_close_backtest, backtest_close)[-50:])

    #                 # Store MA_50 predictions
    #                 pred_values[step, predictor_idx] = ma50_pred
    #                 backtest_values[step, predictor_idx] = ma50_backtest

    #             elif predictor == "MA_200" and "Close" in predictors:
    #                 close_idx = predictor_to_idx["Close"]

    #                 # Similar approach for MA_200
    #                 if step == 0:
    #                     hist_close_pred = observation["Close"].values[-199:]
    #                     hist_close_backtest = backtest["Close"].values[-199:]
    #                 else:
    #                     pred_close_values = pred_values[:step, close_idx]
    #                     backtest_close_values = backtest_values[:step, close_idx]

    #                     hist_close_pred = np.concatenate([
    #                         observation["Close"].values[-(200-step):] if 200-step > 0 else [],
    #                         pred_close_values
    #                     ])
    #                     hist_close_backtest = np.concatenate([
    #                         backtest["Close"].values[-(200-step):] if 200-step > 0 else [],
    #                         backtest_close_values
    #                     ])

    #                 pred_close = pred_values[step, close_idx]
    #                 backtest_close = backtest_values[step, close_idx]

    #                 # Calculate MA_200 with vectorized operations
    #                 ma200_pred = np.mean(np.append(hist_close_pred, pred_close)[-200:])
    #                 ma200_backtest = np.mean(np.append(hist_close_backtest, backtest_close)[-200:])

    #                 pred_values[step, predictor_idx] = ma200_pred
    #                 backtest_values[step, predictor_idx] = ma200_backtest

    #             else:
    #                 # For other predictors, use the model
    #                 features = [col for col in predictors if col != predictor]

    #                 # Prepare input features
    #                 if step == 0:
    #                     # Get features from the last row
    #                     pred_features = last_pred_row[features].values
    #                     backtest_features = last_backtest_row[features].values
    #                 else:
    #                     # Get features from previous predictions
    #                     pred_features = np.array([
    #                         pred_values[step-1, predictor_to_idx[feat]]
    #                         for feat in features
    #                     ])
    #                     backtest_features = np.array([
    #                         backtest_values[step-1, predictor_to_idx[feat]]
    #                         for feat in features
    #                     ])

    #                 # Make predictions
    #                 model = self.models[predictor][model_type]
    #                 pred_value = model.predict(pred_features.reshape(1, -1))[0]
    #                 backtest_value = model.predict(backtest_features.reshape(1, -1))[0]

    #                 # Apply error correction
    #                 pred_value *= error_correction[predictor]
    #                 backtest_value *= error_correction[predictor]

    #                 # Store predictions
    #                 pred_values[step, predictor_idx] = pred_value
    #                 backtest_values[step, predictor_idx] = backtest_value

    #         # Apply business logic constraints
    #         if "Low" in predictors and "Close" in predictors:
    #             low_idx = predictor_to_idx["Low"]
    #             close_idx = predictor_to_idx["Close"]

    #             # Ensure Close is not less than Low
    #             if pred_values[step, close_idx] < pred_values[step, low_idx]:
    #                 pred_values[step, close_idx] = pred_values[step, low_idx]
    #             if backtest_values[step, close_idx] < backtest_values[step, low_idx]:
    #                 backtest_values[step, close_idx] = backtest_values[step, low_idx]

    #         # Error correction feedback loop
    #         # If we have actual data for comparison (e.g., for backtest)
    #         # and we're past the first few steps, update the correction factors
    #         if step > 0 and step % 5 == 0:  # Update every 5 steps
    #             for predictor in predictors:
    #                 pred_idx = predictor_to_idx[predictor]

    #                 # Compare predicted vs actual for previous steps where actual data is available
    #                 available_history = min(step, 5)  # Use up to 5 previous points

    #                 # For backtest, we can compare with actual historical data
    #                 if step < len(backtest):
    #                     actual_values = backtest[predictor].values[-available_history:]
    #                     predicted_values = backtest_values[step-available_history:step, pred_idx]

    #                     # Calculate ratio of actual to predicted (as correction factor)
    #                     if np.all(predicted_values != 0):
    #                         ratio = np.mean(actual_values / predicted_values)
    #                         # Limit correction factor to reasonable range
    #                         ratio = max(0.85, min(1.15, ratio))
    #                         # Smooth the correction factor update
    #                         error_correction[predictor] = 0.75 * error_correction[predictor] + 0.25 * ratio

    #     # Convert predictions to DataFrames
    #     prediction_result = pd.DataFrame(
    #         pred_values,
    #         columns=predictors,
    #         index=pred_dates
    #     )

    #     backtest_result = pd.DataFrame(
    #         backtest_values,
    #         columns=predictors,
    #         index=backtest_dates
    #     )

    #     # Include the original data
    #     prediction = pd.concat([prediction, prediction_result])
    #     backtest = pd.concat([backtest, backtest_result])

    #     return prediction, backtest

    # # DS Code
    # def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
    #     """Maintains original interface but uses loop for predictions"""
    #     # Initialize containers as per original structure
    #     prediction = self.data[predictors].iloc[-horizon:].copy()
    #     backtest = self.data[predictors].iloc[:-horizon].copy()
    #     observation = self.data[predictors].copy().dropna()

    #     # Maintain original date handling
    #     last_real_date = self.data.index[-1]
    #     dates = []
    #     current_date = last_real_date

    #     # For-loop prediction core
    #     forecast_values = []
    #     for _ in range(horizon):
    #         # Get next valid business day
    #         current_date = get_next_valid_date(current_date + timedelta(days=1))
    #         dates.append(current_date)

    #         # Prepare input with time feature
    #         input_data = self.data[predictors].drop(columns='Close').iloc[_].values.reshape(1, -1)
    #         # np.hstack([
    #         #     self.data[predictors].iloc[-1].values,
    #         #     [len(forecast_values) + 1]  # Time step feature
    #         # ]).reshape(1, -1)

    #         # Single-step prediction
    #         try:
    #             pred = self.models[predictors[0]][model_type].predict(input_data)[0]
    #         except:
    #             pred = self.data[predictors[0]].iloc[-1]

    #         forecast_values.append(pred)

    #         # Update data with prediction (for recursive models)
    #         self.data.loc[current_date] = {**self.data.iloc[-1].to_dict(),
    #                                     predictors[0]: pred}

    #     # Build prediction DataFrame (preserve original format)
    #     prediction = pd.DataFrame(
    #         np.array(forecast_values).reshape(-1, 1),
    #         index=dates,
    #         columns=[predictors[0]]
    #     )

    #     # Backtest loop (original structure)
    #     backtest_preds = []
    #     for i in range(len(backtest), len(backtest)+horizon):
    #         if i >= len(self.data): break
    #         # input_data = np.hstack([
    #         #     self.data[predictors].iloc[i].values,
    #         #     [i - len(backtest) + 1]
    #         # ]).reshape(1, -1)
    #         input_data = self.data[predictors].drop(columns='Close').iloc[i].values.reshape(1, -1)
    #         backtest_preds.append(self.models[predictors[0]][model_type].predict(input_data)[0])

    #     # Maintain original backtest format
    #     backtest = backtest.combine_first(
    #         pd.DataFrame(
    #             np.array(backtest_preds).reshape(-1, 1),
    #             index=self.data.index[-horizon:],
    #             columns=[predictors[0]]
    #         )
    #     )

    #     # Original error correction logic
    #     if len(self.history) >= 5:
    #         recent_errors = [
    #             observation[predictors[0]].iloc[-i-1] - self.history[-i-1]
    #             for i in range(1, 6)
    #         ]
    #         error_bias = np.mean(recent_errors)
    #         prediction.iloc[:, 0] += error_bias * 0.3
    #         backtest.iloc[-horizon:] += error_bias * 0.3

    #     self.history.extend(prediction[predictors[0]].values)

    #     return prediction, backtest

    # # Original Code
    # def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
    #     """
    #     Perform one-step forward predictions for all predictors.

    #     Parameters:
    #     -----------
    #     predictors : List[str]
    #         List of predictor column names
    #     model_type : str
    #         one of the
    #     horizon : int, optional
    #         Number of days to forecast (default: 20)

    #     Returns:
    #     --------
    #     Tuple[pd.DataFrame, pd.DataFrame]z
    #         Forecasted data and backtest data
    #     """
    #     # Ensure models are prepared
    #     if not self.models:
    #         raise ValueError("Please run prepare_models() first")

    #     # Initialize prediction and backtest DataFrames
    #     prediction = self.data[predictors].copy().iloc[-horizon:,].dropna()
    #     backtest = self.data[predictors].copy().iloc[:-horizon,].dropna()
    #     observation = self.data[predictors].copy().dropna()

    #     # last_row = prediction.iloc[-1]
    #     # last_row1 = backtest.iloc[-1]

    #     # Dictionaries to store predictions for each predictor and model
    #     predictions = {
    #         predictor: {
    #             model: []
    #             for model in ["ridge", "linear", "polynomial", "arimaxgb", "combined"]
    #         }
    #         for predictor in predictors
    #     }

    #     for _ in range(horizon):
    #         # Predict each predictor using different models
    #         last_row = prediction.iloc[-horizon:,]
    #         last_row1 = backtest.iloc[-horizon:,]

    #         for predictor in predictors:
    #             # Select features excluding the current predictor
    #             features = [col for col in predictors if col != predictor]

    #             # Predictions using different models
    #             models = self.models[predictor]
    #             scaler = self.scalers[predictor]
    #             transformer = self.transformers[predictor]

    #             # Calculate rolling averages for MA_50 and MA_200 based on the Close price
    #             if predictor == "MA_50":
    #                 rolling_avg_50 = (
    #                     pd.Series(
    #                         np.concatenate(
    #                             (
    #                                 observation["Close"],
    #                                 predictions["Close"]["arimaxgb"][-2],
    #                             )
    #                         )
    #                     )
    #                     .rolling(window=50)
    #                     .mean()
    #                     .iloc[-1]
    #                 )
    #                 predictions[predictor]["arimaxgb"].append(rolling_avg_50)

    #                 rolling_avg_50_1 = (
    #                     pd.Series(
    #                         np.concatenate(
    #                             (
    #                                 backtest["Close"],
    #                                 predictions["Close"]["arimaxgb"][-1],
    #                             )
    #                         )
    #                     )
    #                     .rolling(window=50)
    #                     .mean()
    #                     .iloc[-1]
    #                 )
    #                 predictions[predictor]["arimaxgb"].append(rolling_avg_50_1)
    #             elif predictor == "MA_200":
    #                 rolling_avg_200 = (
    #                     pd.Series(
    #                         np.concatenate(
    #                             (
    #                                 observation["Close"],
    #                                 predictions["Close"]["arimaxgb"][-2],
    #                             )
    #                         )
    #                     )
    #                     .rolling(window=200)
    #                     .mean()
    #                     .iloc[-1]
    #                 )
    #                 predictions[predictor]["arimaxgb"].append(rolling_avg_200)

    #                 rolling_avg_200_1 = (
    #                     pd.Series(
    #                         np.concatenate(
    #                             (
    #                                 backtest["Close"],
    #                                 predictions["Close"]["arimaxgb"][-1],
    #                             )
    #                         )
    #                     )
    #                     .rolling(window=200)
    #                     .mean()
    #                     .iloc[-1]
    #                 )
    #                 predictions[predictor]["arimaxgb"].append(rolling_avg_200_1)

    #             else:

    #                 average_row = np.average(
    #                     last_row[features],
    #                     axis=0,
    #                     weights=[0.1 * i for i in range(1, len(last_row) + 1)],
    #                 )
    #                 average_row1 = np.average(
    #                     last_row1[features],
    #                     axis=0,
    #                     weights=[0.1 * i for i in range(1, len(last_row1) + 1)],
    #                 )

    #                 # Prepare input data
    #                 input_data = average_row
    #                 input_data1 = average_row1

    #                 # Make predictions

    #                 pred_arima_ml = models["arimaxgb"].predict(
    #                     input_data.reshape(1, -1)
    #                 )
    #                 # pred_arima_ml = models["arimaxgb"].predict(input_data.values.reshape(1, -1))
    #                 # pred_arima_ml = models["arimaxgb"].predict(input_data)

    #                 # pred_ridge = models['ridge'].predict(scaler.transform(input_data.values.reshape(1, -1)))[0]
    #                 # pred_linear = models['linear'].predict(input_data.values.reshape(1, -1))[0]
    #                 # pred_poly = models['polynomial'].predict(
    #                 #     transformer.transform(scaler.transform(input_data.values.reshape(1, -1)))
    #                 # )[0]
    #                 # if predictor not in ['Daily Returns', 'Volatility']:
    #                 #     pred_combined = models["arimaxgb"].predict(input_data.values.reshape(1, -1))
    #                 # else:
    #                 #     pred_combined =  models['linear'].predict(input_data.values.reshape(1, -1))[0]

    #                 pred_arima_ml1 = models["arimaxgb"].predict(
    #                     input_data1.reshape(1, -1)
    #                 )
    #                 # pred_arima_ml1 = models["arimaxgb"].predict(input_data1.values.reshape(1, -1))
    #                 # pred_arima_ml1 = models["arimaxgb"].predict(input_data1)

    #                 # pred_ridge1 = models['ridge'].predict(scaler.transform(input_data1.values.reshape(1, -1)))[0]
    #                 # pred_linear1 = models['linear'].predict(input_data1.values.reshape(1, -1))[0]
    #                 # pred_poly1 = models['polynomial'].predict(
    #                 #     transformer.transform(scaler.transform(input_data1.values.reshape(1, -1)))
    #                 # )[0]
    #                 # if predictor not in ['Daily Returns', 'Volatility']:
    #                 #     pred_combined1 = models["arimaxgb"].predict(input_data1.values.reshape(1, -1))
    #                 # else:
    #                 #     pred_combined1 =  models['linear'].predict(input_data1.values.reshape(1, -1))[0]

    #                 # Store predictions
    #                 # predictions[predictor]['ridge'].append(pred_ridge)
    #                 # predictions[predictor]['linear'].append(pred_linear)
    #                 # predictions[predictor]['polynomial'].append(pred_poly)
    #                 predictions[predictor]["arimaxgb"].append(pred_arima_ml)
    #                 # predictions[predictor]['combined'].append(pred_combined)

    #                 # predictions[predictor]['ridge'].append(pred_ridge1)
    #                 # predictions[predictor]['linear'].append(pred_linear1)
    #                 # predictions[predictor]['polynomial'].append(pred_poly1)
    #                 predictions[predictor]["arimaxgb"].append(pred_arima_ml1)
    #             # predictions[predictor]['combined'].append(pred_combined1)

    #         # Create new row with predictions
    #         next_day_date = get_next_valid_date(pd.Timestamp(last_row.iloc[-1].name))
    #         next_day_date1 = get_next_valid_date(pd.Timestamp(last_row1.iloc[-1].name))
    #         if "Low" in predictors:
    #             if predictor == "Close":
    #                 if (
    #                     predictions["Close"][model_type][-2]
    #                     > predictions["Low"][model_type][-2]
    #                 ):
    #                     predictions["Close"][model_type][-2] = predictions["Low"][
    #                         model_type
    #                     ][-2]
    #                 elif (
    #                     predictions["Close"][model_type][-1]
    #                     > predictions["Low"][model_type][-1]
    #                 ):
    #                     predictions["Close"][model_type][-1] = predictions["Low"][
    #                         model_type
    #                     ][-1]

    #         new_row = pd.DataFrame(
    #             {
    #                 predictor: predictions[predictor][model_type][-2]
    #                 for predictor in predictors
    #             },  # take the last one
    #             index=[next_day_date],
    #         )
    #         new_row1 = pd.DataFrame(
    #             {
    #                 predictor: predictions[predictor][model_type][-1]
    #                 for predictor in predictors
    #             },  # take the last one
    #             index=[next_day_date1],
    #         )

    #         # Update DataFrames
    #         prediction = pd.concat([prediction, new_row])
    #         observation = pd.concat([observation, new_row])
    #         last_row = prediction.iloc[-1]
    #         # last_row = prediction.iloc[-5:,]
    #         backtest = pd.concat([backtest, new_row1])
    #         last_row1 = backtest.iloc[-1]

    #         # last_row1 = backtest.iloc[-5:,]

    #     return prediction, backtest

    def full_workflow(
        start_date, end_date, predictors=None, companies=None, stock_settings=None, model = None
    ):
        """
        This function is used to output the prediction of the stock price for the future based on the stock price data from the start date to the end date.

        Args:
        start_date (str): The start date of the stock price data
        end_date (str): The end date of the stock price data
        predictors (list): The list of predictors used to predict the stock price
        companies (list): The list of company names of the stocks
        stock_settings (dict): The dictionary of the stock settings
        """
        default_horizons = [5, 10, 15]
        default_weight = False
        default_refit = True
        default_model = "arimaxgb"
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
                predictors = (
                    ["Market_State",
                        "Close",
                        # "MA_50",
                        # "MA_200",
                        "MA_7",
                        "MA_21",
                        "SP500",
                        "TNX",
                        # "USDCAD=X",
                        "Tech",
                        "Fin",
                        "VIX",
                        # 'Fourier_trans_3_comp_real',
                        # 'Fourier_trans_6_comp_real',
                        # 'Fourier_trans_9_comp_real',
                        # 'Fourier_trans_100_comp_real',
                        # 'Fourier_trans_3_comp_img',
                        # 'Fourier_trans_6_comp_img',
                        # 'Fourier_trans_9_comp_img',
                        # 'Fourier_trans_100_comp_img',

                        # "Fourier_PCA_0",
                        # "Fourier_PCA_1"
                        "FT_real",
                        "FT_img",
                        


                    ]
                    + [
                        "rolling_min",
                        "rolling_median",
                        "rolling_sum",
                        "rolling_ema",
                        "rolling_25p",
                        "rolling_75p",
                    ]
                    + [
                        "RSI",
                        "MACD", "ATR", "Upper_Bollinger", "Lower_Bollinger"] +
                    [   # "Volatility"
                        # 'Daily Returns', 
                        # 'Williams_%R',
                        "Momentum_Interaction",
                        "Volatility_Adj_Momentum",
                        'Stochastic_%K', 'Stochastic_%D', 
                        'Momentum_Score'
                    ]
                )

            predictors = predictors

            predictor = prediction_dataset
            if stock_settings is not None and (
                len(stock_settings) != 0 and company in stock_settings
            ):
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
                    predictors, horizon=horizon, weight=weight, refit=default_refit
                )
                # prediction_dataset._evaluate_models('Close')
                if model is None:
                    pred_model = default_model
                else:
                    pred_model = model
                (
                    prediction,
                    backtest,
                    raw_prediction,
                    raw_backtest,
                ) = predictor.one_step_forward_forecast(  # final_prediction, final_backtest, final_raw_prediction, final_raw_backtest
                    predictors, model_type=pred_model, horizon=horizon
                )
                # print(prediction)
                # print(backtest)
                first_day = pd.to_datetime(
                    end_date - timedelta(days=int(round(1.5 * horizon)))
                )

                backtest_mape = mean_absolute_percentage_error(
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ],
                    backtest[backtest.index >= first_day].Close,
                )
                print("MSE of backtest period vs real data", backtest_mape)
                print("Horizon: ", horizon)
                print(
                    "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                )
                if horizon <= 20:
                    if backtest_mape > 0.15:
                        continue
                else:
                    if backtest_mape > 0.30:
                        continue

                # Data Viz (Not that key)
                plt.figure(figsize=(12, 6))

                # first_day = pd.to_datetime(end_date - timedelta(days=5 + horizon))

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
                    raw_prediction[raw_prediction.index >= first_day].index,
                    raw_prediction[raw_prediction.index >= first_day].Close,
                    label="Raw Prediction",
                    color="green",
                )

                plt.plot(
                    backtest[backtest.index >= first_day].index,
                    backtest[backtest.index >= first_day].Close,
                    label="Backtest",
                    color="red",
                )
                plt.plot(
                    raw_backtest[raw_backtest.index >= first_day].index,
                    raw_backtest[raw_backtest.index >= first_day].Close,
                    label="Raw Backtest",
                    color="orange",
                )
                plt.plot(
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ].index,
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ],
                    label="Actual",
                    color="black",
                )
                # cursor(hover=True)
                plt.title(
                    f"Price Prediction ({prediction_dataset.symbol}) (horizon = {horizon}) (weight = {weight}) (refit = {default_refit}) (model = {pred_model})"
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
