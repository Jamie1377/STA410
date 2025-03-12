from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, SGDRegressor, LinearRegression
from sklearn.ensemble import (
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

class ARIMAXGBoost(BaseEstimator, RegressorMixin):
    """Hybrid SARIMAX + Boosting ensemble with configurable components

    Parameters:
        sarima_order (tuple): (p,d,q) order for SARIMAX
        use_sarima (bool): Include SARIMAX component
        use_ses (bool): Include Simple Exponential Smoothing
        use_hwes (bool): Include Holt-Winters Exponential Smoothing
        use_stacking (bool): Include Stacking Regressor
        use_lgbm (bool): Include LightGBM
        use_catboost (bool): Include CatBoost
        weights (dict): Custom weights for model blending
    """

    # def __init__(
    #     self,
    #     sarima_order=(0, 1, 4),
    #     seasonal_order=(2, 1, 2, 6),
    #     use_sarima=True,
    #     use_ses=True,
    #     use_hwes=True,
    #     use_stacking=True,
    #     use_lgbm=True,
    #     use_catboost=True,
    #     weights=None,
    # ):

    #     self.sarima_order = sarima_order
    #     self.seasonal_order = seasonal_order
    #     self.use_sarima = use_sarima
    #     self.use_ses = use_ses
    #     self.use_hwes = use_hwes
    #     self.use_stacking = use_stacking
    #     self.use_lgbm = use_lgbm
    #     self.use_catboost = use_catboost
    #     self.weights = weights or {}

    #     # Initialize components
    #     self.base_models = [
    #         ("rf", RandomForestRegressor(n_estimators=100)),
    #         ("gb", GradientBoostingRegressor(n_estimators=100)),
    #     ]
    #     self.stacking_regressor = (
    #         StackingRegressor(
    #             estimators=self.base_models, final_estimator=LGBMRegressor()
    #         )
    #         if use_stacking
    #         else None
    #     )

    #     self.lgbm = LGBMRegressor() if use_lgbm else None
    #     self.catboost = CatBoostRegressor(verbose=0) if use_catboost else None

    # def fit(self, X, y):
    #     """Fit selected components to training data"""
    #     self.models_ = {}

    #     # Time series components
    #     if self.use_sarima:
    #         self.models_["sarima"] = SARIMAX(
    #             y, order=self.sarima_order, seasonal_order=self.seasonal_order
    #         ).fit(disp=False)

    #     if self.use_ses:
    #         self.models_["ses"] = SimpleExpSmoothing(y).fit()

    #     if self.use_hwes:
    #         self.models_["hwes"] = ExponentialSmoothing(y).fit()

    #     # Machine learning components
    #     residuals = self._calculate_residuals(X, y)

    #     if self.use_stacking:
    #         self.stacking_regressor.fit(X, residuals)

    #     if self.use_lgbm:
    #         self.lgbm.fit(X, residuals)

    #     if self.use_catboost:
    #         self.catboost.fit(X, residuals)

    #     return self

    # def predict(self, X):
    #     """Generate predictions using selected components"""
    #     predictions = []

    #     # Time series forecasts
    #     if self.use_sarima:
    #         predictions.append(self.models_["sarima"].forecast(steps=len(X)))

    #     if self.use_ses:
    #         predictions.append(self.models_["ses"].forecast(len(X)))

    #     if self.use_hwes:
    #         predictions.append(self.models_["hwes"].forecast(len(X)))

    #     # ML predictions
    #     ml_preds = []
    #     if self.use_stacking:
    #         ml_preds.append(self.stacking_regressor.predict(X))

    #     if self.use_lgbm:
    #         ml_preds.append(self.lgbm.predict(X))

    #     if self.use_catboost:
    #         ml_preds.append(self.catboost.predict(X))

    #     if ml_preds:
    #         predictions.append(np.mean(ml_preds, axis=0))

    #     # Apply custom weights if provided
    #     if self.weights:
    #         weighted = sum(
    #             p * self.weights.get(name, 1)
    #             for name, p in zip(self.models_, predictions)
    #         )
    #         return weighted / sum(self.weights.values())

    #     return np.mean(predictions, axis=0)

    # def _calculate_residuals(self, X, y):
    #     """Calculate residuals from time series models"""
    #     base_preds = []

    #     if self.use_sarima:
    #         base_preds.append(self.models_["sarima"].fittedvalues)

    #     if self.use_ses:
    #         base_preds.append(self.models_["ses"].fittedvalues)

    #     if self.use_hwes:
    #         base_preds.append(self.models_["hwes"].fittedvalues)

    #     if not base_preds:
    #         return y  # If no TS models, use full signal

    #     return y - np.mean(base_preds, axis=0)
    



    def __init__(self, xgb_params=None):
        """
        Initialize the ARIMA + XGBoost model.

        Parameters:
        - arima_order: Tuple, order of the ARIMA model (p, d, q).
        - xgb_params: Dictionary, parameters for the XGBoost model.
        """
        # self.arima_order = arima_order
        self.lstm_model = None
        self.prophet_model = None
        self.arima_model = None
        self.linear_model = LinearRegression()
        self.xgb_model = XGBRegressor()
        self.lasso_model = Lasso()
        self.rf_model = RandomForestRegressor()
        self.lgbm_model = LGBMRegressor(n_jobs=-1, verbose=100, verbosity=-1)
        # self.catboost_model = CatBoostRegressor(**dict([('bagging_temperature', 2), ('boosting_type', 'Plain'), ('border_count', 128), ('depth', 6), ('iterations', 100), ('l2_leaf_reg', 3), ('learning_rate', 0.1), ('loss_function', 'RMSE'), ('min_data_in_leaf', 1), ('random_strength', 1)]))
        self.catboost_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,  # Step size shrinkage
            depth=6,  # Depth of the tree
            loss_function="RMSE",  # Loss function
            verbose=100,  # Log every 100 iterations
        )

        self.params = {}

    def fit(self, X, y):
        """
        Fit the ARIMA and XGBoost models.

        Parameters:
        - X: Features (can include lagged values, external features, etc.).
        - y: Target variable (stock prices or price changes).
        """
        # Step 1: Fit ARIMA model

        # train_size = int(len(y) * 0.8)  # 80% for training
        # train, test = y[:train_size], y[train_size:]

        # p = 4 q = 3
        # d = D =1
        # P = Q = 1

        self.arima_model = SARIMAX(
            y.values, order=(0, 1, 4), seasonal_order=(2, 1, 2, 6)   ##P = 4,# Q = 4 D = 6, #p  =6 ,d =1,q =4
        )
        self.arima_model_fit = self.arima_model.fit(disp=False)
        arima_predictions = self.arima_model_fit.predict()

        # self.var_model = VAR(X)
        # self.var_model_fit = self.var_model.predict()
        # var_predictions = self.var_model.predict(X)
        # if 'Close' not in list(X.columns):
        #     self.ses1 = SimpleExpSmoothing(X['Adj Close'], initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)

        #     self.ses2 = SimpleExpSmoothing(X['Adj Close'], initialization_method="heuristic").fit(smoothing_level=0.6, optimized=False)

        #     self.ses3 = SimpleExpSmoothing(X['Adj Close'], initialization_method="estimated").fit()

        # self.varmax_model = VARMAX( endog= y,exog= X, order =(1,1)).fit()
        self.hwes_model = ExponentialSmoothing(y.values).fit()

        self.ses1 = SimpleExpSmoothing(y.values, initialization_method="heuristic").fit(
            smoothing_level=0.2, optimized=False
        )

        self.ses2 = SimpleExpSmoothing(y.values, initialization_method="heuristic").fit(
            smoothing_level=0.6, optimized=False
        )

        self.ses3 = SimpleExpSmoothing(
            y.values, initialization_method="estimated"
        ).fit()

        # forecast_input = X.values[-self.var_model_fit.k_ar:]  # Get the last 'k_ar' rows for forecasting
        # var_predictions = self.var_model_fit.forecast(y=forecast_input, steps=len(y))


        # Base models (level-0)
        base_models = [
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1),
        ("sdg", SGDRegressor(max_iter=1000, tol=1e-3)),)
        ]
        # Meta-model (level-1)
        meta_model = self.lgbm_model
        # Stacking Regressor
        self.stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
        # Fit the stacking model
        self.stacking_regressor.fit(X, y)


        residuals = y - (1 / 3) * (
            arima_predictions
            # + self.ses1.fittedvalues
            + self.ses2.fittedvalues
            # + self.ses3.fittedvalues
            + self.hwes_model.fittedvalues
            # + self.stacking_regressor.predict(X)
        )
        # residuals = y - self.ses1.fittedvalues#.values

        # self.xgb_model.fit(X, residuals)
        # self.lasso_model.fit(X, residuals)
        # self.lasso_model.fit(StandardScaler().fit_transform(X), residuals)
        # self.rf_model.fit(X, residuals)

        # look_back = 60  # Number of lagged features
        # X_lagged, y_lagged = self.create_lagged_features(y, look_back)
        # X_combined = pd.concat([X.iloc[look_back:].reset_index(drop=True), X_lagged.reset_index(drop=True)], axis=1)
        # Step 4: Align residuals with lagged features
        # residuals_lagged = residuals.iloc[look_back:]

        # Step 5: Fit boosting models on lagged features and residuals
        # self.xgb_model.fit(X_lagged, residuals_lagged)
        # self.lgbm_model.fit(X_lagged, residuals_lagged)
        # self.catboost_model.fit(X_lagged, residuals_lagged)
        # Without lag
        self.lgbm_model.fit(X, residuals)
        self.catboost_model.fit(X, residuals)

        # self.catboost_model.fit(StandardScaler().fit_transform(X), residuals)
        # if 'Adj Close' not in list(X.columns):
        #     X['Return'] = X['Close'].pct_change().dropna()
        # else:
        #     X['Return'] = X['Adj Close'].pct_change().dropna()
        # self.arch_model = arch_model(X['Return'].dropna(), vol='Garch', p=1, q=1).fit(disp='off')
        # self.arch_model = arch_model(X['Return'].dropna(), vol='Garch', p=1, q=1).fit(disp='off')

    def predict(self, X):
        """
        Predict using the ARIMA + XGBoost model.

        Parameters:
        - X: Features (lagged values, external features).

        Returns:
        - Final predictions combining ARIMA and XGBoost.
        """

        arima_predictions = self.arima_model_fit.forecast(steps=len(X))
        print("ARIMA predictions shape:", arima_predictions.shape)
        # vars_predictions = self.var_model_fit.forecast(horizon=len(X))

        ses_predictions_1 = self.ses1.forecast(len(X))  # .rename(r"$\alpha=0.2$")
        ses_predictions_2 = self.ses2.forecast(len(X))  # .rename(r"$\alpha=0.6$")
        ses_predictions_3 = self.ses3.forecast(
            len(X)
        )  
        # varmax_predictions = self.varmax_model.forecast(steps=len(X))
        hwes_predictions = self.hwes_model.forecast(steps=len(X))
        stacking_predictions = self.stacking_regressor.predict(X)


        # Step 3: Get boosting model predictions for residuals
        # xgb_residuals = self.xgb_model.predict(X_lagged)
        # lgbm_residuals = self.lgbm_model.predict(X_lagged)
        # catboost_residuals = self.catboost_model.predict(X_lagged)


        # xgb_predictions = self.xgb_model.predict(X)
        # lasso_predictions = self.lasso_model.predict(X)
        # lasso_predictions = self.lasso_model.predict(StandardScaler().fit_transform(X))
        # rf_predictions = self.rf_model.predict(X)

        lgbm_predictions = self.lgbm_model.predict(X)
        catboost_predictions = self.catboost_model.predict(X)
        # catboost_predictions = self.catboost_model.predict(StandardScaler().fit_transform(X))

        # # X['Return'] = X['Adj Close'].pct_change().dropna()
        # if 'Adj Close' not in list(X.columns):
        #     X['Return'] = X['Close'].pct_change().dropna()
        # else:
        #     X['Return'] = X['Adj Close'].pct_change().dropna()
        # final_predictions = arch_predictions + lgbm_predictions

        # Step 3: Combine ARIMA predictions and XGBoost residuals predictions
        # final_predictions = arima_predictions + xgb_predictions
        # final_predictions = arima_predictions + lasso_predictions
        # final_predictions = arima_predictions + lgbm_predictions

        final_predictions = (1 / 3) * (
            arima_predictions
            # + ses_predictions_1
            + ses_predictions_2
            # + ses_predictions_3
            + hwes_predictions
            # + stacking_predictions
        ) + (1 / 2) * (lgbm_predictions + catboost_predictions)

        # final_predictions = arima_predictions + catboost_predictions
        # final_predictions = arima_predictions + (1 / 3) * (
        #     xgb_residuals
        #     + lgbm_residuals
        #     + catboost_residuals
        # )
        return final_predictions
