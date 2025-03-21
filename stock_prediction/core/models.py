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

from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
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
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
from pmdarima import auto_arima

# Custom Gradient Descent Implementations #######################################
class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """Custom GD implementation with momentum and adaptive learning
    
    Parameters:
        n_iter (int): Number of iterations
        lr (float): Learning rate
        alpha (float): L2 regularization
        l1_ratio (float): L1 regularization
        momentum (float): Momentum term
        batch_size (int): Mini-batch size
        rmsprop (bool): Use RMSProp optimizer
        
    Attributes:
        coef_ (ndarray): Coefficients
        intercept_ (float): Intercept
        loss_history (list): Loss history
        velocity (ndarray): Velocity
        sq_grad_avg (ndarray): Squared gradient average
        gradients_gd (ndarray): Gradients for GD
        gradients_sgd (ndarray): Gradients for SGD
    """
    def __init__(self, n_iter=1000, lr=0.01, alpha=0.0001, l1_ratio=0.0001, momentum=0.9, batch_size=None, rmsprop=False):
        self.n_iter = n_iter
        self.lr = lr
        self.alpha = alpha  # L2 regularization
        self.l1_ratio = l1_ratio # L1 regularization
        self.momentum = momentum
        self.batch_size = batch_size
        self.coef_ = None
        self.intercept_ = 0.0
        self.rmsprop = rmsprop
        self.loss_history = []
        self.velocity = None
        self.sq_grad_avg = None
        self.gradients_gd = None
        self.gradients_sgd = None

    def _add_bias(self, X):
        """Add bias term to input features"""
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X, y):
        """Fit the model using GD or SGD
        
        Parameters:
            X (ndarray): Features
            y (ndarray): Target
        """
        if self.batch_size and self.batch_size < X.shape[0]:
            self._fit_sgd(X, y)
        else:
            self._fit_gd(X, y)
        return self

    def _fit_gd(self, X, y):
        """Fit the model using GD
        
        Parameters:
            X (ndarray): Features
            y (ndarray): Target
        """
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape
        self.coef_ = np.zeros(n_features)
        velocity = np.zeros_like(self.coef_)
        sq_grad_avg = np.zeros_like(self.coef_)


        for _ in range(self.n_iter):
            self.gradients_gd = 2/n_samples * X_b.T @ (X_b @ self.coef_ - y)
            self.gradients_gd += self.alpha * self.coef_  # L2 regularization
            self.gradients_gd += self.l1_ratio * np.sign(self.coef_)  # L1 regularization
            

            # Update with momentum
            if self.rmsprop:
                sq_grad_avg = self.momentum * sq_grad_avg + (1 - self.momentum) * self.gradients_gd**2
                adj_grad = self.gradients_gd / (np.sqrt(sq_grad_avg) + 1e-8)
                # self.velocity = self.momentum * self.velocity + self.lr * adj_grad
                velocity = self.momentum * self.velocity + (1 - self.momentum) * adj_grad

            else:
                velocity = self.momentum * velocity + self.lr * self.gradients_gd
                
                

            # Update with momentum
            # velocity = self.momentum * velocity + (1 - self.momentum) * self.gradients_gd
            # self.coef_ -= self.lr * velocity
            self.coef_ -= velocity
            
            # Store loss
            loss = np.mean((X_b @ self.coef_ - y)**2) + 0.5*self.alpha*np.sum(self.coef_**2)
            self.loss_history.append(loss)

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def _fit_sgd(self, X, y):
        """Fit the model using SGD"""
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape
        self.coef_ = np.zeros(n_features)
        self.velocity = np.zeros_like(self.coef_)
        self.sq_grad_avg = np.zeros_like(self.coef_)

        for _ in range(self.n_iter):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X_b[indices]
            y_batch = y[indices]

            self.gradients_sgd = 2/self.batch_size * X_batch.T @ (X_batch @ self.coef_ - y_batch)
            self.gradients_sgd += self.alpha * self.coef_
            self.gradients_sgd += self.l1_ratio * np.sign(self.coef_)
            
            # Update with momentum
            if self.rmsprop:
                self.sq_grad_avg = self.momentum * self.sq_grad_avg + (1 - self.momentum) * self.gradients_sgd**2
                adj_grad = self.gradients_sgd / (np.sqrt(self.sq_grad_avg) + 1e-8)
                # self.velocity = self.momentum * self.velocity + self.lr * adj_grad
                self.velocity = self.momentum * self.velocity + (1 - self.momentum) * adj_grad

            else:
                self.velocity = self.momentum * self.velocity + self.lr * self.gradients_sgd
                
                

            # velocity = self.momentum * velocity + (1 - self.momentum) * gradients
            # self.coef_ -= self.lr * velocity
            self.coef_ -= self.velocity
            
            # Store loss
            loss = np.mean((X_batch @ self.coef_ - y_batch)**2) + 0.5*self.alpha*np.sum(self.coef_**2) + self.l1_ratio*np.sum(np.abs(self.coef_))
            self.loss_history.append(loss)

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict(self, X):
        """Make predictions

        Parameters:
            X (ndarray): Features

        Returns:
            ndarray: Predictions
        """
        X_b = self._add_bias(X)
        return X_b @ np.r_[self.intercept_, self.coef_]
    
    def newton_step(self, X_b, y):
        """Perform a Newton step
        
        Parameters:
            X_b (ndarray): Features
            y (ndarray): Target
        
        Returns:
            ndarray: Updated coefficients
        """
        # Compute Hessian (O(n³) - use carefully!)
        hessian = 2/X_b.shape[0] * X_b.T @ X_b + self.alpha * np.eye(X_b.shape[1])
        hessian_inv = np.linalg.inv(hessian)
        grad = self._compute_gradients(X_b, y)
        self.coef_ -= hessian_inv @ grad

# Modified ARIMAXGBoost Class ##################################################
class ARIMAXGBoost(BaseEstimator, RegressorMixin):
    """Hybrid SARIMAX + Boosting ensemble with custom GD/SGD
    
    Parameters:
        xgb_params (dict): XGBoost parameters
        
    Attributes:
        arima_model (SARIMAX): ARIMA model
        arima_model_fit (SARIMAXResults): Fitted ARIMA model
        hwes_model (ExponentialSmoothing): Holt-Winters model
        ses2 (SimpleExpSmoothing): Simple Exponential Smoothing model
        gd_model (GradientDescentRegressor): Custom GD model
        sgd_model (GradientDescentRegressor): Custom SGD model
        lgbm_model (LGBMRegressor): LightGBM model
        catboost_model (CatBoostRegressor): CatBoost model
    """
    def __init__(self, xgb_params=None):
        """Initialize the ARIMA + XGBoost model"""
        self.arima_model = None
        self.linear_model = LinearRegression()
        self.xgb_model = XGBRegressor()
        self.gd_model = GradientDescentRegressor(n_iter=2000, lr=0.1, alpha=0.01, l1_ratio=0.01, momentum=0.75)
        self.sgd_model = GradientDescentRegressor(n_iter=2000, lr=0.01, batch_size=32)
        self.lgbm_model = LGBMRegressor(n_jobs=-1, verbose=-1)
        self.catboost_model = CatBoostRegressor(
            iterations=500, learning_rate=0.1, depth=6, verbose=0
        )

    # def fit(self, X, y):
    #     # ARIMA component
    #     self.arima_model = SARIMAX(
    #         y.values, order=(0, 1, 4), seasonal_order=(2, 1, 2, 6))
    #     self.arima_model_fit = self.arima_model.fit(disp=False)
    #     arima_predictions = self.arima_model_fit.predict()

    #     # Exponential smoothing components
    #     self.hwes_model = ExponentialSmoothing(y.values).fit()
    #     self.ses2 = SimpleExpSmoothing(y.values, initialization_method="heuristic").fit(
    #         smoothing_level=0.6, optimized=False
    #     )

    #     # Custom GD/SGD components
    #     self.gd_model.fit(X, y.values)
    #     self.sgd_model.fit(X, y.values)

    #     # Residual calculation
    #     residuals = y - 0.5 * (
    #         arima_predictions 
    #         + self.hwes_model.fittedvalues
    #         + self.gd_model.predict(X)
    #     )

    #     # Boosting on residuals
    #     self.lgbm_model.fit(X, residuals)
    #     # self.catboost_model.fit(X, residuals)
    def fit(self, X, y):
        """
        Fit the ARIMA and XGBoost models.
        
        Parameters:
        - X: Features (can include lagged values, external features, etc.).
        - y: Target variable (stock prices or price changes).
        """
        # Convert to numpy and clean data
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Handle NaNs and infinities
        X = np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)
        y = np.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5)

        # Validate input shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit ARIMA
        try:
            # self.arima_model =  pm.auto_arima(
            # y,
            # seasonal=True, m=6,
            # stepwise=True, trace=True,
            # # start_p=1,
            # d=1,
            # error_action='ignore',
            # suppress_warnings=True,
            # information_criterion='aic',
            # max_order=10  # Limit parameter search space
            # )
            # self.arima_model_fit = self.arima_model
            self.arima_model =  SARIMAX(
                y, order=(0,1,4), seasonal_order=(2,1,2,6))
            self.arima_model_fit = self.arima_model.fit(disp=False, maxiter=200)
        except Exception as e:
            print(f"ARIMA failed: {str(e)}")
            self.arima_model_fit = None

        # Fit GD/SGD models
        self.gd_model.fit(X_scaled, y)
        self.sgd_model.fit(X_scaled, y)

        # Exponential smoothing components
        self.hwes_model = ExponentialSmoothing(y).fit()
        self.ses2 = SimpleExpSmoothing(y, initialization_method="heuristic").fit(
            smoothing_level=0.6, optimized=False
        )

        # Fit residual models
        residuals = y - self.gd_model.predict(X_scaled)
        self.lgbm_model.fit(X_scaled, residuals)
        self.catboost_model.fit(X_scaled, residuals)

    def predict(self, X):
        """
        Make predictions using the ARIMA + XGBoost model.
        
        Parameters:
        - X: Features (lagged values, external features).
        
        Returns:
        - Final predictions combining ARIMA and XGBoost.
        """
        # Validate and clean input
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)

        if self.scaler is None:
            raise RuntimeError("Model not fitted yet")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get component predictions
        predictions = np.zeros(X.shape[0])
        
        # ARIMA forecast
        if self.arima_model_fit:
            try:
                arima_pred = self.arima_model_fit.forecast(steps=X.shape[0])
                # arima_pred = self.arima_model_fit.predict(n_periods=X.shape[0],return_conf_int=False)
            except:
                arima_pred = np.zeros(X.shape[0])
        else:
            arima_pred = np.zeros(X.shape[0])

        # Exponential smoothing forecasts 
        hwes_forecast = self.hwes_model.forecast(len(X))
        ses2_forecast = self.ses2.forecast(len(X))
        

        # Gradient models
        gd_pred = np.clip(self.gd_model.predict(X_scaled), -1e4, 1e4)
        sgd_pred = np.clip(self.sgd_model.predict(X_scaled), -1e4, 1e4)

        # Boosting residuals
        lgbm_pred = self.lgbm_model.predict(X_scaled)
        catboost_pred = self.catboost_model.predict(X_scaled)

        # Combine predictions
        # predictions = (
        #     0.4 * arima_pred +
        #     0.10 * (hwes_forecast + ses2_forecast) +
        #     0.20 * (gd_pred + sgd_pred) +
        #     0.05 * lgbm_pred +
        #     0.05 * catboost_pred 
        # )
        predictions = (
        0.20 * arima_pred +              # Reduce ARIMA dominance
        0.10 * (hwes_forecast * 0.6 +    # Weight HWES more than SES2
            ses2_forecast * 0.4) +
        0.70 * (gd_pred * 0.8 +         # Favor GD over SGD
            sgd_pred * 0.2) +
        0.05 * lgbm_pred +               # Boost residual correction
        0.05 * catboost_pred             # Balance categorical handling
        )

        # Final sanitization
        return np.nan_to_num(predictions, nan=np.nanmean(predictions))

    # def predict(self, X):
    #     # ARIMA forecasts
    #     arima_forecast = self.arima_model_fit.forecast(steps=len(X))
    #     hwes_forecast = self.hwes_model.forecast(len(X))
        
    #     # GD/SGD predictions
    #     gd_pred = self.gd_model.predict(X)
    #     sgd_pred = self.sgd_model.predict(X)
        
    #     # Boosting predictions
    #     lgbm_pred = self.lgbm_model.predict(X)
    #     # catboost_pred = self.catboost_model.predict(X)

    #     # Ensemble
    #     return 0.3*arima_forecast + 0.2*(gd_pred + sgd_pred) + 0.6*(lgbm_pred) + 0.2*hwes_forecast

# class ARIMAXGBoost(BaseEstimator, RegressorMixin):
#     """Hybrid SARIMAX + Boosting ensemble with configurable components

#     Parameters:
#         sarima_order (tuple): (p,d,q) order for SARIMAX
#         use_sarima (bool): Include SARIMAX component
#         use_ses (bool): Include Simple Exponential Smoothing
#         use_hwes (bool): Include Holt-Winters Exponential Smoothing
#         use_stacking (bool): Include Stacking Regressor
#         use_lgbm (bool): Include LightGBM
#         use_catboost (bool): Include CatBoost
#         weights (dict): Custom weights for model blending
#     """

#     def __init__(self, xgb_params=None):
#         """
#         Initialize the ARIMA + XGBoost model.

#         Parameters:
#         - arima_order: Tuple, order of the ARIMA model (p, d, q).
#         - xgb_params: Dictionary, parameters for the XGBoost model.
#         """
#         # self.arima_order = arima_order
#         self.lstm_model = None
#         self.prophet_model = None
#         self.arima_model = None
#         self.linear_model = LinearRegression()
#         self.xgb_model = XGBRegressor()
#         self.lasso_model = Lasso()
#         self.rf_model = RandomForestRegressor()
#         self.lgbm_model = LGBMRegressor(n_jobs=-1, verbose=100, verbosity=-1)
#         # self.catboost_model = CatBoostRegressor(**dict([('bagging_temperature', 2), ('boosting_type', 'Plain'), ('border_count', 128), ('depth', 6), ('iterations', 100), ('l2_leaf_reg', 3), ('learning_rate', 0.1), ('loss_function', 'RMSE'), ('min_data_in_leaf', 1), ('random_strength', 1)]))
#         self.catboost_model = CatBoostRegressor(
#             iterations=500,
#             learning_rate=0.1,  # Step size shrinkage
#             depth=6,  # Depth of the tree
#             loss_function="RMSE",  # Loss function
#             verbose=100,  # Log every 100 iterations
#         )

#         self.params = {}

#     def fit(self, X, y):
#         """
#         Fit the ARIMA and XGBoost models.

#         Parameters:
#         - X: Features (can include lagged values, external features, etc.).
#         - y: Target variable (stock prices or price changes).
#         """
#         self.arima_model = SARIMAX(
#             y.values,
#             order=(0, 1, 4),
#             seasonal_order=(2, 1, 2, 6),  ##P = 4,# Q = 4 D = 6, #p  =6 ,d =1,q =4
#         )
#         self.arima_model_fit = self.arima_model.fit(disp=False)
#         arima_predictions = self.arima_model_fit.predict()

#         # self.var_model = VAR(X)
#         # self.var_model_fit = self.var_model.predict()
#         # var_predictions = self.var_model.predict(X)
#         # if 'Close' not in list(X.columns):
#         #     self.ses1 = SimpleExpSmoothing(X['Adj Close'], initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)

#         #     self.ses2 = SimpleExpSmoothing(X['Adj Close'], initialization_method="heuristic").fit(smoothing_level=0.6, optimized=False)

#         #     self.ses3 = SimpleExpSmoothing(X['Adj Close'], initialization_method="estimated").fit()

#         # self.varmax_model = VARMAX( endog= y,exog= X, order =(1,1)).fit()
#         self.hwes_model = ExponentialSmoothing(y.values).fit()

#         self.ses1 = SimpleExpSmoothing(y.values, initialization_method="heuristic").fit(
#             smoothing_level=0.2, optimized=False
#         )

#         self.ses2 = SimpleExpSmoothing(y.values, initialization_method="heuristic").fit(
#             smoothing_level=0.6, optimized=False
#         )

#         self.ses3 = SimpleExpSmoothing(
#             y.values, initialization_method="estimated"
#         ).fit()

#         # forecast_input = X.values[-self.var_model_fit.k_ar:]  # Get the last 'k_ar' rows for forecasting
#         # var_predictions = self.var_model_fit.forecast(y=forecast_input, steps=len(y))

#         # Base models (level-0)
#         base_models = [
#             ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
#             (
#                 "gradient_boosting",
#                 GradientBoostingRegressor(n_estimators=100, random_state=42),
#             ),
#             (
#                 "svr",
#                 SVR(kernel="rbf", C=1.0, epsilon=0.1),
#                 ("sdg", SGDRegressor(max_iter=1000, tol=1e-3)),
#             ),
#         ]
#         # Meta-model (level-1)
#         meta_model = self.lgbm_model
#         # Stacking Regressor
#         self.stacking_regressor = StackingRegressor(
#             estimators=base_models, final_estimator=meta_model, cv=5
#         )
#         # Fit the stacking model
#         self.stacking_regressor.fit(X, y)

#         residuals = y - (1 / 3) * (
#             arima_predictions
#             # + self.ses1.fittedvalues
#             + self.ses2.fittedvalues
#             # + self.ses3.fittedvalues
#             + self.hwes_model.fittedvalues
#             # + self.stacking_regressor.predict(X)
#         )
#         # residuals = y - self.ses1.fittedvalues#.values

#         # self.xgb_model.fit(X, residuals)
#         # self.lasso_model.fit(X, residuals)
#         # self.lasso_model.fit(StandardScaler().fit_transform(X), residuals)
#         # self.rf_model.fit(X, residuals)

#         # look_back = 60  # Number of lagged features
#         # X_lagged, y_lagged = self.create_lagged_features(y, look_back)
#         # X_combined = pd.concat([X.iloc[look_back:].reset_index(drop=True), X_lagged.reset_index(drop=True)], axis=1)
#         # Step 4: Align residuals with lagged features
#         # residuals_lagged = residuals.iloc[look_back:]

#         # Step 5: Fit boosting models on lagged features and residuals
#         # self.xgb_model.fit(X_lagged, residuals_lagged)
#         # self.lgbm_model.fit(X_lagged, residuals_lagged)
#         # self.catboost_model.fit(X_lagged, residuals_lagged)
#         # Without lag
#         self.lgbm_model.fit(X, residuals)
#         self.catboost_model.fit(X, residuals)

#         # self.catboost_model.fit(StandardScaler().fit_transform(X), residuals)
#         # if 'Adj Close' not in list(X.columns):
#         #     X['Return'] = X['Close'].pct_change().dropna()
#         # else:
#         #     X['Return'] = X['Adj Close'].pct_change().dropna()
#         # self.arch_model = arch_model(X['Return'].dropna(), vol='Garch', p=1, q=1).fit(disp='off')
#         # self.arch_model = arch_model(X['Return'].dropna(), vol='Garch', p=1, q=1).fit(disp='off')

#     def predict(self, X):
#         """
#         Predict using the ARIMA + XGBoost model.

#         Parameters:
#         - X: Features (lagged values, external features).

#         Returns:
#         - Final predictions combining ARIMA and XGBoost.
#         """

#         arima_predictions = self.arima_model_fit.forecast(steps=len(X))
#         print("ARIMA predictions shape:", arima_predictions.shape)
#         # vars_predictions = self.var_model_fit.forecast(horizon=len(X))

#         ses_predictions_1 = self.ses1.forecast(len(X))  # .rename(r"$\alpha=0.2$")
#         ses_predictions_2 = self.ses2.forecast(len(X))  # .rename(r"$\alpha=0.6$")
#         ses_predictions_3 = self.ses3.forecast(len(X))
#         # varmax_predictions = self.varmax_model.forecast(steps=len(X))
#         hwes_predictions = self.hwes_model.forecast(steps=len(X))
#         stacking_predictions = self.stacking_regressor.predict(X)

#         # Step 3: Get boosting model predictions for residuals
#         # xgb_residuals = self.xgb_model.predict(X_lagged)
#         # lgbm_residuals = self.lgbm_model.predict(X_lagged)
#         # catboost_residuals = self.catboost_model.predict(X_lagged)

#         # xgb_predictions = self.xgb_model.predict(X)
#         # lasso_predictions = self.lasso_model.predict(X)
#         # lasso_predictions = self.lasso_model.predict(StandardScaler().fit_transform(X))
#         # rf_predictions = self.rf_model.predict(X)

#         lgbm_predictions = self.lgbm_model.predict(X)
#         catboost_predictions = self.catboost_model.predict(X)
#         # catboost_predictions = self.catboost_model.predict(StandardScaler().fit_transform(X))

#         # # X['Return'] = X['Adj Close'].pct_change().dropna()
#         # if 'Adj Close' not in list(X.columns):
#         #     X['Return'] = X['Close'].pct_change().dropna()
#         # else:
#         #     X['Return'] = X['Adj Close'].pct_change().dropna()
#         # final_predictions = arch_predictions + lgbm_predictions

#         # Step 3: Combine ARIMA predictions and XGBoost residuals predictions
#         # final_predictions = arima_predictions + xgb_predictions
#         # final_predictions = arima_predictions + lasso_predictions
#         # final_predictions = arima_predictions + lgbm_predictions

#         final_predictions = (1 / 3) * (
#             arima_predictions
#             # + ses_predictions_1
#             + ses_predictions_2
#             # + ses_predictions_3
#             + hwes_predictions
#             # + stacking_predictions
#         ) + (1 / 2) * (lgbm_predictions + catboost_predictions)

#         return final_predictions
    

