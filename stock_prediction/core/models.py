from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# Boosting Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np

# Time Series Models
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import pmdarima as pm
from pmdarima import auto_arima


# Custom Gradient Descent Implementations
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

    def __init__(
        self,
        n_iter=1000,
        lr=0.01,
        alpha=0.0001,
        l1_ratio=0.0001,
        momentum=0.9,
        batch_size=None,
        rmsprop=False,
    ):
        self.n_iter = n_iter
        self.lr = lr
        self.alpha = alpha  # L2 regularization
        self.l1_ratio = l1_ratio  # L1 regularization
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
        # Initialize velocity and sq_grad_avg properly
        if self.velocity is None:
            self.velocity = 0.0
        if self.sq_grad_avg is None:
            self.sq_grad_avg = 0.0

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

        for _ in range(self.n_iter):
            self.gradients_gd = 2 / n_samples * X_b.T @ (X_b @ self.coef_ - y)
            self.gradients_gd += self.alpha * self.coef_  # L2 regularization
            self.gradients_gd += self.l1_ratio * np.sign(
                self.coef_
            )  # L1 regularization

            # Update with momentum
            if self.rmsprop:
                self.sq_grad_avg = (
                    self.momentum * self.sq_grad_avg
                    + (1 - self.momentum) * self.gradients_gd**2
                )
                adj_grad = self.gradients_gd / (np.sqrt(self.sq_grad_avg) + 1e-8)
                # self.velocity = self.momentum * self.velocity + self.lr * adj_grad
                self.velocity = (
                    self.momentum * self.velocity + (1 - self.momentum) * adj_grad
                )

            else:
                self.velocity = (
                    self.momentum * self.velocity + self.lr * self.gradients_gd
                )

            # Update with momentum
            # velocity = self.momentum * velocity + (1 - self.momentum) * self.gradients_gd
            # self.coef_ -= self.lr * velocity
            self.coef_ -= self.velocity

            # Store loss
            loss = np.mean((X_b @ self.coef_ - y) ** 2) + 0.5 * self.alpha * np.sum(
                self.coef_**2
            )
            self.loss_history.append(loss)

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def _fit_sgd(self, X, y):
        """Fit the model using SGD"""
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape
        self.coef_ = np.zeros(n_features)
        # self.velocity = np.zeros_like(self.coef_)
        # self.sq_grad_avg = np.zeros_like(self.coef_)

        # Initialize velocity and sq_grad_avg if not done
        if self.velocity is None:
            self.velocity = np.zeros_like(self.coef_)
        if self.sq_grad_avg is None:
            self.sq_grad_avg = np.zeros_like(self.coef_)

        for _ in range(self.n_iter):
            indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch = X_b[indices]
            y_batch = y[indices]

            self.gradients_sgd = (
                2 / self.batch_size * X_batch.T @ (X_batch @ self.coef_ - y_batch)
            )
            self.gradients_sgd += self.alpha * self.coef_
            self.gradients_sgd += self.l1_ratio * np.sign(self.coef_)

            # Update with momentum
            if self.rmsprop:
                self.sq_grad_avg = (
                    self.momentum * self.sq_grad_avg
                    + (1 - self.momentum) * self.gradients_sgd**2
                )
                adj_grad = self.gradients_sgd / (np.sqrt(self.sq_grad_avg) + 1e-8)
                # self.velocity = self.momentum * self.velocity + self.lr * adj_grad
                self.velocity = (
                    self.momentum * self.velocity + (1 - self.momentum) * adj_grad
                )

            else:
                self.velocity = (
                    self.momentum * self.velocity + self.lr * self.gradients_sgd
                )

            # velocity = self.momentum * velocity + (1 - self.momentum) * gradients
            # self.coef_ -= self.lr * velocity
            self.coef_ -= self.velocity

            # Store loss
            loss = (
                np.mean((X_batch @ self.coef_ - y_batch) ** 2)
                + 0.5 * self.alpha * np.sum(self.coef_**2)
                + self.l1_ratio * np.sum(np.abs(self.coef_))
            )
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
        hessian = 2 / X_b.shape[0] * X_b.T @ X_b + self.alpha * np.eye(X_b.shape[1])
        hessian_inv = np.linalg.inv(hessian)
        grad = self._compute_gradients(X_b, y)
        self.coef_ -= hessian_inv @ grad


# Modified ARIMAXGBoost Class
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
        self.gd_model = GradientDescentRegressor(
            n_iter=2000, lr=0.1, alpha=0.01, l1_ratio=0.01, momentum=0.75
        )
        self.sgd_model = GradientDescentRegressor(n_iter=2000, lr=0.01, batch_size=32)
        self.lgbm_model = LGBMRegressor(
            n_jobs=-1, verbosity=-1, scale_pos_weight=2, loss_function="Logloss"
        )
        self.catboost_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            loss_function="Huber:delta=1.5",
        )
        self.autoarima = False

    def fit(self, X, y):
        """
        Fit the ARIMA and XGBoost models.

        Parameters:
        - X: Features (can include lagged values, external features, etc.).
        - y: Target variable (stock prices or price changes).
        - autoarima: Whether use auto_arima
        """
        # Convert to numpy and clean data
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Handle NaNs and infinities
        X = np.nan_to_num(X, nan=0.0, posinf=1e15, neginf=-1e15)
        y = np.nan_to_num(y, nan=0.0, posinf=1e15, neginf=-1e15)

        # Validate input shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit ARIMA
        try:
            if self.autoarima:
                self.arima_model = pm.auto_arima(
                    y,
                    seasonal=True,
                    stepwise=True,
                    trace=True,
                    start_p=1,
                    d=1,
                    error_action="ignore",
                    suppress_warnings=True,
                    information_criterion="bic",
                    max_order=8,  # Limit parameter search space
                )
                self.arima_model_fit = self.arima_model
            else:
                self.arima_model = SARIMAX(
                    y, order=(0, 1, 4), seasonal_order=(2, 1, 2, 6)
                )
                self.arima_model.initialize_approximate_diffuse()  # this line
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

        # Add momentum regime detection
        momentum_threshold = 65  # RSI-based threshold
        momentum_regime = np.where(
            X[:, -10] > momentum_threshold,  # 'RSI' index
            0.1,  # Strong upward momentum
            -0.1,
        )  # Weak/downward momentum

        if self.scaler is None:
            raise RuntimeError("Model not fitted yet")

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get component predictions
        predictions = np.zeros(X.shape[0])

        # ARIMA forecast
        if self.arima_model_fit:
            try:
                if self.autoarima:
                    arima_pred = self.arima_model_fit.predict(
                        n_periods=X.shape[0], return_conf_int=False
                    )
                else:
                    arima_pred = self.arima_model_fit.forecast(steps=X.shape[0])
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

        
        # Modify predictions based on momentum regime
        predictions = (
            0.20 * arima_pred * (1 + 0.02 * momentum_regime)
            + 0.10 * (hwes_forecast * 0.6 + ses2_forecast * 0.4)
            + 0.70 * (gd_pred * 0.8 + sgd_pred * 0.2) * (1 + 0.02 * momentum_regime)
            + 0.05 * lgbm_pred
            + 0.05 * catboost_pred
        )

        # Final sanitization
        return np.nan_to_num(predictions, nan=np.nanmean(predictions))