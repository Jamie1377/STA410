from stock_prediction.utils import seed_everything

# seed_everything(42)
import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.optimize import minimize
from scipy.linalg import block_diag
from scipy.linalg import solve_triangular
from statistics import mode

# Boosting Models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor  # Optional: causes numpy compatibility issues

# Time Series Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import pmdarima as pm
from pmdarima import auto_arima  # Computationally expensive

# Alternative of ARIMA or Time Series Models
from statsmodels.tsa.api import VAR
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.utils import AirPassengersDF

# Optuna for advanced hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Suppress warnings
import warnings
from scipy.optimize import OptimizeWarning
import pandas as pd
import numpy as np


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
        n_iter=int(1000),
        lr=0.01,
        alpha=0.0001,
        l1_ratio=0.0001,
        momentum=0.9,
        batch_size=None,
        rmsprop=False,
        random_state=42,  # Add random_state parameter
        newton=False,
        early_stopping=False,
    ):
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.n_iter = n_iter
        self.lr = lr
        self.alpha = alpha  # L2 regularization
        self.l1_ratio = l1_ratio  # L1 regularization
        self.momentum = momentum
        self.batch_size = batch_size
        self.rmsprop = rmsprop
        self.newton = newton
        self.coef_ = None  # w
        self.intercept_ = 0.0  # b
        self.mse_history = []
        self.loss_history = []
        self.loss_mape_history = []
        self.val_mse_history = []
        self.val_loss_history = []
        self.coef_history = []
        self.grad_history = []
        self.velocity = None  # Velocity is also called decay factor
        self.sq_grad_avg = None
        self.gradients_gd = None
        self.gradients_sgd = None
        self.weights = None  # Weights for log weights
        self.early_stopping = early_stopping

    def _add_bias(self, X):
        """Add bias term to input features"""
        return np.c_[np.ones(X.shape[0]), X]

    def _qr_initialization(self, X_b, y):
        """Compute initial coefficients (not intercept) using QR decomposition."""
        Q, R = np.linalg.qr(X_b)  # Decompose X_b = Q @ R
        # R maybe singular, so we use try-except to handle it
        QTy = Q.T @ y  # Project y onto Q's orthogonal basis

        try:
            # print("Using QR decomposition for initialization")
            return solve_triangular(R, QTy)  # Solve R @ coef = Q^T y

        except np.linalg.LinAlgError:
            # Handle singular matrix case
            # print("Matrix is singular, using pseudoinverse")
            # SVD
            U, S, Vt = np.linalg.svd(R)
            S_inv = np.zeros_like(R)
            S_inv[: len(S), : len(S)] = np.diag(1 / S)
            return Vt.T @ S_inv @ U.T @ QTy  # Pseudoinverse solution

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the model using GD or SGD

        Parameters:
            X (ndarray): Features
            y (ndarray): Target
        """
        # Initialize velocity and sq_grad_avg properly
        # if self.velocity is None:
        #     self.velocity = 0.0
        # if self.sq_grad_avg is None:
        #     self.sq_grad_avg = 0.0
        # Reset velocity and sq_grad_avg to None to force reinitialization
        self.mse_history = []
        self.loss_history = []
        self.val_mse_history = []
        self.val_loss_history = []
        self.loss_mape_history = []

        self.velocity = None
        self.sq_grad_avg = None

        if self.batch_size and self.batch_size < X.shape[0]:
            self._fit_sgd(X, y)
        else:
            self._fit_gd(X, y, X_val, y_val)
        return self

    def _fit_gd(self, X, y, X_val=None, y_val=None):
        """Fit the model using GD

        Parameters:
            X (ndarray): Features
            y (ndarray): Target
        """
        seed_everything(self.random_state)  # Use instance seed
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        X_b = self._add_bias(X)
        if X_val is not None and y_val is not None:
            X_val = self._add_bias(X_val)
        n_samples, n_features = X_b.shape
        # self.coef_ = np.zeros(n_features)
        # self.coef_ = np.random.randn(n_features) * 0.01  # Initialize with small random values
        self.coef_ = self._qr_initialization(X_b, y)
        self.intercept_ = mode(y)
        self.coef_[0] = self.intercept_  # Set intercept to the first coefficient

        # Initialize velocity and sq_grad_avg as zero vectors
        self.velocity = np.zeros(n_features)
        self.sq_grad_avg = np.zeros(n_features)

        tol = 0
        for _ in range(
            self.n_iter
        ):  # loss =  1/ n_samples * (X_b.T*self.coef_ -y)**2 (MSE)
            # Compute gradients from the loss function (2 is from the square)
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
            self.coef_history.append(self.coef_.copy())

            if self.newton:
                self.newton_step(X_b, y)
            # Store gradients
            self.gradients_gd = (
                2 / n_samples * X_b.T @ (X_b @ self.coef_ - y)
                + self.alpha * self.coef_
                + self.l1_ratio * np.sign(self.coef_)
            )
            # print('The shape of the gradient: ',self.gradients_gd.shape)
            self.grad_history.append(self.gradients_gd)

            # Track validation loss
            if X_val is not None and y_val is not None:
                val_pred = X_val @ self.coef_
                val_mse = np.mean((val_pred - y_val) ** 2)
                val_loss = (
                    val_mse
                    + 0.5 * self.alpha * np.sum(self.coef_**2)
                    + self.l1_ratio * np.sum(np.abs(self.coef_))
                )

                self.val_loss_history.append(val_loss)
                self.val_mse_history.append(val_mse)

            # Store loss
            mse = np.mean((X_b @ self.coef_ - y) ** 2)
            loss = mse
            +0.5 * self.alpha * np.sum(self.coef_**2)
            +self.l1_ratio * np.sum(
                np.abs(self.coef_)
            )  ### regularization form loss function MSE but stock price is different so MAPE maybe better

            self.mse_history.append(mse)
            self.loss_history.append(loss)

            # Early stopping condition
            loss_mape = np.mean(np.abs((X_b @ self.coef_ - y) / y))
            self.loss_mape_history.append(loss_mape)

            if self.early_stopping and len(self.loss_history) > 2:
                if X_val is not None and y_val is not None:
                    if val_loss < 0.7:

                        print(
                            f"Early stopping at iteration {_} with validation loss: {val_loss:.4f}"
                        )
                        break
                potential_stop_idx = (
                    np.argmin(pd.Series(self.val_mse_history).diff().dropna().values)
                    if len(pd.Series(self.val_mse_history).diff().dropna().values) > 0
                    else 0
                )
                if (
                    potential_stop_idx > 0
                    and self.val_mse_history[potential_stop_idx]
                    < self.val_mse_history[_]
                ):
                    tol += 1
                if tol > 10:
                    print(
                        f"Early stopping at iteration {_} with validation loss (MSE): {self.val_mse_history[potential_stop_idx]:.4f}"
                    )
                    break

                else:
                    if loss_mape < 0.01:
                        print(
                            f"Early stopping at iteration {_} with training loss (MAPE): {loss_mape:.4f}"
                        )
                        break

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def _fit_sgd(self, X, y):
        """Fit the model using SGD"""
        seed_everything(self.random_state)  # Use instance seed
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        X_b = self._add_bias(X)
        n_samples, n_features = X_b.shape
        # self.coef_ = np.zeros(n_features)
        # self.coef_ = np.random.randn(n_features) * 0.01  # Initialize with small random values
        self.coef_ = self._qr_initialization(X_b, y)
        self.intercept_ = np.mean(y)
        self.coef_[0] = self.intercept_  # Set intercept to the first coefficient

        # Initialize velocity and sq_grad_avg as zero vectors
        self.velocity = np.zeros(n_features)
        self.sq_grad_avg = np.zeros(n_features)

        for _ in range(self.n_iter):
            indices = np.random.choice(
                n_samples, self.batch_size, replace=False
            )  # Random Choice of indices
            X_batch = X_b[indices]
            try:
                y_batch = y[indices]
            except IndexError:
                y_batch = y.iloc[indices]

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
            mse = np.mean((X_batch @ self.coef_ - y_batch) ** 2)
            loss = (
                mse
                + 0.5 * self.alpha * np.sum(self.coef_**2)
                + self.l1_ratio * np.sum(np.abs(self.coef_))
            )
            self.mse_history.append(mse)
            self.loss_history.append(loss)
            self.grad_history.append(self.gradients_sgd)
            self.coef_history.append(self.coef_.copy())

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

    def newton_step(
        self, X_b, y
    ):  # DO NOT USE IF IT IS NOT NECESSARY (COMPUTATIONALLY EXPENSIVE)
        """Perform a Newton step using QR decomposition for stability.

        Parameters:
            X_b (ndarray): Features (with bias term)
            y (ndarray): Target

        Returns:
            ndarray: Updated coefficients
        """
        # Compute Hessian matrix (with L2 regularization)
        n_samples = X_b.shape[0]
        hessian = (2 / n_samples) * X_b.T @ X_b + self.alpha * np.eye(X_b.shape[1])

        # Compute gradients (with L1/L2 regularization)
        grad = (
            2 / n_samples * X_b.T @ (X_b @ self.coef_ - y)
            + self.alpha * self.coef_  # L2 term
            + self.l1_ratio * np.sign(self.coef_)  # L1 term
        )

        # QR decomposition for numerical stability
        Q, R = np.linalg.qr(hessian)

        # Solve R * Δ = Q.T @ grad using triangular solver
        try:
            delta = np.linalg.solve(R, Q.T @ grad)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular (should rarely happen with L2 reg)
            delta = np.linalg.lstsq(R, Q.T @ grad, rcond=None)[0]

        # Update coefficients
        self.coef_ -= delta

        return self.coef_

    def optimize_hyperparameters(self, X, y, param_bounds=None, n_iter=1000):
        """Optimize GD/SGD hyperparameters using directional accuracy objective. The direction
        of the prediction is more important than the actual value.
        This is a custom objective function that combines RMSE and directional accuracy.

        Parameters:
            X (ndarray): Features
            y (ndarray): Target
            param_bounds (dict): Bounds for parameters to optimize
            n_iter (int): Number of optimization iterations

        Returns:
            dict: Optimized parameters
        """
        # Default parameter bounds (These paramters appear both with or without rmsprop)
        if param_bounds is None:
            param_bounds = {
                "lr": (0.0001, 0.1),
                "momentum": (0.7, 0.99),
                "alpha": (0.0001, 0.1),  # L2 regularization
                "l1_ratio": (0.0001, 0.1),
                "rmsprop": [False, True],
            }

        # Store original parameters
        original_params = {
            "lr": self.lr,
            "momentum": self.momentum,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "rmsprop": self.rmsprop,
        }

        # Split data for validation
        split_idx = int(len(X) * 0.8)  # Split of time series data
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        def objective(params):
            """Custom objective/loss function combining RMSE and directional accuracy."""
            # Unpack parameters
            self.lr = params[0]
            self.momentum = params[1]
            self.alpha = params[2]
            self.l1_ratio = params[3]
            self.rmsprop = params[4]  # RMSProp or not

            # Train with current params
            self.fit(X_train, y_train)

            # Get predictions and actual values
            preds = self.predict(X_val)
            actual_changes = np.sign(
                np.diff(y_val)
            )  # directional changes of actual values
            pred_changes = np.sign(
                np.diff(preds)
            )  # directional changes of predicted values

            # Calculate metrics
            rmse = root_mean_squared_error(y_val, preds)
            mape = mean_absolute_percentage_error(y_val, preds)

            # Volatility (standard deviation of returns)
            volatility = np.diff(preds) - np.diff(preds).mean()
            volatility = np.std(volatility)

            # Directional accuracy
            min_len = min(len(actual_changes), len(pred_changes))
            dir_acc = np.mean(
                actual_changes[:min_len] == pred_changes[:min_len]
            )  # classfication accuracy

            last_two_prediction = np.mean(
                actual_changes[-2:] == pred_changes[-2:]
            )  # classfication accuracy

            # First prediction deviation

            first_prediction_deviation = np.abs((preds[0] - y_val[0]) / y_val[0])
            mean_pred_deviations = sum(
                [
                    np.abs((preds[i] - y_val[i]) / y_val[i])
                    for i in range(max(len(preds) // 2, len(preds) - 5), len(preds))
                ]
            ) // (len(preds) - max(len(preds) // 2, len(preds) - 5))

            # Combined loss (prioritize both accuracy and error)
            return (
                0.7 * rmse
                + 0.3 * mape
                - 0.2 * dir_acc
                - 0.1 * volatility
                + 30 * first_prediction_deviation
                + 10 * mean_pred_deviations
            )

        # Rationale: if accuracy is high, the loss is low, and vice versa. In other words, if the model's directions are not accurate, the loss is high so it is penalized
        #  Volatility is encouraged to be high so that the model can be more flexible and adaptive to the market changes. The model is penalized if it is too conservative and not adaptive to the market changes.
        # First prediction deviation is encouraged to be low so that the model can be more accurate in the first prediction. The model is penalized if it is too conservative and not adaptive to the market changes.

        # Optimization setup
        initial_guess = [
            self.lr,
            self.momentum,
            self.alpha,
            self.l1_ratio,
            self.rmsprop,
        ]
        bounds = list(param_bounds.values())

        # Constraints
        constraints = [
            {"type": "ineq", "fun": lambda x: x[0] - 0.00001},  # lr > 0.00001
            {"type": "ineq", "fun": lambda x: 0.999 - x[1]},  # momentum < 0.99
            {"type": "ineq", "fun": lambda x: x[2] - 0.0001},  # alpha > 0.0001
            {"type": "ineq", "fun": lambda x: x[3] - 0.0001},  # l1_ratio > 0.0001
            {"type": "ineq", "fun": lambda x: x[4] - 0},  # rmsprop >= 0
            {"type": "ineq", "fun": lambda x: 1 - x[4]},  # rmsprop <= 1
        ]

        # Suppress warnings from scipy.optimize
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        # Run optimization
        result = minimize(
            fun=objective,
            x0=initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": n_iter, "disp": False},
        )

        # Restore original parameters if optimization fails
        if not result.success:
            self.__dict__.update(original_params)
            print(f"Optimization failed")
            return original_params

        # Update with optimized parameters
        optimized_params = {
            "lr": result.x[0],
            "momentum": result.x[1],
            "alpha": result.x[2],
            "l1_ratio": result.x[3],
            "rmsprop": result.x[4],
        }
        self.__dict__.update(
            optimized_params
        )  # Update model parameters after optimization (No need to reinitialize)
        # print(f"Optimized parameters for {n_iter} iterations, { {k: self.__dict__[k] for k in list(self.__dict__.keys())[:8]} }") #list(self.__dict__.items())[:8]
        if optimized_params != original_params:
            print("Optimization successful")
        else:
            print("Optimization failed, parameters are not changed")
        return optimized_params

    def optimize_hyperparameters_optuna(
        self,
        X,
        y,
        n_trials=100,
        study_name=None,
        walk_forward=True,
        window_size=252,
        custom_objective="geometric_expectancy_mdd",
    ):
        """
        Advanced Optuna-based hyperparameter optimization with walk-forward validation.
        Inspired by the XAUUSD strategy from Reddit article.

        Parameters:
            X (ndarray): Features
            y (ndarray): Target values
            n_trials (int): Number of Optuna trials
            study_name (str): Name for the Optuna study
            walk_forward (bool): Whether to use walk-forward validation
            window_size (int): Size of training window for walk-forward (default: 252 trading days = 1 year)
            custom_objective (str): Objective function type
                - "geometric_expectancy_mdd": Geometric expectancy / max drawdown
                - "sharpe_sortino": Combined Sharpe and Sortino ratio
                - "directional_rmse": Directional accuracy with RMSE penalty

        Returns:
            dict: Best parameters found by Optuna
        """

        def calculate_trading_metrics(predictions, actual):
            """Calculate comprehensive trading metrics similar to the Reddit article."""
            # Price changes (returns)
            actual_returns = np.diff(actual) / actual[:-1]
            pred_returns = np.diff(predictions) / predictions[:-1]

            # Clean infinite/NaN values
            actual_returns = np.nan_to_num(actual_returns, nan=0, posinf=0, neginf=0)
            pred_returns = np.nan_to_num(pred_returns, nan=0, posinf=0, neginf=0)

            # Directional accuracy
            actual_direction = np.sign(actual_returns)
            pred_direction = np.sign(pred_returns)
            directional_accuracy = (
                np.mean(actual_direction == pred_direction)
                if len(actual_direction) > 0
                else 0
            )

            # Simulated trading returns (assuming perfect execution on direction)
            trading_returns = np.where(
                pred_direction == actual_direction,
                np.abs(actual_returns),
                -np.abs(actual_returns),
            )

            # Risk metrics
            if len(trading_returns) > 0 and np.std(trading_returns) > 0:
                sharpe_ratio = (
                    np.mean(trading_returns) / np.std(trading_returns) * np.sqrt(252)
                )
                downside_returns = trading_returns[trading_returns < 0]
                sortino_ratio = (
                    np.mean(trading_returns) / np.std(downside_returns) * np.sqrt(252)
                    if len(downside_returns) > 0 and np.std(downside_returns) > 0
                    else 0
                )
            else:
                sharpe_ratio = 0
                sortino_ratio = 0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + trading_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.abs(np.min(drawdown)) if len(drawdown) > 0 else 1

            # Geometric expectancy (average geometric return)
            geometric_mean = (
                np.prod(1 + trading_returns) ** (1 / len(trading_returns)) - 1
                if len(trading_returns) > 0
                else 0
            )

            return {
                "directional_accuracy": directional_accuracy,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "geometric_expectancy": geometric_mean,
                "total_return": (
                    np.prod(1 + trading_returns) - 1 if len(trading_returns) > 0 else 0
                ),
            }

        def objective_function(trial):
            """Optuna objective function with custom metrics."""

            # Sample hyperparameters
            lr = trial.suggest_float("lr", 0.0001, 0.1, log=True)
            momentum = trial.suggest_float("momentum", 0.7, 0.99)
            alpha = trial.suggest_float("alpha", 0.0001, 0.1, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0001, 0.1, log=True)
            rmsprop = trial.suggest_categorical("rmsprop", [True, False])
            n_iter = trial.suggest_int("n_iter", 500, 2000, step=100)

            # Create temporary model with trial parameters
            temp_model = GradientDescentRegressor(
                lr=lr,
                momentum=momentum,
                alpha=alpha,
                l1_ratio=l1_ratio,
                rmsprop=rmsprop,
                n_iter=n_iter,
                random_state=self.random_state,
                early_stopping=self.early_stopping,
            )

            if walk_forward:
                # Walk-forward validation similar to the Reddit article
                all_predictions = []
                all_actuals = []

                # Ensure we have enough data
                if (
                    len(X) < window_size + 50
                ):  # Need at least window_size + test samples
                    window_size = max(len(X) // 3, 50)  # Adaptive window size

                start_idx = window_size
                step_size = max(
                    1, (len(X) - start_idx) // 20
                )  # Limit to ~20 validation windows

                for i in range(start_idx, len(X) - 10, step_size):
                    # Training window
                    train_start = max(0, i - window_size)
                    X_train = X[train_start:i]
                    y_train = y[train_start:i]

                    # Test window (small out-of-sample)
                    test_end = min(i + 10, len(X))
                    X_test = X[i:test_end]
                    y_test = y[i:test_end]

                    if len(X_train) < 10 or len(X_test) < 1:
                        continue

                    try:
                        # Fit and predict
                        temp_model.fit(X_train, y_train)
                        preds = temp_model.predict(X_test)

                        all_predictions.extend(preds)
                        all_actuals.extend(y_test)

                    except Exception as e:
                        # Skip this iteration if training fails
                        continue

                if len(all_predictions) < 10:
                    return float("inf")  # Not enough valid predictions

                predictions = np.array(all_predictions)
                actuals = np.array(all_actuals)

            else:
                # Simple train-test split
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                try:
                    temp_model.fit(X_train, y_train)
                    predictions = temp_model.predict(X_test)
                    actuals = y_test
                except Exception as e:
                    return float("inf")

            # Calculate trading metrics
            metrics = calculate_trading_metrics(predictions, actuals)

            # Custom objective based on specified type
            if custom_objective == "geometric_expectancy_mdd":
                # Similar to the Reddit article: geometric expectancy / max drawdown
                if metrics["max_drawdown"] > 0:
                    objective_value = -(
                        metrics["geometric_expectancy"] / metrics["max_drawdown"]
                    )
                else:
                    objective_value = -metrics["geometric_expectancy"]

            elif custom_objective == "sharpe_sortino":
                # Combined Sharpe and Sortino ratio
                objective_value = -(
                    0.6 * metrics["sharpe_ratio"] + 0.4 * metrics["sortino_ratio"]
                )

            elif custom_objective == "directional_rmse":
                # Your original approach: directional accuracy with RMSE penalty
                rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
                mape = np.mean(
                    np.abs((predictions - actuals) / np.maximum(np.abs(actuals), 1e-8))
                )
                objective_value = (
                    0.4 * rmse + 0.3 * mape - 0.3 * metrics["directional_accuracy"]
                )

            else:
                raise ValueError(f"Unknown objective type: {custom_objective}")

            # Store additional metrics for analysis
            trial.set_user_attr("directional_accuracy", metrics["directional_accuracy"])
            trial.set_user_attr("sharpe_ratio", metrics["sharpe_ratio"])
            trial.set_user_attr("sortino_ratio", metrics["sortino_ratio"])
            trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
            trial.set_user_attr("total_return", metrics["total_return"])

            return objective_value

        # Create Optuna study
        study_name = study_name or f"gd_optimization_{custom_objective}"

        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20)

        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner, study_name=study_name
        )

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        print(f"Starting Optuna optimization with {n_trials} trials...")
        print(f"Objective: {custom_objective}")
        print(f"Walk-forward validation: {walk_forward}")

        try:
            study.optimize(
                objective_function, n_trials=n_trials, timeout=3600
            )  # 1 hour timeout
        except Exception as e:
            print(f"Optimization encountered an error: {e}")
            return self._get_current_params()

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        # Update model with best parameters
        self.lr = best_params["lr"]
        self.momentum = best_params["momentum"]
        self.alpha = best_params["alpha"]
        self.l1_ratio = best_params["l1_ratio"]
        self.rmsprop = best_params["rmsprop"]
        self.n_iter = best_params["n_iter"]

        # Print results
        print(f"\nOptuna optimization completed!")
        print(f"Best objective value: {best_value:.6f}")
        print(f"Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # Print best trial metrics
        best_trial = study.best_trial
        if hasattr(best_trial, "user_attrs"):
            print(f"\nBest trial metrics:")
            print(
                f"  Directional accuracy: {best_trial.user_attrs.get('directional_accuracy', 'N/A'):.4f}"
            )
            print(
                f"  Sharpe ratio: {best_trial.user_attrs.get('sharpe_ratio', 'N/A'):.4f}"
            )
            print(
                f"  Sortino ratio: {best_trial.user_attrs.get('sortino_ratio', 'N/A'):.4f}"
            )
            print(
                f"  Max drawdown: {best_trial.user_attrs.get('max_drawdown', 'N/A'):.4f}"
            )
            print(
                f"  Total return: {best_trial.user_attrs.get('total_return', 'N/A'):.4f}"
            )

        return best_params

    def _get_current_params(self):
        """Get current model parameters."""
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "rmsprop": self.rmsprop,
            "n_iter": self.n_iter,
        }


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
        seed_everything(42)
        self.arima_model = None
        self.linear_model = LinearRegression()
        self.xgb_model = XGBRegressor(random_state=42, is_provide_training_metric=True)
        self.gd_model = GradientDescentRegressor(
            n_iter=1000,
            lr=0.05,
            alpha=0.01,
            l1_ratio=0.01,
            momentum=0.9,
            rmsprop=False,
            random_state=42,
            early_stopping=True,
        )
        self.sgd_model = GradientDescentRegressor(
            n_iter=1200, lr=0.01, batch_size=32, rmsprop=True, random_state=42
        )  # To ensure reproducibility
        self.lgbm_model = LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            scale_pos_weight=2,
            loss_function="Logloss",
            is_provide_training_metric=True,
        )
        self.catboost_model = CatBoostRegressor(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            loss_function="Huber:delta=1.5",
            random_seed=42,
        )
        self.autoarima = False

    def fit(self, X, y, display=False):
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

        # Optimize hyperparameters for GD/SGD
        _ = self.gd_model.optimize_hyperparameters(X_scaled, y)
        _ = self.sgd_model.optimize_hyperparameters(X_scaled, y)
        if display:
            print(
                f"GD model parameters: { {k: self.gd_model.__dict__[k] for k in list(self.gd_model.__dict__.keys())[:8]} }"
            )
            print(
                f"SGD model parameters: { {k: self.sgd_model.__dict__[k] for k in list(self.sgd_model.__dict__.keys())[:8]}}"
            )
        # Fit GD/SGD models
        self.gd_model.fit(X_scaled, y)
        self.sgd_model.fit(X_scaled, y)

        # Exponential smoothing components
        self.hwes_model = ExponentialSmoothing(y).fit()
        self.ses2 = SimpleExpSmoothing(y, initialization_method="heuristic").fit(
            smoothing_level=0.6, optimized=False
        )

        # Fit residual models (Allow flexibility)
        residuals = y - 0.5 * (
            self.gd_model.predict(X_scaled) + self.sgd_model.predict(X_scaled)
        )
        self.lgbm_model.fit(X_scaled, residuals)
        self.catboost_model.fit(X_scaled, residuals)
        if display:
            print(
                f"residuals mean: {np.sum(residuals)/len(residuals)}, stock price mean {np.mean(y)}"
            )  # residuals mean (by day) on natural scale

        # Collect gradient histories
        # self.gd_loss = self.gd_model.loss_history
        # self.sgd_loss = self.sgd_model.loss_history
        # self.gd_grad_norms = [np.linalg.norm(g)
        #                     for g in self.gd_model.gradients_gd]
        # self.sgd_grad_norms = [np.linalg.norm(g)
        #                      for g in self.sgd_model.gradients_sgd]

    def optimize_ensemble_optuna(self, X, y, n_trials=50, optimize_models=True):
        """
        Optimize ensemble weights and optionally model hyperparameters using Optuna.
        Similar to the walk-forward optimization approach from the Reddit article.

        Parameters:
            X (ndarray): Features
            y (ndarray): Target values
            n_trials (int): Number of Optuna trials
            optimize_models (bool): Whether to also optimize individual model hyperparameters

        Returns:
            dict: Best parameters and ensemble weights
        """

        def calculate_ensemble_metrics(predictions, actual):
            """Calculate trading-focused metrics for ensemble evaluation."""
            # Handle edge cases
            if len(predictions) < 2 or len(actual) < 2:
                return {
                    "directional_accuracy": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 1,
                    "total_return": 0,
                    "rmse": float("inf"),
                }

            # Price returns
            actual_returns = np.diff(actual) / np.maximum(actual[:-1], 1e-8)
            pred_returns = np.diff(predictions) / np.maximum(predictions[:-1], 1e-8)

            # Clean data
            actual_returns = np.nan_to_num(actual_returns, nan=0, posinf=0, neginf=0)
            pred_returns = np.nan_to_num(pred_returns, nan=0, posinf=0, neginf=0)

            # Directional accuracy
            actual_direction = np.sign(actual_returns)
            pred_direction = np.sign(pred_returns)
            directional_accuracy = (
                np.mean(actual_direction == pred_direction)
                if len(actual_direction) > 0
                else 0
            )

            # RMSE
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))

            # Simulated trading returns
            trading_returns = np.where(
                pred_direction == actual_direction,
                np.abs(actual_returns),
                -np.abs(actual_returns),
            )

            # Risk metrics
            if len(trading_returns) > 0 and np.std(trading_returns) > 0:
                sharpe_ratio = (
                    np.mean(trading_returns) / np.std(trading_returns) * np.sqrt(252)
                )
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + trading_returns)
            if len(cumulative_returns) > 0:
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / np.maximum(
                    running_max, 1e-8
                )
                max_drawdown = np.abs(np.min(drawdown))
            else:
                max_drawdown = 1

            # Total return
            total_return = (
                np.prod(1 + trading_returns) - 1 if len(trading_returns) > 0 else 0
            )

            return {
                "directional_accuracy": directional_accuracy,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "rmse": rmse,
            }

        def ensemble_objective(trial):
            """Optuna objective function for ensemble optimization."""

            # Optimize individual model hyperparameters if requested
            if optimize_models:
                # GD model parameters
                gd_lr = trial.suggest_float("gd_lr", 0.001, 0.1, log=True)
                gd_momentum = trial.suggest_float("gd_momentum", 0.7, 0.99)
                gd_alpha = trial.suggest_float("gd_alpha", 0.0001, 0.1, log=True)

                # SGD model parameters
                sgd_lr = trial.suggest_float("sgd_lr", 0.001, 0.1, log=True)
                sgd_momentum = trial.suggest_float("sgd_momentum", 0.7, 0.99)
                sgd_batch_size = trial.suggest_int("sgd_batch_size", 16, 128, step=16)

                # Update model parameters
                self.gd_model.lr = gd_lr
                self.gd_model.momentum = gd_momentum
                self.gd_model.alpha = gd_alpha

                self.sgd_model.lr = sgd_lr
                self.sgd_model.momentum = sgd_momentum
                self.sgd_model.batch_size = sgd_batch_size

            # Optimize ensemble weights
            w_arima = trial.suggest_float("w_arima", 0.0, 0.5)
            w_hwes = trial.suggest_float("w_hwes", 0.0, 0.4)
            w_ses = trial.suggest_float("w_ses", 0.0, 0.4)
            w_gd = trial.suggest_float("w_gd", 0.1, 0.8)
            w_sgd = trial.suggest_float("w_sgd", 0.1, 0.8)
            w_lgbm = trial.suggest_float("w_lgbm", 0.0, 0.2)
            w_catboost = trial.suggest_float("w_catboost", 0.0, 0.2)

            # Normalize weights to sum to 1
            total_weight = w_arima + w_hwes + w_ses + w_gd + w_sgd + w_lgbm + w_catboost
            if total_weight == 0:
                return float("inf")

            w_arima /= total_weight
            w_hwes /= total_weight
            w_ses /= total_weight
            w_gd /= total_weight
            w_sgd /= total_weight
            w_lgbm /= total_weight
            w_catboost /= total_weight

            # Store weights for later use
            trial.set_user_attr(
                "weights",
                {
                    "w_arima": w_arima,
                    "w_hwes": w_hwes,
                    "w_ses": w_ses,
                    "w_gd": w_gd,
                    "w_sgd": w_sgd,
                    "w_lgbm": w_lgbm,
                    "w_catboost": w_catboost,
                },
            )

            # Walk-forward validation
            window_size = min(252, len(X) // 3)  # 1 year or 1/3 of data
            start_idx = window_size
            step_size = max(1, (len(X) - start_idx) // 15)  # ~15 validation windows

            all_predictions = []
            all_actuals = []

            for i in range(start_idx, len(X) - 10, step_size):
                try:
                    # Training window
                    train_start = max(0, i - window_size)
                    X_train = X[train_start:i]
                    y_train = y[train_start:i]

                    # Test window
                    test_end = min(i + 5, len(X))
                    X_test = X[i:test_end]
                    y_test = y[i:test_end]

                    if len(X_train) < 20 or len(X_test) < 1:
                        continue

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train models
                    self.gd_model.fit(X_train_scaled, y_train)
                    self.sgd_model.fit(X_train_scaled, y_train)

                    # Exponential smoothing on training data
                    hwes_model = ExponentialSmoothing(y_train).fit()
                    ses_model = SimpleExpSmoothing(
                        y_train, initialization_method="heuristic"
                    ).fit(smoothing_level=0.6, optimized=False)

                    # Train residual models
                    residuals = y_train - 0.5 * (
                        self.gd_model.predict(X_train_scaled)
                        + self.sgd_model.predict(X_train_scaled)
                    )
                    lgbm_model = LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1)
                    catboost_model = CatBoostRegressor(
                        iterations=50,
                        learning_rate=0.1,
                        depth=4,
                        verbose=0,
                        random_seed=42,
                    )
                    lgbm_model.fit(X_train_scaled, residuals)
                    catboost_model.fit(X_train_scaled, residuals)

                    # Get component predictions
                    gd_pred = self.gd_model.predict(X_test_scaled)
                    sgd_pred = self.sgd_model.predict(X_test_scaled)
                    hwes_pred = hwes_model.forecast(len(X_test))
                    ses_pred = ses_model.forecast(len(X_test))
                    lgbm_pred = lgbm_model.predict(X_test_scaled)
                    catboost_pred = catboost_model.predict(X_test_scaled)

                    # ARIMA predictions (simplified)
                    try:
                        arima_model = SARIMAX(y_train, order=(1, 1, 1))
                        arima_fit = arima_model.fit(disp=False, maxiter=50)
                        arima_pred = arima_fit.forecast(steps=len(X_test))
                    except:
                        arima_pred = np.full(len(X_test), np.mean(y_train))

                    # Ensemble prediction
                    ensemble_pred = (
                        w_arima * arima_pred
                        + w_hwes * hwes_pred
                        + w_ses * ses_pred
                        + w_gd * gd_pred
                        + w_sgd * sgd_pred
                        + w_lgbm * lgbm_pred
                        + w_catboost * catboost_pred
                    )

                    all_predictions.extend(ensemble_pred)
                    all_actuals.extend(y_test)

                except Exception as e:
                    # Skip this iteration if training fails
                    continue

            if len(all_predictions) < 10:
                return float("inf")

            # Calculate metrics
            metrics = calculate_ensemble_metrics(
                np.array(all_predictions), np.array(all_actuals)
            )

            # Custom objective: similar to Reddit article (geometric expectancy / max drawdown)
            if metrics["max_drawdown"] > 0:
                objective_value = -(metrics["total_return"] / metrics["max_drawdown"])
            else:
                objective_value = -metrics["total_return"]

            # Store metrics
            trial.set_user_attr("directional_accuracy", metrics["directional_accuracy"])
            trial.set_user_attr("sharpe_ratio", metrics["sharpe_ratio"])
            trial.set_user_attr("max_drawdown", metrics["max_drawdown"])
            trial.set_user_attr("total_return", metrics["total_return"])
            trial.set_user_attr("rmse", metrics["rmse"])

            return objective_value

        # Create and run Optuna study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="ensemble_optimization",
        )

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"Starting Optuna ensemble optimization with {n_trials} trials...")
        print(f"Optimize individual models: {optimize_models}")

        try:
            study.optimize(
                ensemble_objective, n_trials=n_trials, timeout=1800
            )  # 30 min timeout
        except Exception as e:
            print(f"Ensemble optimization encountered an error: {e}")
            return {}

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial

        # Update model with best parameters if optimizing models
        if optimize_models:
            self.gd_model.lr = best_params.get("gd_lr", self.gd_model.lr)
            self.gd_model.momentum = best_params.get(
                "gd_momentum", self.gd_model.momentum
            )
            self.gd_model.alpha = best_params.get("gd_alpha", self.gd_model.alpha)

            self.sgd_model.lr = best_params.get("sgd_lr", self.sgd_model.lr)
            self.sgd_model.momentum = best_params.get(
                "sgd_momentum", self.sgd_model.momentum
            )
            self.sgd_model.batch_size = best_params.get(
                "sgd_batch_size", self.sgd_model.batch_size
            )

        # Store ensemble weights
        self.ensemble_weights = best_trial.user_attrs.get("weights", {})

        # Print results
        print(f"\nOptuna ensemble optimization completed!")
        print(f"Best objective value: {best_value:.6f}")
        print(f"Best ensemble weights:")
        for weight_name, weight_value in self.ensemble_weights.items():
            print(f"  {weight_name}: {weight_value:.4f}")

        if hasattr(best_trial, "user_attrs"):
            print(f"\nBest trial metrics:")
            print(
                f"  Directional accuracy: {best_trial.user_attrs.get('directional_accuracy', 'N/A'):.4f}"
            )
            print(
                f"  Sharpe ratio: {best_trial.user_attrs.get('sharpe_ratio', 'N/A'):.4f}"
            )
            print(
                f"  Max drawdown: {best_trial.user_attrs.get('max_drawdown', 'N/A'):.4f}"
            )
            print(
                f"  Total return: {best_trial.user_attrs.get('total_return', 'N/A'):.4f}"
            )
            print(f"  RMSE: {best_trial.user_attrs.get('rmse', 'N/A'):.4f}")

        return {
            "best_params": best_params,
            "ensemble_weights": self.ensemble_weights,
            "best_metrics": (
                best_trial.user_attrs if hasattr(best_trial, "user_attrs") else {}
            ),
        }

    def fit_with_optuna(self, X, y, n_trials=50, optimize_models=True, display=False):
        """
        Fit the ensemble model with Optuna optimization.
        This replaces the regular fit method with optimized parameters.

        Parameters:
            X (ndarray): Features
            y (ndarray): Target values
            n_trials (int): Number of Optuna trials
            optimize_models (bool): Whether to optimize individual model hyperparameters
            display (bool): Whether to display fitting information
        """

        # First, run the optimization
        optimization_results = self.optimize_ensemble_optuna(
            X, y, n_trials, optimize_models
        )

        # Now fit the model with optimized parameters
        self.fit(X, y, display=display)

        return optimization_results

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

        # Boosting residuals (give the fluttering model a chance)
        lgbm_pred = self.lgbm_model.predict(X_scaled)
        catboost_pred = self.catboost_model.predict(X_scaled)

        # Use optimized ensemble weights if available (from Optuna optimization)
        if hasattr(self, "ensemble_weights") and self.ensemble_weights:
            weights = self.ensemble_weights
            predictions = (
                weights.get("w_arima", 0.20) * arima_pred * (1 + 0.02 * momentum_regime)
                + weights.get("w_hwes", 0.18) * hwes_forecast
                + weights.get("w_ses", 0.12) * ses2_forecast
                + weights.get("w_gd", 0.40) * gd_pred * (1 + 0.02 * momentum_regime)
                + weights.get("w_sgd", 0.10) * sgd_pred * (1 + 0.02 * momentum_regime)
                + weights.get("w_lgbm", 0.02) * lgbm_pred
                + weights.get("w_catboost", 0.02) * catboost_pred
            )
        else:
            # Default weights (original implementation)
            predictions = (
                0.20 * arima_pred * (1 + 0.02 * momentum_regime)
                + 0.30 * (hwes_forecast * 0.6 + ses2_forecast * 0.4)
                + 0.50 * (gd_pred * 0.8 + sgd_pred * 0.2) * (1 + 0.02 * momentum_regime)
                + 0.02 * lgbm_pred
                + 0.02 * catboost_pred
            )

        # Final sanitization
        return np.nan_to_num(predictions, nan=np.nanmean(predictions))
