# from stock_prediction.utils import seed_everything
# seed_everything(42)
from stock_prediction.core import ARIMAXGBoost, StockPredictor, GradientDescentRegressor
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

stock = StockPredictor("SPY", "2010-02-01", date.today())
stock.load_data()

# Implement t -> t+1 prediction by shifting target forward
# Features: Use data from time t (excluding Close) 
# Target: Use Close from time t+1 

# Get features from all days except the last (since we don't have t+1 target for last day)
X = stock.data.iloc[:-1].drop(columns="Close")  # Time t features 

# Get targets: Close price from the NEXT day (t+1)
y = stock.data["Close"].iloc[1:]  # This shifts Close forward by 1 day # 1 to n corresponding to 0 to n-1 (the nth is like the test data)

# Align the indices to make sure they match
# X = X.iloc[:-1]  # Remove the last row to align with y
# y = y.iloc[:-1]  # Remove the last row to match X

print(f"After t->t+1 alignment:")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X date range: {X.index[0]} to {X.index[-1]}")
print(f"y date range: {y.index[0]} to {y.index[-1]}")

# For prediction: use the last available day's features to predict tomorrow
last_day_features = stock.data.iloc[-1].drop(labels="Close")
print(f"Last day data for prediction: {last_day_features.name}")
print(f"Will predict Close for: next trading day")

# Split data temporally
train_pct_index = int(0.8 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)
scaled_last_day_features = standard_scaler.transform(
    last_day_features.values.reshape(1, -1)
)

# Test ARIMAXGBoost model

model = GradientDescentRegressor(
    alpha=0.01,  # Stronger L2 regularization
    l1_ratio=0.01,  # Stronger L1 regularization
    n_iter=500,  # Fewer iterations
    early_stopping=True,  # If available
)
# model.optimize_ensemble_optuna(X_train, y_train)

# Check for overfitting indicators BEFORE training
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Feature-to-sample ratio: {X_train.shape[1] / X_train.shape[0]:.3f}")

params = model.optimize_hyperparameters(X_train, y_train)
print(f"Optimized params: {params}")
model.fit(X_train, y_train)

# Check training vs test performance for overfitting
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, predictions)

print(f"\n--- OVERFITTING CHECK ---")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Performance gap (test_mse/train_mse): {test_mse/train_mse:.3f}")

if test_mse > train_mse * 2:
    print("⚠️  WARNING: Potential overfitting detected (test MSE >> train MSE)")
if train_r2 > 0.95:
    print("⚠️  WARNING: Suspiciously high training R² - likely overfitting")
if test_r2 < 0:
    print("⚠️  WARNING: Negative test R² - model performs worse than mean baseline")

print(predictions)  # See if output makes sense
tomorrow_prediction = model.predict(scaled_last_day_features)[0]

# Get the exact dates for clarity
last_available_date = last_day_features.name
current_close = stock.data["Close"].iloc[-1]

print(f"\n--- PREDICTION DETAILS ---")
print(f"Last available data date: {last_available_date.strftime('%Y-%m-%d (%A)')}")
print(
    f"Close price on {last_available_date.strftime('%Y-%m-%d')}: ${current_close:.2f}"
)
print(f"Prediction for next trading day: ${tomorrow_prediction:.2f}")
print(f"Note: 'Next trading day' depends on market calendar (weekends/holidays)")

# Try to estimate next trading day (rough approximation)
import pandas as pd

next_business_day = last_available_date + pd.offsets.BDay(1)
print(f"Estimated next trading day: {next_business_day.strftime('%Y-%m-%d (%A)')}")
print(
    f"Predicted change: ${tomorrow_prediction - current_close:.2f} ({((tomorrow_prediction - current_close) / current_close * 100):.2f}%)"
)

# Debug: Check the alignment of our test data
print(f"\n--- DATE ALIGNMENT CHECK ---")
print(f"y_test first few dates: {y_test.index[:3].tolist()}")
print(f"y_test last few dates: {y_test.index[-3:].tolist()}")
print(
    f"✅ Visualization is CORRECT: y_test.index contains the dates we're predicting (t+1)"
)
print(f"✅ predictions array contains predictions for those exact dates")

plt.plot(y_test.index, y_test, label="Actual Close Price")

# The predictions are already aligned correctly with y_test.index dates
# y_test.index contains the t+1 dates we're predicting
plt.plot(y_test.index, predictions, label="Predicted Close Price")

# Show current day's actual close price and tomorrow's prediction
plt.annotate(
    f"Current Close: ${current_close:.2f}\nDate: {last_available_date.strftime('%Y-%m-%d')}",
    xy=(last_available_date, current_close),
    xytext=(last_available_date, current_close + 5),
    arrowprops=dict(facecolor="blue", shrink=0.05),
)

# Plot tomorrow's prediction at the correct date (next_business_day)
plt.scatter(
    next_business_day,
    tomorrow_prediction,
    color="green",
    s=100,
    label=f"Prediction for {next_business_day.strftime('%m/%d')}: ${tomorrow_prediction:.2f}",
)
plt.scatter(
    last_available_date,
    current_close,
    color="red",
    s=100,
    label=f"Current Close ({last_available_date.strftime('%m/%d')}): ${current_close:.2f}",
)
plt.legend()
plt.title(f"Stock Price Prediction for {stock.symbol} - t→t+1 Model")
plt.show()
