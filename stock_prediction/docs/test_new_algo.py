# from stock_prediction.utils import seed_everything
# seed_everything(42)
from stock_prediction.core import ARIMAXGBoost, StockPredictor, GradientDescentRegressor
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
stock = StockPredictor("JPM", "2022-02-01")
stock.load_data()


X =  stock.data.drop(columns="Close")
y = stock.data["Close"]
train_pct_index = int(0.8 * len(stock.data))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# Test ARIMAXGBoost model

model = ARIMAXGBoost()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions) # See if output makes sense
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.legend()
plt.show()