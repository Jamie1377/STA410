from stock_prediction.utils import seed_everything
# seed_everything(42)

from stock_prediction.core import ARIMAXGBoost, StockPredictor
from datetime import date
from sklearn.model_selection import train_test_split


stock = StockPredictor("NVDA", "2024-01-01")
stock.load_data()


X =  stock.data.drop(columns="Close")
y = stock.data["Close"]
train_pct_index = int(0.8 * len(stock.data))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]

# print(X_test.index)
model = None
model = ARIMAXGBoost()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions) # See if output makes sense