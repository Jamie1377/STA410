from stock_prediction.core import ARIMAXGBoost, StockPredictor
from datetime import date
from sklearn.model_selection import train_test_split

stock = StockPredictor("NVDA", "2024-01-01")
stock.load_data()

X_train, X_test, y_train, y_test = train_test_split(
    stock.data.drop(columns="Close"), stock.data["Close"], shuffle=False
)
print(X_test.index)
model = ARIMAXGBoost()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions) # See if output makes sense