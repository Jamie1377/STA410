from stock_prediction.core import ARIMAXGBoost, StockPredictor
from datetime import date
from sklearn.model_selection import train_test_split

wtf = StockPredictor("F", "2024-01-01")
wtf.load_data()


print()


X_train, X_test, y_train, y_test = train_test_split(
    wtf.data.drop(columns="Close"), wtf.data["Close"], shuffle=False
)
print(X_test.index)
model = ARIMAXGBoost()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)
