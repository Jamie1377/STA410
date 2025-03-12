from stock_prediction.core import StockPredictor, ARIMAXGBoost
from datetime import date, timedelta
import matplotlib.pyplot as plt
import time
import pandas as pd
# # Initialize predictor
# predictor = StockPredictor("AAPL", "2024-01-01")
# predictor.load_data()

# # Custom model configuration
# model_config = {
#     "use_sarima": True,
#     "use_stacking": True,
#     "use_lgbm": True,
#     "weights": {"sarima": 0.6, "stacking": 0.4},
#     "sarima_order": (1, 1, 1),
#     "seasonal_order": (1, 1, 1, 5),
# }

# # Train and predict
# predictor.train_model(model_config)
# forecast = predictor.predict(days=5)

# print("Next 5 day forecast:")
# print(forecast)



# for company in list(finance.top_companies.index):
# for company in list(tech.top_etfs.keys()):
# for company in list(tech.top_companies.iloc[30:45,].index):

stock_settings = {
    "V": {"horizons": [7], "weight": False},
    "GE": {"horizons": [5], "weight": False},
    "ANF": {"horizons": [7, 10], "weight": False},
    "AVGO": {"horizons": [7, 5, 10, 12], "weight": False},
    "AXP": {"horizons": [5], "weight": False},
    "NVDA": {"horizons": [12], "weight": False},
    "MCO": {"horizons": [1], "weight": False},
    "PYPL": {"horizons": [5], "weight": False},
    "SPGI": {"horizons": [7], "weight": True},
    "AAPL": {"horizons": [3], "weight": True},
    "ORCL": {"horizons": [3, 7], "weight": False},
    "LCID": {"horizons": [7], "weight": False},
    "CRWD": {"horizons": [7, 12], "weight": False},
    "KDP":  {"horizons": [3, 5, 12], "weight": False},
    "SBUX": {"horizons": [ 4, 5, 7], "weight": False},
    "ZS": {"horizons": [4, 5], "weight": False},
    "MET": {"horizons": [5], "weight": False},
    "EBAY": {"horizons": [10], "weight": False},
    "TMUS": {"horizons": [10], "weight": False},
    "ASML": {"horizons": [10], "weight": False},
    "MTCH": {"horizons": [10], "weight": False},}

# Default settings f1or other companies
default_horizons = [5, 7, 10]
default_weight = False
 
for company in [
    # [random.choice(Tickers["stock"]) if random.choice(Tickers["stock"]) not in stock_settings.keys() else ValueError][0],
    # [random.choice(Tickers["stock"]) if random.choice(Tickers["stock"]) not in stock_settings.keys() else ValueError][0],
    # [random.choice(Tickers["stock"]) if random.choice(Tickers["stock"]) not in stock_settings.keys() else ValueError][0],
    # 'ZS',
    # 'SBUX',
    # "LCID", #shit
    # "IBKR",
    # "AFL",
    # "MET",
    # "AAPL",
    # "ORCL",
    # # "AMD",
    # "ANF",  # not good for day ahead
    # "AVGO", # ok day ahead try True
    # "AXP", # not good for day ahead try True
    # "V",
    # "MCO",
    # "GE",
    # "NVDA",
    # "PYPL",
    # "CRWD", # watch
    # "SPGI", # not good for day ahead
    # "EBAY",
    # "KDP",
    # "MET",
    "MTCH"
]:
    # [
    #     "MET",
    #     "AVGO",
    #     "SPGI",
    #     "AIG",
    #     "AXP",
        # "MCO",
    #     "PYPL",
    #     "V",
    #     "MS"]:

    # [
    #     "V",
    #     "MS",
    #     "GS",
    #     "AXP",
    #     "BX",
    #     "BLK",
    #     "SPGI",
    #     "MCO",
    #     "PYPL",
    #     "IBKR",
    #     "AFL",
    #     "MET",
    #     "ARES",
    #     "NU",
    #     "AIG",
    #     "PRU",
    # ]:
  
    prediction_dataset = StockPredictor(
        company,
        start_date="2023-01-01",
        end_date= date.today(),
        interval="1d",
        #  + pd.Timedelta(days=1)
    )
    prediction_dataset.load_data()
# Volume mau not help stock price prediction
   
    predictors =  ['Close', 'MA_50', 'MA_200',
                    # 'BB_Low',
                    'SP500','TNX','USDCAD=X','Tech','Fin','VIX'] + ['rolling_min', 'rolling_median',
       'rolling_sum', 'rolling_ema', 'rolling_25p','rolling_75p',] 
    
    predictor = prediction_dataset
    if company in stock_settings:
        # Use custom settings for the stock
        settings = stock_settings[company]
        horizons = settings["horizons"]
        weight = settings["weight"]
    else:
        # Use default settings for other stocks
        horizons = default_horizons
        weight = default_weight
    for horizon in horizons: #[5, 7, 10, 12]:  # [3, 5, 7]:,
        prediction_dataset.prepare_models(predictors, horizon=horizon, weight=weight)
        # prediction_dataset._evaluate_models('Close')
        prediction, backtest, predictions_dict = predictor.one_step_forward_forecast(
            predictors, model_type="arimaxgb", horizon=horizon
        )

        # Data Viz (Not that key)
        plt.figure(figsize=(12, 6))

        first_day = pd.to_datetime(date.today() - timedelta(days=10 + horizon))

        plt.plot(
            prediction[prediction.index >= prediction_dataset.data.iloc[-1].name].index,
            prediction[prediction.index >= prediction_dataset.data.iloc[-1].name].Close,
            label="Prediction",
            color="blue",
        )
        plt.plot(
            backtest[backtest.index > first_day].index,
            backtest[backtest.index > first_day].Close,
            label="Backtest",
            color="red",
        )
        plt.plot(
            prediction_dataset.data.Close[
                prediction_dataset.data.index > first_day
            ].index,
            prediction_dataset.data.Close[prediction_dataset.data.index > first_day],
            label="Actual",
            color="black",
        )
        # cursor(hover=True)

        current_time = time.strftime("%H:%M:%S", time.localtime())
        plt.title(
            f"Price Prediction ({prediction_dataset.symbol}) (horizon = {horizon}) (weight = {weight})"
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