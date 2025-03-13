from stock_prediction.core import StockPredictor
from datetime import date
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






# def full_workflow(start_date, end_date, predictors = None, companies = None, ):
#     """
#     This function is used to output the prediction of the stock price for the next 10 days

#     Args:
#     start_date (str): The start date of the stock price data
#     end_date (str): The end date of the stock price data
#     predictors (list): The list of predictors used to predict the stock price
#     companies (list): The list of company names of the stocks
#     """
#     default_horizons = [5, 7, 10]
#     default_weight = False
#     if companies is None:
#         companies = ["AXP"]
#     for company in companies:
#         prediction_dataset = StockPredictor(
#             company,
#             start_date=start_date,
#             end_date= end_date,
#             interval="1d",
#         )
#         prediction_dataset.load_data()
#         if predictors is None:
#             predictors =  ['Close', 'MA_50', 'MA_200',
#                         'SP500','TNX','USDCAD=X','Tech','Fin','VIX'] + ['rolling_min', 'rolling_median',
#         'rolling_sum', 'rolling_ema', 'rolling_25p','rolling_75p',] 
#         predictors = predictors
        
#         predictor = prediction_dataset
#         if company in stock_settings:
#             # Use custom settings for the stock
#             settings = stock_settings[company]
#             horizons = settings["horizons"]
#             weight = settings["weight"]
#         else:
#             # Use default settings for other stocks
#             horizons = default_horizons
#             weight = default_weight
#         for horizon in horizons: 
#             prediction_dataset.prepare_models(predictors, horizon=horizon, weight=weight)
#             # prediction_dataset._evaluate_models('Close')
#             prediction, backtest, predictions_dict = predictor.one_step_forward_forecast(
#                 predictors, model_type="arimaxgb", horizon=horizon
#             )

#             # Data Viz (Not that key)
#             plt.figure(figsize=(12, 6))

#             first_day = pd.to_datetime(end_date - timedelta(days=10 + horizon))

#             plt.plot(
#                 prediction[prediction.index >= prediction_dataset.data.iloc[-1].name].index,
#                 prediction[prediction.index >= prediction_dataset.data.iloc[-1].name].Close,
#                 label="Prediction",
#                 color="blue",
#             )
#             plt.plot(
#                 backtest[backtest.index > first_day].index,
#                 backtest[backtest.index > first_day].Close,
#                 label="Backtest",
#                 color="red",
#             )
#             plt.plot(
#                 prediction_dataset.data.Close[
#                     prediction_dataset.data.index > first_day
#                 ].index,
#                 prediction_dataset.data.Close[prediction_dataset.data.index > first_day],
#                 label="Actual",
#                 color="black",
#             )
#             # cursor(hover=True)
#             plt.title(
#                 f"Price Prediction ({prediction_dataset.symbol}) (horizon = {horizon}) (weight = {weight})"
#             )
#             plt.axvline(
#                 x=backtest.index[-1],
#                 color="g",
#                 linestyle="--",
#                 label="Reference Line (Last Real Data Point)",
#             )
#             plt.text(
#                 backtest.index[-1],
#                 backtest.Close[-1],
#                 f"x={str(backtest.index[-1].date())}",
#                 ha="right",
#                 va="bottom",
#             )

#             plt.xlabel("Date")
#             plt.ylabel("Stock Price")
#             plt.legend()
#             plt.show()

StockPredictor.full_workflow('2023-01-01',date.today())
