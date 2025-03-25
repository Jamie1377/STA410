from stock_prediction.core import StockPredictor
from datetime import date
import time
import yfinance as yf

stock_settings = {
    "V": {"horizons": [10], "weight": False},
    "GE": {"horizons": [5], "weight": False},
    "ANF": {"horizons": [7, 10], "weight": False},
    "AVGO": {"horizons": [7, 5, 10, 12], "weight": False},
    "AXP": {"horizons": [5, 10, 13], "weight": False},
    "NVDA": {"horizons": [5, 10, 12], "weight": False},
    "MCO": {"horizons": [3], "weight": False},
    "PYPL": {"horizons": [5], "weight": False},
    "SPGI": {"horizons": [12], "weight": False},
    "AAPL": {"horizons": [3, 5, 7, 10], "weight": False},
    "ORCL": {"horizons": [3, 7], "weight": False},
    "LCID": {"horizons": [5, 10, 15], "weight": False},
    "c": {"horizons": [5, 7, 12], "weight": False},
    "KDP": {"horizons": [3, 5, 12], "weight": False},
    "SBUX": {"horizons": [5, 10, 15], "weight": False},
    "ZS": {"horizons": [4, 5], "weight": False},
    "MET": {"horizons": [5], "weight": False},
    "EBAY": {"horizons": [10], "weight": False},
    "TMUS": {"horizons": [10], "weight": False},
    "ASML": {"horizons": [10], "weight": False},
    "MTCH": {"horizons": [10], "weight": False},
    "DINO": {"horizons": [15], "weight": False},
    "XOM": {"horizons": [15], "weight": False},
    "CVX":  {"horizons": [15], "weight": False},
    "COP": {"horizons": [15], "weight": False},
    "WMB":  {"horizons": [15], "weight": False},
    "EPD": {"horizons": [15], "weight": False},

}

print(list(yf.Sector("energy").top_companies.index))
lis = [
    "XOM",
    "CVX",
    "COP",
    "WMB",
    "EPD",
    "EOG",
    "ET",
    "KMI",
    "OKE",
    "SLB",
    "MPLX",
    "LNG",
    "PSX",
    "HES",
    "MPC",
    "FANG",
    "OXY",
    "TRGP",
    "BKR",
    "VLO",
    "EQT",
    "TPL",
    "CQP",
    "EXE",
    "DVN",
    "HAL",
    "CTRA",
    "WES",
    "PAA",
    "FTI",
    "AR",
    "PR",
    "OVV",
    "DTM",
    "RRC",
    "HESM",
    "AM",
    "VNOM",
    "SUN",
    "APA",
    "NFG",
    "CHRD",
    "MTDR",
    "ENLC",
    "DINO",
    "NOV",
    "CRK",
    "CHX",
    "LB",
    "PAGP",
]
toc = time.time()
StockPredictor.full_workflow("2023-09-01", date.today(), companies=["AXP"], stock_settings=stock_settings)
tic = time.time()
tic-toc
print('Time taken', tic-toc)

# AAPL 10

# prediction_dataset = StockPredictor(
#                 "SPGI",
#                 start_date="2024-01-01",
#                 end_date= date.today(),
#                 interval="1d",)
# prediction_dataset.load_data()
# predictor = prediction_dataset
# for horizon in [5]:
#     prediction_dataset.prepare_models(predictors= [
#                     "Close",
#                     "MA_50",
#                     "MA_200",
#                     "SP500",
#                     "TNX",
#                     "USDCAD=X",
#                     "Tech",
#                     "Fin",
#                     "VIX",
#                 ] + [
#                     "rolling_min",
#                     "rolling_median",
#                     "rolling_sum",
#                     "rolling_ema",
#                     "rolling_25p",
#                     "rolling_75p",
#                 ], horizon=horizon, weight=False)
# prediction, backtest, backtest2 = (predictor.one_step_forward_forecast(
#                         [
#                     "Close",
#                     "MA_50",
#                     "MA_200",
#                     "SP500",
#                     "TNX",
#                     "USDCAD=X",
#                     "Tech",
#                     "Fin",
#                     "VIX",
#                 ] + [
#                     "rolling_min",
#                     "rolling_median",
#                     "rolling_sum",
#                     "rolling_ema",
#                     "rolling_25p",
#                     "rolling_75p",
#                 ], model_type="arimaxgb", horizon=horizon
#                     )
#                 )
# print(prediction)
# print(backtest)

# print(prediction_dataset.data.head())
# print(prediction_dataset.data.iloc[-10:,].mean(axis=0))
