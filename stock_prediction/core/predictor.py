from stock_prediction.utils import seed_everything

seed_everything(42)
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from hmmlearn.hmm import GaussianHMM
from xgboost import XGBRegressor

# Custom imports
from stock_prediction.core import ARIMAXGBoost
from stock_prediction.utils import get_next_valid_date, optimize_lookback

# Sample Dataset
stock_data = yf.download("AAPL", start="2024-01-01", end=date.today())
stock_data.columns = stock_data.columns.droplevel(1)
stock_data

# Add to models.py
import requests
import pandas as pd
from datetime import timedelta
import hashlib
import joblib

# API imports
api_key = "PKR0BKC0QMVXGC6WUYZB"
secret_key = "nIonxysbSHIIC77ojMjiQPgCrug78echi1IcZMe8"
paper = True
# DO not change this
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None

# Alpaca API imports
import os
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    OrderClass,
    QueryOrderStatus,
)

trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
# api = tradeapi.REST(
#     key_id='PKR0BKC0QMVXGC6WUYZB',
#     secret_key='nIonxysbSHIIC77ojMjiQPgCrug78echi1IcZMe8',
#     base_url='https://paper-api.alpaca.markets/v2'  # Paper trading endpoint
# )

# def generate_trading_signal(predictor, symbol):
#     predictor.load_data()
#     lookback = optimize_lookback(
#         predictor.data.drop(columns="Close"),
#         predictor.data["Close"],
#         model=XGBRegressor(
#             n_estimators=20,
#             max_depth=3,
#             learning_rate=0.1,
#             random_state=42,
#             n_jobs=-1,
#         ),
#         min_window=60,
#         step_size=10,
#         n_splits=5,
#     )
#     print(f"Optimal lookback window: {lookback}")
#     predictor.data = predictor.data.iloc[-lookback:]
#     features = [
#         # "Market_State",
#         "Close",
#         "MA_50",
#         # "MA_200",
#         # "High",
#         # "Low",
#         "MA_7",
#         # "MA_21",
#         "SP500",
#         "TNX",
#         "USDCAD=X",
#         # "Tech",
#         # "Fin",
#         "VIX",
#         "Energy",
#     ]
#     + [
#         "rolling_min",
#         "rolling_median",
#         "rolling_sum",
#         "rolling_ema",
#         "rolling_25p",
#         "rolling_75p",
#     ]
#     + ["RSI", "MACD", "ATR", "Upper_Bollinger", "Lower_Bollinger"]
#     + ["VWAP"]
#     + [  # "Volatility",
#         "Daily Returns",
#         "Williams_%R",
#         "Momentum_Interaction",
#         # "Volatility_Adj_Momentum",
#         "Stochastic_%K",
#         "Stochastic_%D",
#         "Momentum_Score",
#     ] # Use same features as in notebook
#     horizon = 5  # Prediction window

#     predictor.prepare_models(features, horizon=horizon)
#     forecast, _, _, _ = predictor.one_step_forward_forecast(
#         predictors=features,
#         model_type="arimaxgb",
#         horizon=horizon
#     )

#     # Get latest prediction
#     predicted_price = forecast['Close'].iloc[-1]
#     current_price = predictor.data['Close'].iloc[-1]

#     # Generate signal
#     if predicted_price > current_price * 1.01:  # 1% threshold
#         return 'BUY'
#     elif predicted_price < current_price * 0.99:
#         return 'SELL'
#     else:
#         return 'HOLD'

# def execute_trade(symbol, signal, api):
#     position = api.get_position(symbol)
#     cash = api.get_account().cash
#     current_price = api.get_last_trade(symbol).price
#     # In execute_trade()
#     max_risk_per_trade = 0.02  # 2% of portfolio
#     portfolio_value = float(api.get_account().equity)
#     max_loss = portfolio_value * max_risk_per_trade

#     if signal == 'BUY' and not position:
#         # Risk management: don't use more than 20% of cash
#         max_qty = (float(cash) * 0.2) // current_price
#         api.submit_order(
#             symbol=symbol,
#             qty=max_qty,
#             side='buy',
#             type='market',
#             time_in_force='day'
#         )
#     elif signal == 'SELL' and position:
#         api.submit_order(
#             symbol=symbol,
#             qty=position.qty,
#             side='sell',
#             type='market',
#             time_in_force='day'
#         )


class MarketSentimentAnalyzer:  # Compuationally expensive, try to use volatility to replace the sentiment
    """Get market sentiment scores using free financial APIs"""

    def __init__(self, api_key=None):
        self.api_key = api_key or "YOUR_API_KEY"  # Get free key from Alpha Vantage
        self.sentiment_cache = {}

    def get_alpha_vantage_sentiment(self, ticker="SPY"):
        """Get news sentiment from Alpha Vantage's API"""
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"

        try:
            response = requests.get(url)
            data = response.json()
            recent_items = data.get("feed", [])[:10]  # Last 10 articles

            sentiment_scores = []
            for item in recent_items:
                for ticker_sentiment in item.get("ticker_sentiment", []):
                    if ticker_sentiment["ticker"] == ticker:
                        sentiment_scores.append(
                            float(ticker_sentiment["ticker_sentiment_score"])
                        )

            return np.mean(sentiment_scores) if sentiment_scores else 0.5

        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            return 0.5  # Neutral fallback

    def get_fear_greed_index(self):
        """Get Crypto Fear & Greed Index (works for general market)"""
        try:
            response = requests.get("https://api.alternative.me/fng/")
            data = response.json()
            return int(data["data"][0]["value"])
        except:
            return 50  # Neutral fallback

    def get_historical_sentiment(self, ticker, days):
        """Get smoothed historical sentiment (cached)"""
        if ticker in self.sentiment_cache:
            return self.sentiment_cache[ticker]

        dates = stock_data.index[-days:]
        scores = []

        for _ in range(days):
            scores.append(self.get_alpha_vantage_sentiment(ticker))

        # Create smoothed series
        series = pd.Series(scores, index=dates).rolling(3).mean().bfill()
        self.sentiment_cache[ticker] = series
        return series


class StockPredictor:
    """Stock price prediction pipeline

    Parameters:
        symbol (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        interval (str): Data interval (1d, 1h, etc)
    """

    def __init__(self, symbol, start_date, end_date=None, interval="1d"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else date.today()
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_params = {}
        self.data = None
        self.feature_sets = {
            "Close": {"target": "Close", "features": None},
            "Low": {"target": "Low", "features": None},
            "Daily Returns": {"target": "Daily Returns", "features": None},
            "Volatility": {"target": "Volatility", "features": None},
            "TNX": {"target": "TNX", "features": None},
            "Treasury_Yield": {"target": "Treasury_Yield", "features": None},
            "SP500": {"target": "SP500", "features": None},
            "USDCAD=X": {"target": "USDCAD=X", "features": None},
        }
        self.scalers = {}
        self.transformers = {}
        self.interval = interval
        self.history = []  # New attribute for error correction
        self.risk_params = {
            "max_portfolio_risk": 0.05,  # 5% total portfolio risk
            "per_trade_risk": 0.01,  # 1% risk per trade
            "stop_loss_pct": 0.03,  # 3% trailing stop
            "take_profit_pct": 0.06,  # 6% take profit
            "max_sector_exposure": 0.4,  # 40% max energy sector exposure
            "daily_loss_limit": -0.03,  # -3% daily loss threshold
        }
        self.api = trading_client
        self.model_cache_dir = f"model_cache/{self.symbol}"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.data_hash = None
        self.forecast_record = {}

    # def _get_data_hash(self):
    #     """Generate unique hash of the training data"""
    #     return hashlib.sha256(pd.util.hash_pandas_object(self.data).h)

    def _get_data_hash(self):
        """Generate unique hash of the training data"""
        hash_series = pd.util.hash_pandas_object(self.data, index=True)
        hash_bytes = hash_series.values.tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()

    # def _load_cached_model(self, predictor):
    #     cache_path = f"{self.model_cache_dir}/{predictor}.pkl"
    #     if os.path.exists(cache_path):
    #         return joblib.load(cache_path)
    #     return None
    
    def _load_cached_model(self, predictor):
        cache_path = f"{self.model_cache_dir}/{predictor}.pkl"
        try:
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:  # At least 100 bytes
                return joblib.load(cache_path)
            else:
                print(f"Invalid cache file {cache_path} - regenerating")
                os.remove(cache_path)  # Clean up invalid cache
                return None
        except Exception as e:
            print(f"Cache load failed: {str(e)} - regenerating model") # HE START OF REGENERATING (KEY PART)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None

    def _save_model_cache(self, predictor, model):
        cache_path = f"{self.model_cache_dir}/{predictor}.pkl"
        joblib.dump(model, cache_path)


    def _load_cached_result(self, model_type, horizon, output_type):
        cache_path = f"{self.model_cache_dir}/{model_type}/Horizon_{horizon}/{output_type}.pkl"
        try:
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:  # At least 100 bytes
                return joblib.load(cache_path)
            else:
                print(f"Invalid cache result {cache_path} - regenerating")
                os.remove(cache_path)  # Clean up invalid cache
                return None
        except Exception as e:
            print(f"Cache load failed: {str(e)} - regenerating results") # HE START OF REGENERATING (KEY PART)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None


    
    def _save_result(self, model_type, forecast, horizon, output_type):
        """Save the forecast result to a cache file"""
    
        cache_path = f"{self.model_cache_dir}/{horizon}days_{output_type}_{model_type}.pkl"
        joblib.dump(forecast, cache_path)

    
    

    def _model_needs_retraining(self, predictor):
        if get_next_valid_date(self.data.index[-1]) != pd.Timestamp(date.today()):
            return True
        # # current_hash = self._get_data_hash()
        # hash_file = f"{self.model_cache_dir}/{predictor}.hash"

        # # If there is no hash file, create one
        # if not os.path.exists(hash_file):
        #     return True

        # with open(hash_file, "r") as f:
        #     saved_hash = f.read()
        # # Need to check if the data is updated
        # if current_hash != saved_hash:
        #     with open(hash_file, "w") as f:
        #         f.write(current_hash)
        #     return True
        return False

    def _compute_rsi(self, window=14):
        """Custom RSI implementation"""
        delta = self.data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        return 100 - (
            100 / (1 + (gain.rolling(window).mean() / loss.rolling(window).mean()))
        )

    def _compute_atr(self, window=14):
        """Average True Range"""
        high_low = self.data["High"] - self.data["Low"]
        high_close = (self.data["High"] - self.data["Close"].shift()).abs()
        low_close = (self.data["Low"] - self.data["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    # def calculate_position_size(self):
    #     """Calculate position size using Average True Range"""
    #     account = self.api.get_account()
    #     portfolio_value = float(account.equity)

    #     # Risk per trade calculation
    #     dollar_risk = portfolio_value * self.risk_params["per_trade_risk"]
    #     volatility_risk = atr * current_price  # ATR in dollar terms

    #     position_size = dollar_risk / volatility_risk
    #     return int(position_size)

    def get_sector_exposure(self):
        """Calculate current energy sector exposure"""
        positions = self.api.get_all_positions()
        energy_positions = [p for p in positions if p == "energy"]
        total_value = sum(float(p.market_value) for p in energy_positions)
        return total_value / float(self.api.get_account().equity)

    def generate_trading_signal(self, predictor, symbol):
        predictor.load_data()
        lookback = optimize_lookback(
            predictor.data.drop(columns="Close"),
            predictor.data["Close"],
            model=XGBRegressor(
                n_estimators=20,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            ),
            min_window=60,
            step_size=10,
            n_splits=5,
        )
        print(f"Optimal lookback window: {lookback}")
        predictor.data = predictor.data.iloc[-lookback:]
        features = [
            "Close",
            "MA_50",
            # "MA_200",
            # "High",
            # "Low",
            "MA_7",
            # "MA_21",
            "SP500",
            "TNX",
            "USDCAD=X",
            # "Tech",
            # "Fin",
            "VIX",
            "Energy",
            "rolling_min",
            "rolling_median",
            "rolling_sum",
            "rolling_ema",
            "rolling_25p",
            "rolling_75p",
            "RSI",
            "MACD",
            "ATR",
            "Upper_Bollinger",
            "Lower_Bollinger",
            "VWAP",
            "Volatility",
            "Daily Returns",
            "Williams_%R",
            "Momentum_Interaction",
            "Stochastic_%K",
            "Stochastic_%D",
            "Momentum_Score",
        ]  # Use same features as in notebook
        horizon = 5  # Prediction window

        predictor.prepare_models(features, horizon=horizon)
        forecast, _, _, _ = predictor.one_step_forward_forecast(
            predictors=features, model_type="arimaxgb", horizon=horizon
        )

        # Get latest prediction
        predicted_price = forecast["Close"].iloc[-1]
        self.forecast_record[symbol] = predicted_price
        # current_price = predictor.data['Close'].iloc[-1]
        if datetime.now().hour < 16 and datetime.now().hour > 9:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1m")
                .Close.iloc[-1]
                .values[0]
            )
        else:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1d")
                .Close.iloc[-1]
                .values[0]
            )

        # symbol = "AAPL"
        # trade = alpaca.get_latest_trade(symbol)
        # print(f"{symbol} Live Price: ${trade.price}")

        # Generate signal
        if predicted_price > current_price * 1.01:  # 1% threshold
            return "BUY"
        elif predicted_price < current_price * 0.99:
            return "SELL"
        else:
            return "HOLD"

    def generate_hft_signals(self, symbol, profit_target=0.005):
        """Generate immediate execution signals with tight spreads"""
        signals = []
        # get cached model
        if datetime.now().hour < 16 and datetime.now().hour > 9:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1m")
                .Close.iloc[-1]
                .values[0]
            )
        else:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1d")
                .Close.iloc[-1]
                .values[0]
            )

        # Calculate bid/ask spread
        bid_price = round(current_price * 0.99, 2)
        ask_price = round(current_price * 1.005, 2)

        # Profit targets
        sell_target = round(current_price * (1 + profit_target), 2)
        buy_target = round(current_price * (1 - profit_target), 2)

        # Existing position check
        try:
            # position = self.api.get_open_position(self.symbol)
            positions = [position.symbol for position in self.api.get_all_positions()]

            if symbol in positions:
                position = self.api.get_open_position(symbol)
                if float(position.unrealized_plpc) >= profit_target:
                    signals.append(("SELL", int(position.qty), sell_target))
                elif self.forecast_record[symbol] > current_price * 1.001:
                    print(f"Have position for {self.symbol}, but want to buy.")
                    signals.append(("BUY", int(position.qty), buy_target))

            else:  # No open position of the symbol
                if self.forecast_record[symbol] > current_price * 1.001:
                    print(f"No open position for {self.symbol}, but want to buy.")
                    signals.append(
                        (
                            "BUY",
                            round(self._calculate_position_size()),
                            buy_target,
                        )
                    )
                elif self.forecast_record[symbol] < current_price * 0.999:
                    print(f"No open position for {self.symbol}, but want to sell.")
                    signals.append(
                        (
                            "SELL",
                            round(self._calculate_position_size()),
                            sell_target,
                        )
                    )
        except Exception:
            pass

        # Add market making signals
        # signals.extend([
        #     ("BUY", self._calculate_position_size(), bid_price),
        #     ("SELL", self._calculate_position_size(), ask_price)
        # ])

        return signals

    # def _calculate_position_size(self):
    #     """Risk-managed position sizing"""
    #     account = self.api.get_account()
    #     return float(account.buying_power) * 0.01 / self.data["Close"].iloc[-1]

    def _calculate_position_size(self):
        """Ensure minimum quantity with fractional safety"""
        account = self.api.get_account()
        if datetime.now().hour < 16 and datetime.now().hour > 9:
            current_price = (
                yf.download(start=date.today(), tickers=self.symbol, interval="1m")
                .Close.iloc[-1]
                .values[0]
            )
        else:
            current_price = (
                yf.download(start=date.today(), tickers=self.symbol, interval="1d")
                .Close.iloc[-1]
                .values[0]
            )
        
        # Calculate dollar amount
        risk_amount = float(account.buying_power) * 0.01  # 1% risk
        size = risk_amount / current_price
        
        # Enforce minimum quantity rules
        if size < 0.5:  # Prevent tiny fractional orders
            return 1
        if 0.5 <= size < 1:
            return round(size, 4)  # Allow fractional shares
        else:
            return round(size, 2)  # Round to nearest whole share

    def execute_trade(self, signal):
        # Cancel stale orders every 2 minutes
        if datetime.now().minute % 2 == 0:
            self._cancel_old_orders()

        symbol = self.symbol
        # current_price = self.data['Close'].iloc[-1]
        if datetime.now().hour < 16 and datetime.now().hour > 9:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1m")
                .Close.iloc[-1]
                .values[0]
            )
        else:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1d")
                .Close.iloc[-1]
                .values[0]
            )
        atr = self.data["ATR"].iloc[-1]

        # Check daily loss limit
        if self.check_daily_loss():
            print("Daily loss limit hit - no trading allowed")
            return

        # Check sector exposure
        # if self.get_sector_exposure() >= self.risk_params['max_sector_exposure']:
        #     print("Max sector exposure reached")
        #     return

        # position_size = self.calculate_position_size(current_price, atr)
        position_size = round(self._calculate_position_size(current_price, atr))
        #  In case insufficient qty of shares

        # position_size = 20  # Placeholder for position size calculation

        if signal == "BUY":
            # Calculate dynamic stop levels based on ATR
            # stop_price = current_price - (0.2 * atr)
            # take_profit = current_price + (0.3 * atr)
            # limit_price = current_price + (0.1 * atr)
            # Round to 2 decimal places
            take_profit = round(current_price + (0.3 * atr), 2)
            # limit_price = round(current_price + (0.1 * atr), 2)
            stop_price = round(current_price - (0.2 * atr), 2)

            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=position_size,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    take_profit=take_profit, limit_price=round(take_profit * 1.01, 2)
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop_price, limit_price=round(stop_price * 0.99, 2)
                ),  # 1% trailing stop
            )

            # order = self.api.submit_order(
            #     order_data=market_order_data)
            # order
            # status = self.api.get_orders(order.id)
            # print(f"Order {order.id} status: {status}")

            try:
                order = self.api.submit_order(market_order_data)
                print(f"Order submitted: {order.id}")

                # Verify order status
                status = self.api.get_orders(order.id).status
                print(f"Order status: {status}")

                # Check fills
                if status == "filled":
                    print(f"Filled at avg price: {order.filled_avg_price}")
                else:
                    print("Order not filled - check price/quantity")
            except Exception as e:
                print(f"Order failed: {str(e)}")

        elif signal == "SELL":
            positions = self.api.get_all_positions()
            for p in positions:
                if p.symbol == symbol:
                    market_order_data_sell = MarketOrderRequest(
                        symbol=symbol,
                        qty=int(p.qty),
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        # stop_loss=StopLossRequest(stop_price=current_price - (1 * atr), limit_price=current_price + (1 * atr) * 1.01),
                        # take_profit=TakeProfitRequest(take_profit=current_price + (2 * atr), limit_price=current_price - (2 * atr) * 0.99)  # 1% trailing stop
                        # For short positions (SELL)
                        stop_loss=StopLossRequest(
                            stop_price=round(
                                current_price + (1 * atr), 2
                            ),  # Stop price above current for shorts
                            limit_price=round(current_price + (1 * atr) * 1.01, 2),
                        ),
                        take_profit=TakeProfitRequest(
                            take_profit=round(
                                current_price - (2 * atr), 2
                            ),  # Take profit below current for shorts
                            limit_price=round(current_price - (2 * atr) * 0.99, 2),
                        ),
                    )
                    # order = self.api.submit_order(
                    #     order_data=market_order_data_sell
                    # )
                    # market_order_data_sell
                    # status = self.api.get_orders(order.id)
                    # print(f"Order {order.id} status: {status}")
                    try:
                        order = self.api.submit_order(market_order_data_sell)
                        print(f"Order submitted: {order.id}")

                        # Verify order status
                        status = self.api.get_orders(order.id).status
                        print(f"Order status: {status}")

                        # Check fills
                        if status == "filled":
                            print(f"Filled at avg price: {order.filled_avg_price}")
                        else:
                            print("Order not filled - check price/quantity")
                    except Exception as e:
                        print(f"Order failed: {str(e)}")
            if symbol not in [p.symbol for p in positions]:
                print(f"No position found for {symbol} to sell. But want to short.")
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=position_size,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                    stop_loss=StopLossRequest(
                        stop_price=current_price - (1 * atr),
                        limit_price=current_price + (1 * atr) * 1.01,
                    ),
                    take_profit=TakeProfitRequest(
                        take_profit=current_price + (2 * atr),
                        limit_price=current_price - (2 * atr) * 0.99,
                    ),  # 1% trailing stop
                )

                try:
                    order = self.api.submit_order(req)
                    print(f"Order submitted: {order.id}")

                    # Verify order status
                    status = self.api.get_orders(order.id).status
                    print(f"Order status: {status}")

                    # Check fills
                    if status == "filled":
                        print(f"Filled at avg price: {order.filled_avg_price}")
                    else:
                        print("Order not filled - check price/quantity")
                except Exception as e:
                    print(f"Order failed: {str(e)}")

    # def execute_hft(self, symbol):
    #     """Execute HFT strategy with cached models"""
    #     # Get signals using cached models
    #     signals = self.generate_hft_signals(symbol=symbol)


    #     # Batch order processing
    #     orders = []
    #     for side, qty, price in signals:
    #         order_request = MarketOrderRequest(
    #             symbol=symbol,
    #             qty=abs(float(qty)),
    #             side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
    #             limit_price=price,
    #             type=OrderType.LIMIT,
    #             time_in_force=TimeInForce.GTC,
    #             order_class = OrderClass.BRACKET,
    #             take_profit=TakeProfitRequest(
    #                 limit_price=(
    #                     round(price * 1.005, 2)
    #                     if side == "BUY"
    #                     else round(price * 0.995, 2)
    #                 )
    #             ),
    #             stop_loss=StopLossRequest(
    #                 stop_price=(
    #                     round(price * 0.98, 2)
    #                     if side == "BUY"
    #                     else round(price * 1.02, 2)
    #                 ),
    #                 limit_price=(
    #                     round(price * 0.98, 2)
    #                     if side == "BUY"
    #                     else round(price * 1.02, 2)
    #                 ),
    #             ),
    #         )

    #         print(signals)
            
    #         self.api.submit_order(order_request)
    def execute_hft(self, symbol, manual=False):
        """Execute HFT strategy with cached models"""
        # Get signals using cached models
        signals = self.generate_hft_signals(symbol=symbol)
        technology_sector = list(yf.Sector("technology").top_companies.index)

        manual_signals = []
        if manual == True:
            if symbol in technology_sector:
               for side, qty, price in signals:
                    # manuall make side to be "BUY" 
                    manual_signals.append(("BUY", qty, price))
                    
                    
        orders = []
        if manual == True:
            signals = manual_signals
        for side, qty, price in signals:
            # Ensure quantity is positive and valid
            qty = abs(float(qty))
            if qty <= 0:
                print(f"Skipping invalid quantity: {qty}")
                continue
                
            # Set appropriate take profit and stop loss levels
            if side == "BUY":
                take_profit_price = round(price * 1.003, 2)
                stop_price = round(price * 0.985, 2)
                stop_limit_price = round(price * 0.98, 2)  # Slightly lower than stop price
            else:  # SELL
                take_profit_price = round(price * 0.997, 2)
                stop_price = round(price * 1.015, 2)
                stop_limit_price = round(price * 1.02, 2)  # Slightly higher than stop price
      
                    
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                limit_price=price,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(
                    limit_price=take_profit_price
                ),
                stop_loss=StopLossRequest(
                    stop_price=stop_price,
                    limit_price=stop_limit_price,
                ),
            )
            
            print(f"Attempting to submit {side} order for {qty} shares of {symbol} at {price}")
            
            try:
                order = self.api.submit_order(order_request)
                print(f"Order submitted: {order.id}")
                
                # Verify the order status
                status = self.api.get_orders(order.id).status
                print(f"Order status: {status}")
                
                orders.append(order)
            except Exception as e:
                print(f"Order submission failed: {str(e)}")
        
        return orders
   

    def _cancel_old_orders(self):
        """Cancel orders older than 2 minutes"""
        # orders = self.api.get_orders(filter=QueryOrderStatus.OPEN)
        orders = self.api.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )

        for order in orders:
            if (datetime.now(order.created_at.tzinfo) - order.created_at).seconds > 120:
                self.api.cancel_order_by_id(order.id)

    def check_daily_loss(self):
        """Check portfolio-wide daily loss limits"""
        account = self.api.get_account()
        daily_pnl = float(account.equity) - float(account.last_equity)

        if (
            daily_pnl / float(account.last_equity)
            < self.risk_params["daily_loss_limit"]
        ):
            # Liquidate all positions
            positions = self.api.list_positions()
            for p in positions:
                req = MarketOrderRequest(
                    symbol=p.symbol,
                    qty=p.qty,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.GTC,
                )
                res = self.api.submit_order(order_data=req)
                res

                # self.api.submit_order(
                #     symbol=p.symbol,
                #     qty=p.qty,
                #     side='sell',
                #     type='market',
                #     time_in_force='gtc'
                # )
            return True
        return False

    def load_data(self):
        """Load and prepare stock data with features"""
        # Add momentum-specific features
        window = 15  # Standard momentum window
        self.data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
        )
        self.data.columns = self.data.columns.get_level_values(0)  # Remove multi-index
        self.data.ffill()
        self.data.dropna()

        ### 1. Add rolling indicators
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()
        self.data["MA_7"] = self.data["Close"].rolling(window=7).mean()
        self.data["MA_21"] = self.data["Close"].rolling(window=21).mean()

        ### 2. Fourier transform
        data_FT = self.data.copy().reset_index()[["Date", "Close"]]
        close_fft = np.fft.fft(np.asarray(data_FT["Close"].tolist()))
        self.data["FT_real"] = np.real(close_fft)
        self.data["FT_img"] = np.imag(close_fft)

        # # Fourier Transformation is not used
        # fft_df = pd.DataFrame({'fft': close_fft})
        # fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        # fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        # fft_list = np.asarray(fft_df['fft'].tolist())
        # for num_ in [3, 6, 9, 100]:
        #     fft_list_m10 = np.copy(fft_list)
        #     fft_list_m10[num_:-num_] = 0
        #     complex_num = np.fft.ifft(fft_list_m10)
        #     self.data[f'Fourier_trans_{num_}_comp_real'] = np.real(complex_num)
        #     self.data[f'Fourier_trans_{num_}_comp_img'] = np.imag(complex_num)

        ### Fourier Transformation PCA
        X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
        pca = PCA(n_components=2)  # Keep top 2 components
        X_pca = pca.fit_transform(X_fft)
        for i in range(X_pca.shape[1]):
            self.data[f"Fourier_PCA_{i}"] = X_pca[:, i]

        ### 3. Add rolling statistics
        self.data["rolling_std"] = self.data["Close"].rolling(window=50).std()
        self.data["rolling_min"] = self.data["Close"].rolling(window=50).min()
        # self.data['rolling_max'] = self.data['Close'].rolling(window=window).max()
        self.data["rolling_median"] = self.data["Close"].rolling(window=50).median()
        self.data["rolling_sum"] = self.data["Close"].rolling(window=50).sum()
        self.data["rolling_var"] = self.data["Close"].rolling(window=50).var()
        self.data["rolling_ema"] = (
            self.data["Close"].ewm(span=50, adjust=False).mean()
        )  # Exponential Moving Average
        # Add rolling quantiles (25th and 75th percentiles)
        self.data["rolling_25p"] = self.data["Close"].rolling(window=50).quantile(0.25)
        self.data["rolling_75p"] = self.data["Close"].rolling(window=50).quantile(0.75)
        # Drop rows with NaN values (due to rolling window)
        self.data.dropna(inplace=True)
        stock_data.index.name = "Date"  # Ensure the index is named "Date"

        ### 4. Advanced Momentum
        self.data["RSI"] = self._compute_rsi(window=14)
        self.data["MACD"] = (
            self.data["Close"].ewm(span=12).mean()
            - self.data["Close"].ewm(span=26).mean()
        )
        ### 5. Williams %R
        high_max = self.data["High"].rolling(window).max()
        low_min = self.data["Low"].rolling(window).min()
        self.data["Williams_%R"] = (
            (high_max - self.data["Close"]) / (high_max - low_min)
        ) * -100

        ### 6. Stochastic Oscillator
        self.data["Stochastic_%K"] = (
            (self.data["Close"] - low_min) / (high_max - low_min)
        ) * 100
        self.data["Stochastic_%D"] = self.data["Stochastic_%K"].rolling(3).mean()

        ### 7. Momentum Divergence Detection
        self.data["Price_Change"] = self.data["Close"].diff()
        self.data["Momentum_Divergence"] = (
            (self.data["Price_Change"] * self.data["MACD"].diff()).rolling(5).sum()
        )

        ### 8. Volatility-adjusted Channels
        self.data["ATR"] = self._compute_atr(window=14)
        self.data["Upper_Bollinger"] = (
            self.data["MA_21"] + 2 * self.data["Close"].rolling(50).std()
        )
        self.data["Lower_Bollinger"] = (
            self.data["MA_21"] - 2 * self.data["Close"].rolling(50).std()
        )

        ### 9. Volume-based Features
        # self.data['OBV'] = self._compute_obv()
        self.data["VWAP"] = (
            self.data["Volume"]
            * (self.data["High"] + self.data["Low"] + self.data["Close"])
            / 3
        ).cumsum() / self.data["Volume"].cumsum()

        ### 10. Economic Indicators
        sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)["Close"]
        # Fetch S&P 500 Index (GSPC) and Treasury Yield ETF (IEF) from Yahoo Finance
        sp500 = sp500 - sp500.mean()
        tnx = yf.download(
            "^TNX", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        treasury_yield = yf.download(
            "IEF", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        exchange_rate = yf.download(
            "USDCAD=X", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        technology_sector = yf.download(
            "XLK", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        financials_sector = yf.download(
            "XLF", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        energy_sector = yf.download(
            "XLE", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]
        vix = yf.download(
            "^VIX", start=self.start_date, end=self.end_date, interval=self.interval
        )["Close"]

        # self.data["SP500"] = sp500
        # self.data["TNX"] = tnx
        # self.data["Treasury_Yield"] = treasury_yield
        # self.data["USDCAD=X"] = exchange_rate
        # self.data["Tech"] = technology_sector
        # self.data["Fin"] = financials_sector
        # self.data["VIX"] = vix
        # self.data["Energy"] = energy_sector

        economic_data = (
            pd.concat(
                [
                    sp500,
                    tnx,
                    treasury_yield,
                    exchange_rate,
                    technology_sector,
                    financials_sector,
                    vix,
                    energy_sector,
                ],
                axis=1,
                keys=[
                    "SP500",
                    "TNX",
                    "Treasury_Yield",
                    "USDCAD=X",
                    "Tech",
                    "Fin",
                    "VIX",
                    "Energy",
                ],
            )
            .reset_index()
            .rename(columns={"index": "Date"})
            # .dropna()
        )
        economic_data.columns = economic_data.columns.get_level_values(0)
        economic_data["Date"] = pd.to_datetime(economic_data["Date"])
        economic_data.set_index("Date", inplace=True)
        # Issue of Yfinance API of USDCAD=X
        # Fill missing values with the mean
        economic_data["USDCAD=X"] = economic_data["USDCAD=X"].fillna(
            economic_data["USDCAD=X"].mean()
        )

        # 11. Whether the next or previous day is a non-trading day
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
        economic_data["is_next_non_trading_day"] = economic_data.index.shift(
            -1, freq="1d"
        ).isin(schedule.index).astype(int) + economic_data.index.shift(
            1, freq="1d"
        ).isin(
            schedule.index
        ).astype(
            int
        )

        # Merge with stock data
        self.data = pd.merge(self.data, economic_data, on="Date", how="left")

        ### 12. Volatility and Momentum
        # self.data["Daily Returns"] = self.data["Close"].pct_change() # Percentage change
        self.data["Daily Returns"] = (
            self.data["Close"].pct_change(window) * 100
        )  # Percentage change in the standard window for the momentum
        self.data["Volatility"] = self.data["Daily Returns"].rolling(window=20).std()
        # Adaptive Momentum Score
        vol_weight = self.data["Volatility"] * 100
        self.data["Momentum_Score"] = (
            self.data["RSI"] * 0.4
            + self.data["Daily Returns"] * 0.3
            + self.data["Williams_%R"] * 0.3
        ) / (1 + vol_weight)
        # Drop rows with NaN values
        self.data["Momentum_Interaction"] = (
            self.data["RSI"] * self.data["Daily Returns"]
        )
        self.data["Volatility_Adj_Momentum"] = self.data["Momentum_Score"] / (
            1 + self.data["Volatility"]
        )
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].clip(lower=0.1)
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].clip(upper=10.0)
        self.data["Volatility_Adj_Momentum"] = self.data[
            "Volatility_Adj_Momentum"
        ].fillna(0.0)

        ### 13. Market Regime Detection by HMM
        # hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        # hmm.fit(self.data["Close"].pct_change().dropna().values.reshape(-1, 1))
        # # Predict hidden states
        # market_state = hmm.predict(
        #     self.data["Close"].pct_change().dropna().values.reshape(-1, 1)
        # )
        # hmm_sp = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        # hmm_sp.fit(self.data["SP500"].pct_change().dropna().values.reshape(-1, 1))
        # market_state_sp500 = hmm_sp.predict(
        #     self.data["SP500"].pct_change().dropna().values.reshape(-1, 1)
        # )
        # # Initialize the Market_State column
        # self.data["Market_State"] = np.zeros(len(self.data))
        # if (
        #     len(set(list(market_state))) != 1
        #     and len(set(list(market_state_sp500))) != 1
        # ):
        #     self.data["Market_State"][0] = 0
        #     self.data.iloc[1:]["Market_State"] = market_state + market_state_sp500

        # ### 14. Sentiment Analysis (Computationally expensive)
        # self.data["Market_Sentiment"] = 0.0
        # sentimement = MarketSentimentAnalyzer().get_historical_sentiment(
        #     self.symbol, self.data.shape[0]
        # )
        # self.data["Market_Sentiment"] = sentimement

        # Final cleaning
        self.data = self.data.dropna()
        if len(self.data) < 50:
            print("Not enough data to train the model.")
            raise ValueError("Not enough data to train the model.")

        return self

    def prepare_models(
        self, predictors: list[str], horizon, weight: bool = False, refit: bool = True
    ):
        """
        Prepare models for each predictor.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        horizon : int
            Number of days to forecast
        weight : bool
            Whether to apply feature weighting
        refit : bool
            Whether to refit models on full data
        """
        self.models = {}
        self.scalers = {}
        self.transformers = {}
        self.feature_importances = {}

        for predictor in predictors:
            cached_model = self._load_cached_model(predictor)
            needs_retrain = self._model_needs_retraining(predictor)

            # if cached_model and not needs_retrain:
            if cached_model and needs_retrain is False:
            # if cached_model:
                print(f"Using cached model for {predictor}")
                self.models[predictor] = cached_model
                continue

            # Select features excluding the current predictor
            features = [col for col in predictors if col != predictor]

            # Prepare data
            X = self.data[features].iloc[:-horizon,]
            y = self.data[predictor].iloc[:-horizon,]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # # Polynomial features
            # poly = PolynomialFeatures(degree=2)
            # X_train_poly = poly.fit_transform(X_train_scaled)
            # X_test_poly = poly.transform(X_test_scaled)

            # Train models
            models = {
                "linear": LinearRegression(),
                # "ridge": Ridge(alpha=1.0),
                # "polynomial": LinearRegression(),
                "arimaxgb": ARIMAXGBoost(),
            }

            # Fit models
            models["linear"].fit(X_train, y_train)
            # models["ridge"].fit(X_train_scaled, y_train)
            # models["polynomial"].fit(X_train_poly, y_train)
            models["arimaxgb"].fit(X_train, y_train)

            # Cache the trained models
            self._save_model_cache(predictor, models)
            print(f"Retrained and cached model for {predictor}")

            result = {}
            for name, model in models.items():

                if name == "linear":
                    y_pred = model.predict(X_test)
                    # 1 - (1 - model.score(X_test, y_test))
                # elif name == "ridge":
                #     y_pred = model.predict(scaler.transform(X_test))
                #     # 1 - (1 - model.score(X_test_scaled, y_test))
                # elif name == "polynomial":
                #     y_pred = model.predict(poly.transform(scaler.transform(X_test)))
                #     # 1 - (1 - model.score(X_test_poly, y_test))
                elif name == "arimaxgb":
                    y_pred = model.predict(X_test)

                # Compute adjusted R^2  # original one r2_score(y_test, y_pred)
                r2 = r2_score(y_true=y_test, y_pred=y_pred)
                adj_r2 = 1 - (1 - r2_score(y_true=y_test, y_pred=y_pred)) * (
                    X_test.shape[0] - 1
                ) / (X_test.shape[0] - X_test.shape[1] - 1)

                # Compute metrics
                rmse = root_mean_squared_error(y_test, y_pred)
                result[name] = {"rmse": rmse, "r2": r2}

                print(f"{predictor} - {name.capitalize()} Model:")
                print(f"  Test Mean Squared Error: {rmse:.4f}")
                print(f"  R² Score: {r2:.4f}")
                if "arimaxgb" in result:
                    if result["arimaxgb"]["r2"] != max(
                        [result[model]["r2"] for model in result]
                    ):
                        if predictor == "Close" and (result["arimaxgb"]["r2"] < 0.8):
                            raise ValueError(
                                "ARIMAXGBoost model failed to converge (r2 < 0.8). Please check your data period or model parameters."
                            )
            print(
                "-" * 50,
            )

            # Store models, scalers, and transformers
            self.models[predictor] = models
            self.scalers[predictor] = scaler
            # self.transformers[predictor] = poly

            if refit is True:
                # Refit models on full data
                refit_models = {
                    "linear": LinearRegression(),
                    # "ridge": Ridge(alpha=1.0),
                    # "polynomial": LinearRegression(),  # Ridge(alpha=1.0),
                    "arimaxgb": ARIMAXGBoost(),
                }
                refit_models["linear"].fit(X, y)
                # refit_models["ridge"].fit(scaler.transform(X), y)
                # refit_models["polynomial"].fit(poly.transform(scaler.transform(X)), y)
                refit_models["arimaxgb"].fit(X, y)
                self.models[predictor] = refit_models

                # Cache the trained models
                self._save_model_cache(predictor, refit_models)
                print(f"Retrained and cached refitted model for {predictor}")
            

    def one_step_forward_forecast(self, predictors: list[str], model_type, horizon):
        """
        Perform one-step forward predictions for all predictors with enhanced methods.

        Parameters:
        -----------
        predictors : List[str]
            List of predictor column names
        model_type : str
            one of the model types
        horizon : int
            Number of days to forecast

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Forecasted data and backtest data
        """
        
        cached_final_prediction = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="forecast")
        cached_final_backtest = self._load_cached_result(model_type=model_type,horizon=horizon, output_type="backtest")
        cached_final_raw_prediction = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="raw_forecast")
        cached_final_raw_backtest = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="raw_backtest")

        # if cached_model and not needs_retrain:
        if cached_final_prediction is not None:
            print(f"Using cached forecasts and backtests for {self.symbol}")
            return cached_final_prediction, cached_final_backtest, cached_final_raw_prediction, cached_final_raw_backtest   

        # Ensure models are prepared
        if not self.models:
            raise ValueError("Please run prepare_models() first")

        # Initialize prediction and backtest DataFrames
        prediction = self.data[predictors].copy().iloc[-horizon:].dropna()
        backtest = self.data[predictors].copy().iloc[:-horizon].dropna()
        observation = self.data[predictors].copy().dropna()

        # Initialize arrays for storing predictions
        pred_array = np.zeros((horizon, len(predictors)))
        raw_pred_array = np.zeros((horizon, len(predictors)))
        backtest_array = np.zeros((horizon, len(predictors)))
        raw_backtest_array = np.zeros((horizon, len(predictors)))

        # Create maps for quick lookup
        pred_dates = []
        backtest_dates = []
        predictor_indices = {p: i for i, p in enumerate(predictors)}

        # Initialize error correction mechanisms
        # 1. Base correction factors
        error_correction = {predictor: 1.0 for predictor in predictors}

        # 2. Feature-specific correction bounds
        price_vars = ["Open", "High", "Low", "Close"]
        bounds = {}
        for p in predictors:
            if p in price_vars:
                bounds[p] = (0.95, 1.05)  # Tighter bounds for prices
            elif p.startswith("MA_"):
                bounds[p] = (0.97, 1.03)  # Even tighter for moving averages
            else:
                bounds[p] = (0.6, 1.4)  # Wider for other indicators

        # 3. Initialize regime detection
        regime = "normal"  # Default regime
        price_changes = []

        # 4. Initialize Kalman filter parameters (simplified)
        kalman_gain = {p: 0.2 for p in predictors}
        error_variance = {p: 1.0 for p in predictors}

        # 5. Create ensembles of correction factors
        ensemble_corrections = {p: [0.935, 1.0, 1.035] for p in predictors}
        ensemble_weights = {p: np.array([1 / 3, 1 / 3, 1 / 3]) for p in predictors}

        # Calculate initial volatility (if Close is in predictors)
        if "Close" in predictors:
            close_history = observation["Close"].tail(20)
            returns = close_history.pct_change().dropna()
            current_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        else:
            current_volatility = 0.2  # Default volatility assumption

        # Helper functions
        def update_regime(prev_values, new_value):
            """Update market regime based on recent price action"""
            if len(prev_values) < 2:
                return "normal"

            # Calculate recent returns
            recent_returns = np.diff(prev_values) / prev_values[:-1]

            # Calculate volatility
            vol = np.std(recent_returns) * np.sqrt(252)

            # Detect trend
            trend = sum(1 if r > 0 else -1 for r in recent_returns)

            if vol > 0.4:  # High volatility threshold
                return "volatile"
            elif abs(trend) > len(recent_returns) * 0.7:  # Strong trend
                return "trending"
            else:
                return "mean_reverting"

        def adaptive_bounds(predictor, volatility, regime):
            """Calculate adaptive bounds based on volatility and regime"""
            base_lower, base_upper = bounds[predictor]

            # Adjust bounds based on regime
            if regime == "volatile":
                # Wider bounds during volatility
                lower = base_lower - 0.1
                upper = base_upper + 0.1
            elif regime == "trending":
                # Asymmetric bounds for trending markets
                if predictor in price_vars:
                    recent_trend = (
                        np.mean(price_changes[-5:]) if len(price_changes) >= 5 else 0
                    )
                    if recent_trend > 0:
                        # Uptrend - allow more upside correction
                        lower = base_lower
                        upper = base_upper + 0.1
                    else:
                        # Downtrend - allow more downside correction
                        lower = base_lower - 0.1
                        upper = base_upper
                else:
                    lower, upper = base_lower, base_upper
            else:
                # Default bounds
                lower, upper = base_lower, base_upper

            # Further adjust based on volatility
            vol_factor = min(1.0, volatility / 0.2)  # Normalize volatility
            lower -= 0.05 * vol_factor
            upper += 0.05 * vol_factor

            return max(0.5, lower), min(2.0, upper)  # Hard limits

        def apply_kalman_update(predictor, predicted, actual, step):
            """Apply Kalman filter update to correction factor"""
            # global kalman_gain, error_variance

            # Skip if we don't have actual to compare
            if actual is None:
                return error_correction[predictor]

            # Calculate prediction error
            pred_error = (actual - predicted) / actual if predicted != 0 else 0

            # Update error variance estimate (simplified)
            error_variance[predictor] = 0.7 * error_variance[predictor] + 0.3 * (
                pred_error**2
            )

            # Update Kalman gain
            k_gain = error_variance[predictor] / (error_variance[predictor] + 0.1)
            kalman_gain[predictor] = min(0.5, max(0.05, k_gain))  # Bounded gain

            # Exponentially reduce gain with forecast horizon
            horizon_factor = np.exp(-0.1 * step)
            effective_gain = kalman_gain[predictor] * horizon_factor

            # Calculate correction factor
            correction = 1.0 + effective_gain * pred_error

            return correction

        def enforce_constraints(pred_values, step):
            """Enforce cross-variable constraints"""
            if all(p in predictors for p in ["Open", "High", "Low", "Close"]):
                # Get indices
                o_idx = predictor_indices["Open"]
                h_idx = predictor_indices["High"]
                l_idx = predictor_indices["Low"]
                c_idx = predictor_indices["Close"]

                # Ensure High is highest
                highest = max(
                    pred_values[step, o_idx],
                    pred_values[step, c_idx],
                    pred_values[step, h_idx],
                )
                pred_values[step, h_idx] = highest

                # Ensure Low is lowest
                lowest = min(
                    pred_values[step, o_idx],
                    pred_values[step, c_idx],
                    pred_values[step, l_idx],
                )
                pred_values[step, l_idx] = lowest

            return pred_values

        # Main forecasting loop
        for step in range(horizon):
            # Get last known dates
            if step == 0:
                # last_pred_row = prediction.iloc[-1]
                # last_backtest_row = backtest.iloc[-1]
                # last_pred_date = last_pred_row.name
                # last_backtest_date = last_backtest_row.name

                last_pred_row = (
                    prediction.iloc[-horizon:].mean(axis=0)
                    if len(prediction) >= horizon
                    else prediction.iloc[-1]
                )
                last_backtest_row = (
                    backtest.iloc[-horizon:].mean(axis=0)
                    if len(backtest) >= horizon
                    else backtest.iloc[-1]
                )
                last_pred_date = prediction.iloc[-1].name
                last_backtest_date = backtest.iloc[-1].name

                # last_pred_row = prediction.iloc[-horizon:,].mean( axis=0)
                # last_backtest_row = backtest.iloc[-horizon:,].mean(axis=0)
            else:
                last_pred_date = pred_dates[-1]
                last_backtest_date = backtest_dates[-1]

            # Calculate next dates
            next_pred_date = get_next_valid_date(pd.Timestamp(last_pred_date))
            next_backtest_date = get_next_valid_date(pd.Timestamp(last_backtest_date))
            pred_dates.append(next_pred_date)
            backtest_dates.append(next_backtest_date)

            # # Step 1: Update market regime if we have Close
            if "Close" in predictors and step > 0:
                # Get recent close values
                close_idx = predictor_indices["Close"]
                if step > 1:
                    recent_close_vals = pred_array[:step, close_idx]
                    regime = update_regime(recent_close_vals, None)

                    # Also track price changes for trending analysis
                    if step > 1:
                        price_changes.append(
                            pred_array[step - 1, close_idx]
                            - pred_array[step - 2, close_idx]
                        )

            # Step 2: First handle Close price prediction (which others depend on)
            if "Close" in predictors:
                close_idx = predictor_indices["Close"]
                close_features = [col for col in predictors if col != "Close"]

                # Prepare input data - use last available information
                if step == 0:
                    # Use averaged input from last 'horizon' rows for prediction
                    if len(prediction) >= horizon:
                        # pred_input = (
                        #     prediction[close_features]
                        #     .iloc[-horizon:]
                        #     .mean(axis=0)
                        #     .values
                        # )

                        # Option 2: Use Weighted average of last 'horizon' rows
                        pred_input = np.average(
                            prediction[close_features].iloc[-horizon:],
                            axis=0,
                            weights=np.arange(1, horizon + 1)
                            / np.sum(np.arange(1, horizon + 1)),
                        )

                    else:
                        pred_input = last_pred_row[close_features].values
                    # for backtest
                    if len(backtest) >= horizon:
                        # # Option 1: Use average of last 'horizon' rows
                        # backtest_input = (
                        #     backtest[close_features].iloc[-horizon:].mean(axis=0).values
                        # )
                        # raw_backtest_input = (
                        #     backtest[close_features].iloc[-horizon:].mean(axis=0).values
                        # )

                        # Option 2: Use Weighted average of last 'horizon' rows
                        backtest_input = np.average(
                            backtest[close_features].iloc[-horizon:],
                            axis=0,
                            weights=np.arange(1, horizon + 1)
                            / np.sum(np.arange(1, horizon + 1)),
                        )
                        raw_backtest_input = np.average(
                            backtest[close_features].iloc[-horizon:],
                            axis=0,
                            weights=np.arange(1, horizon + 1)
                            / np.sum(np.arange(1, horizon + 1)),
                        )

                    else:
                        backtest_input = last_backtest_row[close_features].values
                        raw_backtest_input = last_backtest_row[close_features].values

                    # Option 2: May only use last row in case a huge change in price in the last horizon (The mean cannot reflect the change)
                    # so we use the last row instead of the mean
                    # Want data to be a dataframe

                    # pred_input = prediction[close_features].iloc[-1].values
                    # backtest_input = backtest[close_features].iloc[-1].values
                    # raw_backtest_input = backtest[close_features].iloc[-1].values

                else:  #  (step > 0)
                    # For subsequent steps, if we have enough predicted values, use their average
                    if step >= horizon:
                        # Use average of last 'horizon' predictions
                        pred_input = np.array(
                            [
                                np.mean(
                                    pred_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        raw_pred_input = np.array(
                            [
                                np.mean(
                                    raw_pred_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        backtest_input = np.array(
                            [
                                np.mean(
                                    backtest_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                        raw_backtest_input = np.array(
                            [
                                np.mean(
                                    raw_backtest_array[
                                        max(0, step - horizon) : step,
                                        predictor_indices[feat],
                                    ]
                                )
                                for feat in close_features
                            ]
                        )
                    else:
                        # If we don't have enough predictions yet, combine historical and predicted
                        pred_inputs = []
                        raw_pred_inputs = []
                        backtest_inputs = []
                        raw_backtest_inputs = []

                        for feat in close_features:
                            feat_idx = predictor_indices[feat]

                            # Get predicted values so far
                            pred_vals = (
                                pred_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            raw_pred_vals = (
                                raw_pred_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            backtest_vals = (
                                backtest_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )
                            raw_backtest_vals = (
                                raw_backtest_array[:step, feat_idx]
                                if step > 0
                                else np.array([])
                            )

                            # Calculate how many historical values we need
                            hist_needed = horizon - len(pred_vals)

                            if hist_needed > 0:
                                # Combine historical and predicted values
                                if feat in prediction.columns:
                                    pred_hist = (
                                        prediction[feat].iloc[-hist_needed:].values
                                    )
                                    all_pred_vals = np.concatenate(
                                        [pred_hist, pred_vals]
                                    )
                                    raw_all_pred_vals = np.concatenate(
                                        [pred_hist, raw_pred_vals]
                                    )
                                    pred_inputs.append(np.mean(all_pred_vals))
                                    raw_pred_inputs.append(np.mean(raw_all_pred_vals))
                                else:
                                    pred_inputs.append(0)  # Fallback

                                if feat in backtest.columns:
                                    backtest_hist = (
                                        backtest[feat].iloc[-hist_needed:].values
                                    )
                                    all_backtest_vals = np.concatenate(
                                        [backtest_hist, backtest_vals]
                                    )
                                    backtest_inputs.append(np.mean(all_backtest_vals))

                                    raw_backtest_hist = (
                                        backtest[feat].iloc[-hist_needed:].values
                                    )
                                    all_raw_backtest_vals = np.concatenate(
                                        [raw_backtest_hist, raw_backtest_vals]
                                    )
                                    raw_backtest_inputs.append(
                                        np.mean(all_raw_backtest_vals)
                                    )
                                else:
                                    backtest_inputs.append(0)  # Fallback
                                    raw_backtest_inputs.append(0)  # Fallback
                            else:
                                # We have enough predicted values already
                                pred_inputs.append(np.mean(pred_vals[-horizon:]))
                                backtest_inputs.append(
                                    np.mean(backtest_vals[-horizon:])
                                )
                                raw_backtest_inputs.append(
                                    np.mean(raw_backtest_vals[-horizon:])
                                )

                        pred_input = np.array(pred_inputs)
                        backtest_input = np.array(backtest_inputs)
                        raw_backtest_input = np.array(raw_backtest_inputs)

                # Apply model for Close price
                close_model = self.models["Close"][model_type]

                # Vector prediction for both datasets
                raw_pred_close = close_model.predict(pred_input.reshape(1, -1))[0]
                raw_backtest_close = close_model.predict(backtest_input.reshape(1, -1))[
                    0
                ]
                raw_backtest_raw_close = close_model.predict(
                    raw_backtest_input.reshape(1, -1)
                )[0]

                # Apply ensemble correction - weighted average of multiple correction factors
                ensemble_pred = 0
                ensemble_backtest = 0
                for i, corr in enumerate(ensemble_corrections["Close"]):
                    ensemble_pred += (
                        raw_pred_close * corr * ensemble_weights["Close"][i]
                    )
                    ensemble_backtest += (
                        raw_backtest_close * corr * ensemble_weights["Close"][i]
                    )

                # Apply the main error correction with adaptive bounds
                lower_bound, upper_bound = adaptive_bounds(
                    "Close", current_volatility, regime
                )
                close_correction = max(
                    lower_bound, min(upper_bound, error_correction["Close"])
                )

                pred_close = ensemble_pred * close_correction
                backtest_close = ensemble_backtest * close_correction

                # test if first prediction is way off
                if (
                    step == 0
                    and abs(
                        1 - backtest_close / self.data.copy().iloc[-horizon]["Close"]
                    )
                    >= 0.075
                ):
                    pred_close = 0.5 * (self.data.copy().iloc[-1]["Close"] + pred_close)
                    backtest_close = 0.5 * (
                        self.data.copy().iloc[-horizon]["Close"] + backtest_close
                    )
                # Store predictions
                pred_array[step, close_idx] = pred_close
                raw_pred_array[step, close_idx] = raw_pred_close
                backtest_array[step, close_idx] = backtest_close
                raw_backtest_array[step, close_idx] = raw_backtest_raw_close

                # Store predictions v2 mirror original code
                pred_array[step, close_idx] = raw_pred_close
                raw_pred_array[step, close_idx] = raw_pred_close
                backtest_array[step, close_idx] = raw_backtest_close
                raw_backtest_array[step, close_idx] = raw_backtest_raw_close

                # Update volatility estimate
                if step > 0:
                    prev_close = pred_array[step - 1, close_idx]
                    returns = (pred_close / prev_close) - 1
                    current_volatility = 0.94 * current_volatility + 0.06 * abs(
                        returns
                    ) * np.sqrt(252)

            # Step 3: Now handle other predictors
            for predictor in predictors:

                if predictor == "Close":
                    continue  # Already handled

                pred_idx = predictor_indices[predictor]

                # Special handling for MA calculations - direct calculation rather than model
                if predictor == "MA_50" and "Close" in predictors:
                    close_idx = predictor_indices["Close"]

                    # Get recent Close values to calculate MA
                    if step == 0:
                        # Use historical data for initial MA calculation
                        hist_close_pred = observation["Close"].values[-49:]
                        hist_close_backtest = backtest["Close"].values[-49:]
                        hist_close_raw_backtest = backtest["Close"].values[-49:]
                    else:
                        # Combine historical with predicted for later steps
                        pred_close_history = pred_array[:step, close_idx]
                        raw_pred_close_history = raw_pred_array[:step, close_idx]
                        backtest_close_history = backtest_array[:step, close_idx]
                        raw_backtest_close_history = raw_backtest_array[
                            :step, close_idx
                        ]

                        # Concatenate with appropriate historical data
                        if len(pred_close_history) < 49:
                            hist_close_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(49 - len(pred_close_history)) :
                                    ],
                                    pred_close_history,
                                ]
                            )
                            hist_close_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(49 - len(backtest_close_history)) :
                                    ],
                                    backtest_close_history,
                                ]
                            )
                            hist_close_raw_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(49 - len(raw_backtest_close_history)) :
                                    ],
                                    raw_backtest_close_history,
                                ]
                            )
                        else:
                            hist_close_pred = pred_close_history[-49:]
                            hist_close_backtest = backtest_close_history[-49:]
                            hist_close_raw_backtest = raw_backtest_close_history[-49:]

                    # Get current Close predictions
                    current_pred_close = pred_array[step, close_idx]
                    current_raw_pred_close = raw_pred_array[step, close_idx]
                    current_backtest_close = backtest_array[step, close_idx]
                    current_raw_close = raw_backtest_array[step, close_idx]

                    # Calculate MA_50 (vectorized)
                    ma50_pred = np.mean(np.append(hist_close_pred, current_pred_close))
                    ma50_raw_pred = np.mean(
                        np.append(hist_close_pred, current_raw_pred_close)
                    )
                    ma50_backtest = np.mean(
                        np.append(hist_close_backtest, current_backtest_close)
                    )
                    ma50_raw_backtest = np.mean(
                        np.append(hist_close_raw_backtest, current_raw_close)
                    )

                    # Store MA_50 values
                    pred_array[step, pred_idx] = ma50_pred
                    raw_pred_array[step, pred_idx] = ma50_raw_pred
                    backtest_array[step, pred_idx] = ma50_backtest
                    raw_backtest_array[step, pred_idx] = ma50_raw_backtest

                elif predictor == "MA_200" and "Close" in predictors:
                    close_idx = predictor_indices["Close"]

                    # Similar approach for MA_200
                    if step == 0:
                        hist_close_pred = observation["Close"].values[-199:]
                        hist_close_raw_pred = observation["Close"].values[-199:]
                        hist_close_backtest = backtest["Close"].values[-199:]
                        hist_close_raw_backtest = backtest["Close"].values[-199:]
                    else:
                        pred_close_history = pred_array[:step, close_idx]
                        raw_pred_close_history = raw_pred_array[:step, close_idx]
                        backtest_close_history = backtest_array[:step, close_idx]
                        raw_backtest_close_history = raw_backtest_array[
                            :step, close_idx
                        ]

                        if len(pred_close_history) < 199:
                            hist_close_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(199 - len(pred_close_history)) :
                                    ],
                                    pred_close_history,
                                ]
                            )
                            hist_close_raw_pred = np.concatenate(
                                [
                                    observation["Close"].values[
                                        -(199 - len(raw_pred_close_history)) :
                                    ],
                                    raw_pred_close_history,
                                ]
                            )
                            hist_close_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(199 - len(backtest_close_history)) :
                                    ],
                                    backtest_close_history,
                                ]
                            )
                            hist_close_raw_backtest = np.concatenate(
                                [
                                    backtest["Close"].values[
                                        -(199 - len(raw_backtest_close_history)) :
                                    ],
                                    raw_backtest_close_history,
                                ]
                            )
                        else:
                            hist_close_pred = pred_close_history[-199:]
                            hist_close_raw_pred = raw_pred_close_history[-199:]
                            hist_close_backtest = backtest_close_history[-199:]
                            hist_close_raw_backtest = raw_backtest_close_history[-199:]

                    current_pred_close = pred_array[step, close_idx]
                    current_raw_pred_close = raw_pred_array[step, close_idx]
                    current_backtest_close = backtest_array[step, close_idx]
                    current_raw_backtest_close = raw_backtest_array[step, close_idx]

                    ma200_pred = np.mean(np.append(hist_close_pred, current_pred_close))
                    ma200_raw_pred = np.mean(
                        np.append(hist_close_raw_pred, current_raw_pred_close)
                    )
                    ma200_backtest = np.mean(
                        np.append(hist_close_backtest, current_backtest_close)
                    )
                    ma200_raw_backtest = np.mean(
                        np.append(hist_close_raw_backtest, current_raw_backtest_close)
                    )

                    pred_array[step, pred_idx] = ma200_pred
                    raw_pred_array[step, pred_idx] = ma200_raw_pred
                    backtest_array[step, pred_idx] = ma200_backtest
                    raw_backtest_array[step, pred_idx] = ma200_raw_backtest

                elif predictor == "VIX" and "Close" in predictors:
                    # Use current volatility estimate directly
                    pred_array[step, pred_idx] = current_volatility
                    raw_pred_array[step, pred_idx] = current_volatility
                    backtest_array[step, pred_idx] = current_volatility
                    raw_backtest_array[step, pred_idx] = current_volatility

                else:
                    # Regular predictor - use model prediction
                    features = [col for col in predictors if col != predictor]

                    # Prepare input data using moving average approach
                    if step == 0:
                        # Use averaged input from last 'horizon' rows
                        if len(prediction) >= horizon:
                            # pred_input = (
                            #     prediction[features].iloc[-horizon:].mean(axis=0).values
                            # )

                            # Option 2: Use Weighted average of last 'horizon' rows
                            pred_input = np.average(
                                prediction[features].iloc[-horizon:],
                                axis=0,
                                weights=np.arange(1, horizon + 1)
                                / np.sum(np.arange(1, horizon + 1)),
                            )

                        else:
                            pred_input = last_pred_row[features].values

                        if len(backtest) >= horizon:
                            # backtest_input = (
                            #     backtest[features].iloc[-horizon:].mean(axis=0).values
                            # )
                            # Option 2: Use Weighted average of last 'horizon' rows
                            backtest_input = np.average(
                                backtest[features].iloc[-horizon:],
                                axis=0,
                                weights=np.arange(1, horizon + 1)
                                / np.sum(np.arange(1, horizon + 1)),
                            )
                        else:
                            backtest_input = last_backtest_row[features].values
                    else:
                        # For subsequent steps, similar approach as Close prediction
                        if step >= horizon:
                            # Use average of last 'horizon' predictions
                            pred_input = np.array(
                                [
                                    np.mean(
                                        pred_array[
                                            max(0, step - horizon) : step,
                                            predictor_indices[feat],
                                        ]
                                    )
                                    for feat in features
                                ]
                            )
                            backtest_input = np.array(
                                [
                                    np.mean(
                                        backtest_array[
                                            max(0, step - horizon) : step,
                                            predictor_indices[feat],
                                        ]
                                    )
                                    for feat in features
                                ]
                            )
                        else:
                            # If we don't have enough predictions yet, combine historical and predicted
                            pred_inputs = []
                            backtest_inputs = []

                            for feat in features:
                                feat_idx = predictor_indices[feat]

                                # Get predicted values so far
                                pred_vals = (
                                    pred_array[:step, feat_idx]
                                    if step > 0
                                    else np.array([])
                                )
                                backtest_vals = (
                                    backtest_array[:step, feat_idx]
                                    if step > 0
                                    else np.array([])
                                )

                                # Calculate how many historical values we need
                                hist_needed = horizon - len(pred_vals)

                                if hist_needed > 0:
                                    # Combine historical and predicted values
                                    if feat in prediction.columns:
                                        pred_hist = (
                                            prediction[feat].iloc[-hist_needed:].values
                                        )
                                        all_pred_vals = np.concatenate(
                                            [pred_hist, pred_vals]
                                        )
                                        pred_inputs.append(np.mean(all_pred_vals))
                                    else:
                                        pred_inputs.append(0)  # Fallback

                                    if feat in backtest.columns:
                                        backtest_hist = (
                                            backtest[feat].iloc[-hist_needed:].values
                                        )
                                        all_backtest_vals = np.concatenate(
                                            [backtest_hist, backtest_vals]
                                        )
                                        backtest_inputs.append(
                                            np.mean(all_backtest_vals)
                                        )
                                    else:
                                        backtest_inputs.append(0)  # Fallback
                                else:
                                    # We have enough predicted values already
                                    pred_inputs.append(np.mean(pred_vals[-horizon:]))
                                    backtest_inputs.append(
                                        np.mean(backtest_vals[-horizon:])
                                    )

                            pred_input = np.array(pred_inputs)
                            backtest_input = np.array(backtest_inputs)

                    # Get model predictions
                    model = self.models[predictor][model_type]

                    raw_pred = model.predict(pred_input.reshape(1, -1))[0]
                    raw_backtest = model.predict(backtest_input.reshape(1, -1))[0]

                    # Apply adaptive correction
                    lower_bound, upper_bound = adaptive_bounds(
                        predictor, current_volatility, regime
                    )
                    predictor_correction = max(
                        lower_bound, min(upper_bound, error_correction[predictor])
                    )

                    # Apply Kalman filter update for backtest
                    # (we can compare backtest with actual historical data)
                    actual_value = None
                    if (
                        next_backtest_date in self.data.index
                        and predictor in self.data.columns
                    ):
                        # actual_value = self.data.loc[next_backtest_date, predictor]
                        actual_value = self.data[self.data.index == next_backtest_date][
                            predictor
                        ].values[0]
                        kalman_correction = apply_kalman_update(
                            predictor, raw_backtest, actual_value, step
                        )
                        # Update the main correction factor with the Kalman result
                        error_correction[predictor] = (
                            0.7 * error_correction[predictor] + 0.3 * kalman_correction
                        )

                    # Apply correction
                    pred_value = raw_pred * predictor_correction
                    backtest_value = raw_backtest * predictor_correction
                    raw_backtest_value = raw_backtest
                    raw_pred_value = raw_pred

                    # # Store predictions
                    pred_array[step, pred_idx] = pred_value
                    raw_pred_array[step, pred_idx] = raw_pred_value
                    backtest_array[step, pred_idx] = backtest_value
                    raw_backtest_array[step, pred_idx] = raw_backtest_value

                    # Store predictions v2 mirror original code
                    # pred_array[step, pred_idx] = raw_pred
                    # backtest_array[step, pred_idx] = raw_backtest
                    # raw_backtest_array[step, pred_idx] = raw_backtest

            # Step 4: Apply cross-variable constraints
            pred_array = enforce_constraints(pred_array, step)
            backtest_array = enforce_constraints(backtest_array, step)

            # # Step 5: Update ensemble weights based on performance (for backtest)
            if step > 0 and step % 5 == 0:
                for predictor in predictors:
                    # Skip if we don't have enough data
                    if len(pred_dates) < 5:
                        continue

                    pred_idx = predictor_indices[predictor]

                    # Check if we have actual data to compare with backtest
                    actual_values = []
                    for date in backtest_dates[-5:]:
                        if date in self.data.index and predictor in self.data.columns:
                            actual_values.append(self.data.loc[date, predictor])

                    if len(actual_values) >= 3:  # Need enough data points
                        # Calculate errors for each ensemble member
                        errors = []
                        for i, corr in enumerate(ensemble_corrections[predictor]):
                            # Get predictions with this correction factor
                            corrected_preds = (
                                backtest_array[-len(actual_values) :, pred_idx] * corr
                            )

                            # Calculate mean squared error
                            mse = np.mean((corrected_preds - actual_values) ** 2)
                            errors.append(mse)

                        # Convert errors to weights (smaller error -> higher weight)
                        if max(errors) > min(errors):  # Avoid division by zero
                            inv_errors = 1.0 / (np.array(errors) + 1e-10)
                            new_weights = inv_errors / sum(inv_errors)

                            # Update weights with smoothing
                            ensemble_weights[predictor] = (
                                0.7 * ensemble_weights[predictor] + 0.3 * new_weights
                            )

        # Convert arrays to DataFrames
        prediction_df = pd.DataFrame(pred_array, columns=predictors, index=pred_dates)

        backtest_df = pd.DataFrame(
            backtest_array, columns=predictors, index=backtest_dates
        )

        raw_backtest_df = pd.DataFrame(
            raw_backtest_array, columns=predictors, index=backtest_dates
        )

        raw_prediction_df = pd.DataFrame(
            raw_pred_array, columns=predictors, index=pred_dates
        )

        # Concatenate with original data to include history
        final_prediction = pd.concat([prediction, prediction_df])
        final_raw_prediction = pd.concat([prediction, raw_prediction_df])
        final_backtest = pd.concat([backtest, backtest_df])
        final_raw_backtest = pd.concat([backtest, raw_backtest_df])

        # Cache the forecast results
        # if the cache path does not exist, create it
        
        self._save_result(model_type=model_type, forecast=final_prediction, horizon=horizon, output_type="forecast")
        self._save_result(model_type=model_type, forecast = final_raw_prediction, horizon=horizon, output_type="raw_forecast")
        self._save_result(model_type=model_type, forecast = final_backtest, horizon=horizon, output_type="backtest")
        self._save_result(model_type=model_type, forecast = final_raw_backtest, horizon=horizon, output_type="raw_backtest")

        print(
            f"Forecast result saved."
        )
        
  

        return (
            final_prediction,
            final_backtest,
            final_raw_prediction,
            final_raw_backtest,
        )

    def full_workflow(
        start_date,
        end_date,
        predictors=None,
        companies=None,
        stock_settings=None,
        model=None,
    ):
        """
        This function is used to output the prediction of the stock price for the future based on the stock price data from the start date to the end date.

        Args:
        start_date (str): The start date of the stock price data
        end_date (str): The end date of the stock price data
        predictors (list): The list of predictors used to predict the stock price
        companies (list): The list of company names of the stocks
        stock_settings (dict): The dictionary of the stock settings
        """
        # np.random.seed(42)
        default_horizons = [10, 12, 15]
        default_weight = False
        default_refit = True
        default_model = "arimaxgb"
        if companies is None:
            companies = ["AXP"]
        for company in companies:
            prediction_dataset = StockPredictor(
                company,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )
            prediction_dataset.load_data()

            if predictors is None:
                predictors = (
                    [
                        # "Market_State",
                        "Close",
                        # "MA_50",
                        # "MA_200",
                        "MA_7",
                        "MA_21",
                        "SP500",
                        "TNX",
                        # "USDCAD=X",
                        "Tech",
                        "Fin",
                        "VIX",
                        # "FT_real",
                        # "FT_img",
                    ]
                    + [
                        "rolling_min",
                        "rolling_median",
                        "rolling_sum",
                        "rolling_ema",
                        "rolling_25p",
                        "rolling_75p",
                    ]
                    + ["RSI", "MACD", "ATR", "Upper_Bollinger", "Lower_Bollinger"]
                    + [  # "Volatility"
                        # 'Daily Returns',
                        # 'Williams_%R',
                        "Momentum_Interaction",
                        "Volatility_Adj_Momentum",
                        "Stochastic_%K",
                        "Stochastic_%D",
                        "Momentum_Score",
                    ]
                )

            predictors = predictors

            predictor = prediction_dataset
            if stock_settings is not None and (
                len(stock_settings) != 0 and company in stock_settings
            ):
                # Use custom settings for the stock
                settings = stock_settings[company]
                horizons = settings["horizons"]
                weight = settings["weight"]
            else:
                # Use default settings for other stocks
                horizons = default_horizons
                weight = default_weight

            for horizon in horizons:
                prediction_dataset.prepare_models(
                    predictors, horizon=horizon, weight=weight, refit=default_refit
                )
                # prediction_dataset._evaluate_models('Close')
                if model is None:
                    pred_model = default_model
                else:
                    pred_model = model
                (
                    prediction,
                    backtest,
                    raw_prediction,
                    raw_backtest,
                ) = predictor.one_step_forward_forecast(  # final_prediction, final_backtest, final_raw_prediction, final_raw_backtest
                    predictors, model_type=pred_model, horizon=horizon
                )
                # print(prediction)
                # print(backtest)
                first_day = pd.to_datetime(
                    end_date - timedelta(days=int(round(1.5 * horizon)))
                )

                backtest_mape = mean_absolute_percentage_error(
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ],
                    backtest[backtest.index >= first_day].Close,
                )
                print("MSE of backtest period vs real data", backtest_mape)
                print("Horizon: ", horizon)
                print(
                    "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
                )
                if horizon <= 20:
                    if backtest_mape > 0.15:
                        continue
                else:
                    if backtest_mape > 0.30:
                        continue

                # Data Viz (Not that key)
                plt.figure(figsize=(12, 6))

                # first_day = pd.to_datetime(end_date - timedelta(days=5 + horizon))

                plt.plot(
                    prediction[
                        prediction.index >= prediction_dataset.data.iloc[-1].name
                    ].index,
                    prediction[
                        prediction.index >= prediction_dataset.data.iloc[-1].name
                    ].Close,
                    label="Prediction",
                    color="blue",
                )
                plt.plot(
                    raw_prediction[raw_prediction.index >= first_day].index,
                    raw_prediction[raw_prediction.index >= first_day].Close,
                    label="Raw Prediction",
                    color="green",
                )

                plt.plot(
                    backtest[backtest.index >= first_day].index,
                    backtest[backtest.index >= first_day].Close,
                    label="Backtest",
                    color="red",
                )
                plt.plot(
                    raw_backtest[raw_backtest.index >= first_day].index,
                    raw_backtest[raw_backtest.index >= first_day].Close,
                    label="Raw Backtest",
                    color="orange",
                )
                plt.plot(
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ].index,
                    prediction_dataset.data.Close[
                        prediction_dataset.data.index >= first_day
                    ],
                    label="Actual",
                    color="black",
                )
                # cursor(hover=True)
                plt.title(
                    f"Price Prediction ({prediction_dataset.symbol}) (horizon = {horizon}) (weight = {weight}) (refit = {default_refit}) (model = {pred_model})"
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


# Example usage
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", start_date="2020-01-01")
