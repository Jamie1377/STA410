from stock_prediction.utils import seed_everything

seed_everything(42)
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import yfinance as yf

from pytickersymbols import PyTickerSymbols
import pandas as pd
from datetime import date, timedelta, datetime
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
paper = True
# DO not change this
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None

# Alpaca API imports - make optional
import os
import requests
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopLimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        GetOrdersRequest,
        ClosePositionRequest
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderType,
        OrderClass,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest, CryptoLatestTradeRequest
    ALPACA_AVAILABLE = True
except ImportError:
    # Use alpaca_trade_api as fallback or disable alpaca functionality
    ALPACA_AVAILABLE = False
    print("Alpaca trading API not available. Some features may be disabled.")


if ALPACA_AVAILABLE:

    api_key = pd.read_json("API_KEYs.json",orient="index").iloc[0].values[0]
    secret_key = pd.read_json("API_KEYs.json",orient="index").iloc[1].values[0]
    data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    crypto_data_client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
else:
    data_client = None
    crypto_data_client = None
    trading_client = None


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
            "per_trade_risk": 0.025,  # 2.5% risk per trade
            "stop_loss_pct": 0.03,  # 3% trailing stop
            "take_profit_pct": 0.003,  # 1.5% take profit
            "max_sector_exposure": 0.4,  # 40% max energy sector exposure
            "daily_loss_limit": -0.2,  # -3% daily loss threshold
        }
        self.api = trading_client
        self.data_client = data_client
        self.crypto_data_client = crypto_data_client
        self.model_cache_dir = f"model_cache/{self.symbol}"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.data_hash = None
        self.forecast_record = {}


################################################################################


    def _get_data_hash(self):
        """Generate a stable hash of the training data"""
        # Create a more deterministic representation of the data
        data_copy = self.data.copy()
        
        # Round numeric columns to reduce floating point variation
        for col in data_copy.select_dtypes(include=['float']).columns:
            data_copy[col] = data_copy[col].round(4)
        
        # Get key identifiers that would determine if retraining is needed
        start_date = str(data_copy.index[0].date())
        end_date = str(data_copy.index[-1].date())
        num_rows = len(data_copy)
        columns = ','.join(sorted(data_copy.columns))

        # Create a deterministic string representation
        hash_input = f"{start_date}_{end_date}_{num_rows}_{columns}"
        
        # Add some data sampling to detect actual data changes
        # Sample a few values from the beginning, middle, and end of the dataset
        if num_rows > 10:
            sample_indices = [0, num_rows//4, num_rows//2, 3*num_rows//4, num_rows-1]
            for idx in sample_indices:
                hash_input += "_" + str(data_copy.iloc[idx]['Close'])
        
        # Generate hash
        return hashlib.sha256(hash_input.encode()).hexdigest()

    
    def _get_predictor_param_hash(self, params):
        """Generate a hash of the model parameters"""
        param_str = "_".join([f"{k}:{v}" for k, v in sorted(params.items())])
        return hashlib.sha256(param_str.encode()).hexdigest()
    

    def _get_model_cache_key(self, predictor, horizon, other_params=None):
        """Generate a unique cache key based on predictor, horizon and parameters"""
        params_str = ""
        if other_params:
            # Sort params to ensure consistent order
            params_str = "_" + "_".join(f"{k}_{v}" for k, v in sorted(other_params.items()))
        
        return f"{predictor}_{horizon}days{params_str}"
        

    def _get_forecast_hash(self, forecast):
        """Generate unique hash of the forecast data"""
        # Round numeric values to reduce floating point variation
        numeric_cols = forecast.select_dtypes(include=['number']).columns
        forecast_copy = forecast.copy()
        hash_series = pd.util.hash_pandas_object(forecast_copy, index=True)
        hash_bytes = hash_series.values.tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()


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
        hash_file = f"{self.model_cache_dir}/{predictor}.hash"
        joblib.dump(model, cache_path)
        current_hash = self._get_data_hash()
        with open(hash_file, "w", encoding='utf-8') as f:
            f.write(current_hash)


    def _model_needs_retraining(self, predictor, crypto=True):
        """Check if model needs retraining based on date and data hash"""
        # Check if today is a new trading day
        # if  crypto == False:
        #     if get_next_valid_date(self.data.index[-1]) != pd.Timestamp(date.today()):
        #         print(f"New trading day detected - retraining {predictor}")
        #         return True
            
        # Calculate current data hash
        current_hash = self._get_data_hash()
        print(f"Current hash: {current_hash}")
        hash_file = f"{self.model_cache_dir}/{predictor}.hash"
        
        # If hash file doesn't exist, create it and retrain
        if not os.path.exists(hash_file):
            with open(hash_file, "w", encoding='utf-8') as f:
                f.write(current_hash)
            print(f"No previous hash found - retraining {predictor}")
            return True
        
        # Compare current hash with saved hash
        with open(hash_file, "r", encoding='utf-8') as f:
                # saved_hash = joblib.load(f)  
                saved_hash = str(f.read().strip())  #rb
                print(f"Saved hash: {saved_hash}")
        
        if current_hash not in saved_hash:
            print(f"Data has changed - retraining {predictor}")
            # Only update hash file AFTER successful training, not here
            return True
            
        print(f"No changes detected - using cached model for {predictor}")
        return False

    
    def _load_cached_result(self, model_type, horizon, output_type):
        """Load cached forecast result if valid"""
        cache_path = f"{self.model_cache_dir}/{horizon}days_{output_type}_{model_type}.pkl"
        
        # First check if we need to regenerate based on data changes
        if not self.forecast_needs_reoutput(horizon, output_type, model_type):
            try:
                if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:  # At least 100 bytes
                    print(f"Using cached forecast for {horizon}days_{output_type}_{model_type}")
                    return joblib.load(cache_path)
            except Exception as e:
                print(f"Cache load failed: {str(e)}")
        
        # Either needs regeneration or load failed
        print(f"Regenerating forecast for {horizon}days_{output_type}_{model_type}")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None


###############################################################################################
   
   
    # Results
    def _save_result(self, model_type, forecast, horizon, output_type):
        """Save the forecast result to a cache file"""
        cache_path = f"{self.model_cache_dir}/{horizon}days_{output_type}_{model_type}.pkl"
        hash_file = f"{self.model_cache_dir}/{horizon}days_{output_type}_{model_type}.hash"
        
        # Save the forecast
        joblib.dump(forecast, cache_path)
   

        
        # Update the hash file
        current_hash = self._get_data_hash()
        # with open(hash_file, "w") as f:
        #     f.write(current_hash)
        # Save the hash 
        with open(hash_file, "w", encoding='utf-8') as f:
            # joblib.dump(current_hash, f)
            f.write(current_hash)
        print(f"Saved forecast for {horizon}days_{output_type}_{model_type}")


    def forecast_needs_reoutput(self, horizon, output_type, model_type):
        """Check if forecast needs reoutput based on date and data hash"""
        # Check if today is a new trading day

        # if self.interval == "1d":
        #     if get_next_valid_date(self.data.index[-1]) != pd.Timestamp(date.today()):
        #         print(f"New trading day detected - reoutput {output_type}")
        #         return True

        # Calculate current data hash
        current_hash = self._get_data_hash()
        hash_file = f"{self.model_cache_dir}/{horizon}days_{output_type}_{model_type}.hash"  # Fixed extension
        
        # If hash file doesn't exist, create it and regenerate
        if not os.path.exists(hash_file):
            print(f"No previous hash found - reoutput {output_type}")
            return True
        
        # Compare current hash with saved hash
        try:
            with open(hash_file, "r", encoding='utf-8') as f:
                saved_hash = str(f.read().strip())  # rb
                # saved_hash = joblib.load(f)  # rb
        except Exception as e:
            print(f"Hash load failed: {str(e)}")
        
        if current_hash not in saved_hash:
            print(f"Data has changed - reoutput {output_type}")
            return True
            
        return False

################################################################################# Utility functions


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
    

    def create_hqm_stocks(start_date, end_date = None, sector = "technology"):
        """Create a list of stocks with high momentum
        Args:
            start_date (str): Start date for data
            end_date (str): End date for data
            sector (str): Sector to filter stocks
        """
        technology_sector = list(yf.Sector(sector).top_companies.index)
        stock_data = PyTickerSymbols()
        nasdaq_tickers = stock_data.get_stocks_by_index('NASDAQ 100')  # Corrected index name
        sp500_tickers = stock_data.get_stocks_by_index('S&P 500')
        nasdaq_tickers_list = [stock['symbol'] for stock in nasdaq_tickers]
        sp500_tickers_list = [stock['symbol'] for stock in sp500_tickers]
        nasdaq_and_top_tech_tickers = list(set(nasdaq_tickers_list + technology_sector))
        table = yf.download(nasdaq_and_top_tech_tickers, start=start_date, end=end_date, interval="1d", group_by='tickers')
        hqm_df = pd.DataFrame(np.zeros((len(table.columns)//5, 5)), columns=['Symbol', 'Diff_21', 'Diff_42', 'Diff_63', 'HQM_Score'])
        i = 0
        for sym, col in (table.columns):
            if col == 'Close':
                diff_21 = table[sym]['Close'].iloc[-1] - table[sym]['Close'].shift(21).iloc[-1]
                percent_diff_21 = diff_21 / table[sym]['Close'].shift(21).iloc[-1]

                diff_42 = table[sym]['Close'].iloc[-1] - table[sym]['Close'].shift(42).iloc[-1]
                percent_diff_42 = diff_42 / table[sym]['Close'].shift(42).iloc[-1]

                diff_63 = table[sym]['Close'].iloc[-1] - table[sym]['Close'].shift(63).iloc[-1]
                percent_diff_63 = diff_63 / table[sym]['Close'].shift(63).iloc[-1]

                hqm_score = np.mean([percent_diff_21, percent_diff_42, percent_diff_63])

                hqm_df.iloc[i, 1] = diff_21
                hqm_df.iloc[i, 2] = diff_42
                hqm_df.iloc[i, 3] = diff_63
                hqm_df.iloc[i, 4] = hqm_score
                hqm_df.iloc[i, 0] = sym
                i += 1

        hqm_df = pd.DataFrame(hqm_df, columns=['Symbol', 'Diff_21', 'Diff_42', 'Diff_63', 'HQM_Score'])
        hqm_df.sort_values(by='HQM_Score', ascending=False, inplace=True)
        return hqm_df


    def get_sector_exposure(self):
        """Calculate current energy sector exposure"""
        positions = self.api.get_all_positions()
        energy_positions = [p for p in positions if p == "energy"]
        total_value = sum(float(p.market_value) for p in energy_positions)
        return total_value / float(self.api.get_account().equity)


    def generate_trading_signal(self, symbol, horizon= 5):
        """Generate trading signal using cached models"""
        self.load_data()
        lookback = optimize_lookback(
            self.data.drop(columns="Close"),
            self.data["Close"],
            model=XGBRegressor(
                n_estimators=20,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            ),
            min_window=60,
            step_size=2,
            n_splits=5,
        )
        print(f"Optimal lookback window: {lookback}")
        self.data = self.data.iloc[-lookback:]
        features = [
            "Close",
            # "MA_50",
            # "MA_200",
            # "High",
            # "Low",
            # "MA_7",
            "MA_21",
            "SP500",
            "TNX",
            # "USDCAD=X",
            "Tech",
            # "Fin",
            # "VIX",
            # "Energy",
            # "rolling_min",
            # "rolling_median",
            # "rolling_sum",
            "rolling_ema",
            "rolling_25p",
            # "rolling_75p",
            "RSI",
            "MACD",
            "ATR",
            "Upper_Bollinger",
            # "Lower_Bollinger",
            "VWAP",
            # "Volatility",
            "Daily Returns",
            "Williams_%R",
            "Momentum_Interaction",
            "Stochastic_%K",
            "Stochastic_%D",
            "Momentum_Score",
        ]  # Use same features as in notebook
        horizon = horizon  # Prediction window

        cached_results = self._load_cached_result(model_type='arimaxgb', horizon=horizon, output_type='forecast')
        needs_reoutput = self.forecast_needs_reoutput(horizon=horizon, output_type='forecast', model_type='arimaxgb')
        cached_results = pd.DataFrame(cached_results)
        print(f"Cached results tail: {cached_results.tail()}, data type: {type(cached_results)}")
        # if cached_model and not needs_retrain:
        if  len(cached_results) > 0: #and needs_reoutput is False:
            print(f"Using cached results for {symbol} prediction")
            forecast = cached_results
        else: # Regenerate model
            print(f"Regenerating model for {symbol} prediction")
            self.prepare_models(features, horizon=horizon)
            forecast, _, _, _ = self.one_step_forward_forecast(
                predictors=features, model_type="arimaxgb", horizon=horizon
            )


        # Get latest prediction
        
        predicted_price_day_1 = forecast["Close"].iloc[-horizon] 
        predicted_price_last_day = forecast["Close"].iloc[-1]
    
        alpaca_symbol = symbol.replace('-','/')  # Remove the last 4 characters
        self.forecast_record[alpaca_symbol] = predicted_price_last_day
        # current_price = predictor.data['Close'].iloc[-1]
        # if datetime.now().hour < 16 and datetime.now().hour > 9 and datetime.now().minute > 30:
        #     current_price = (
        #         yf.download(start=date.today(), tickers=symbol, interval="1m")
        #         .Close.iloc[-1]
        #         .values[0]
        #     )
        # elif datetime.now().hour < 9 and datetime.now().hour >= 0 and datetime.now().minute < 30:
        #     current_price = (
        #         yf.download(start=date.today()-pd.Timedelta(days=3), tickers=symbol, interval="1m")
        #         .Close.iloc[-1]
        #         .values[0]
        #     )
        # else:
        #     current_price = (
        #         yf.download(start=date.today(), tickers=symbol, interval="1d")
        #         .Close.iloc[-1]
        #         .values[0]
        #     )

        
        # Generate signal (Trend of the forecast)
        if predicted_price_last_day >=  predicted_price_day_1 * 1.005:  # 1% threshold
            return "BUY"
        elif predicted_price_last_day <= predicted_price_day_1 * 0.995:
            return "SELL"
        else:
            return "HOLD"


    def generate_hft_signals(self, symbol, profit_target=0.005):
        """Generate immediate execution signals with tight spreads"""
        signals = []
        alpaca_symbol = symbol.replace('-','/')  # Remove the last 4 characters
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
            latest_trade = data_client.get_stock_latest_trade(request)
        except Exception as e:
            print(f"Not Stock but Crypto")
            request = CryptoLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
            latest_trade = crypto_data_client.get_crypto_latest_trade(request)
      
        current_price = latest_trade[alpaca_symbol].price
        print(f"Current price: {current_price}")

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
                position = self.api.get_open_position(alpaca_symbol)
                if float(position.unrealized_plpc) >= profit_target:
                    signals.append(("SELL", int(position.qty), buy_target))
                elif self.forecast_record[symbol] > current_price * 1.001:
                    print(f"Have position for {symbol}, but want to buy and add on.")
                    signals.append(("BUY", int(position.qty), sell_target))

            else:  # No open position of the symbol
                if self.forecast_record[alpaca_symbol] > current_price * 1.001:
                    print(f"No open position for {alpaca_symbol}, but want to buy.")
                    signals.append(
                        (
                            "BUY",
                            round(self._calculate_position_size()),
                            sell_target,
                        )
                    )
                elif self.forecast_record[alpaca_symbol] < current_price * 0.999:
                    print(f"No open position for {self.symbol}, but want to sell.")
                    signals.append(
                        (
                            "SELL",
                            round(self._calculate_position_size()),
                            buy_target,
                        )
                    )
        except Exception:
            pass
        print(f"Generated hft signals: {signals}")
        print(f"forecast_record: {self.forecast_record}")
        return signals


    def generate_reverse_hft_signals(self, symbol, profit_target=0.005):
        """Generate reverse immediate of generation of hft signals"""
        hft_signals = self.generate_hft_signals(symbol, profit_target)
        reverse_signals = []
        for signal in hft_signals:
            if signal[0] == "BUY":
                reverse_signals.append(("SELL", signal[1], signal[2]))
            elif signal[0] == "SELL":
                reverse_signals.append(("BUY", signal[1], signal[2]))
        print(f"Generated reverse hft signals: {reverse_signals}")
        return reverse_signals


    def _calculate_position_size(self):
        """Ensure minimum quantity with fractional safety"""
        account = self.api.get_account()
        alpaca_symbol = self.symbol.replace('-','/')  # Remove the last 4 characters
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
            latest_trade = data_client.get_stock_latest_trade(request)
        except Exception as e:
            request = CryptoLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
            latest_trade = crypto_data_client.get_crypto_latest_trade(request)
        current_price = latest_trade[alpaca_symbol].price

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
        if datetime.now().hour < 16 and datetime.now().hour > 9 and datetime.now().minute > 30:
            current_price = (
                yf.download(start=date.today(), tickers=symbol, interval="1m")
                .Close.iloc[-1]
                .values[0]
            )
        elif datetime.now().hour < 9 and datetime.now().hour >= 0 and datetime.now().minute < 30:
            current_price = (
                yf.download(start=date.today()-pd.Timedelta(days=3), tickers=symbol, interval="1m")
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

        position_size = round(self._calculate_position_size(current_price, atr))

        if signal == "BUY":
            take_profit = round(current_price + (0.3 * atr), 2)
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

    def execute_hft(self, symbol, manual=False, crypto=False):
        """Execute HFT strategy with cached models"""
        # Get signals using cached models
        signals = self.generate_hft_signals(symbol=symbol)
        signals = signals + self.generate_reverse_hft_signals(symbol=symbol, profit_target=0.002)
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
                take_profit_price = round(price * 1.003 , 2)
                take_profit_price = max(take_profit_price, price + 0.01)  # Ensure it's higher than the current price
                stop_price = round(price * 0.985, 2)
                stop_limit_price = round(price * 0.98, 2)  # Slightly lower than stop price
                print(f"BUY order: take_profit_price {take_profit_price}, stop_price {stop_price}, stop_limit_price{stop_limit_price}")
            else:  # SELL
                take_profit_price = round(price * 0.997-0.01, 2) 
                take_profit_price = min(take_profit_price, price - 0.01)
                stop_price = round(price * 1.015, 2)
                stop_limit_price = round(price * 1.02, 2)  # Slightly higher than stop price
                print(f"SELL order: take_profit_price {take_profit_price}, stop_price {stop_price}, stop_limit_price{stop_limit_price}")

           
            
            alpaca_symbol = symbol.replace('-','/')
            print(f"Symbol for order: {alpaca_symbol}")
            if crypto == True:
                req = CryptoLatestTradeRequest(symbol_or_symbols=alpaca_symbol)
                possible_size = dict(dict(self.crypto_data_client.get_crypto_snapshot(req)[alpaca_symbol])['latest_trade'])['size']
                print(f"Available counts: {possible_size}")
                try: 
                    alpaca_symbol = symbol.replace('-','/')
                    # alpaca_symbol = symbol.replace('-','/')
                    order_request = MarketOrderRequest(
                        symbol=alpaca_symbol, #symbol
                        qty=qty,
                        side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                        limit_price=price,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.GTC,
                        # order_class=OrderClass.SIMPLE,
                        take_profit=TakeProfitRequest(
                            limit_price=take_profit_price
                        ),
                        stop_loss=StopLossRequest(
                            stop_price=stop_price,
                            # limit_price=stop_limit_price,
                        ),
                    )
                except Exception as e:
                    print(f"Error in order request: {str(e)}")
                    alpaca_symbol = symbol.replace('-','/')
                    # alpaca_symbol = symbol.replace('-','/')
                    order_request = MarketOrderRequest(
                        symbol=alpaca_symbol, #symbol
                        qty=possible_size,
                        side=OrderSide.SELL if side == "SELL" else OrderSide.BUY,
                        limit_price=price,
                        type=OrderType.LIMIT,
                        time_in_force=TimeInForce.GTC,
                        # order_class=OrderClass.SIMPLE,
                        take_profit=TakeProfitRequest(
                            limit_price=take_profit_price
                        ),
                        stop_loss=StopLossRequest(
                            stop_price=stop_price,
                            # limit_price=stop_limit_price,
                        ),
                    )
                    
                
            else: # not crypto
                order_request = MarketOrderRequest(
                    symbol= alpaca_symbol, #symbol
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
                        # limit_price=stop_limit_price,
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


        # Add to your StockPredictor class in predictor.py
    def get_entry_signal(self, symbol, current_price=None):
            """Generate real-time entry signal with confidence scoring
            Returns: (decision, confidence, rationale)
            """
            
            # Get real-time price if available
            if current_price is None:
                if datetime.now().hour < 16 and datetime.now().hour > 9 and datetime.now().minute > 30:
                    current_price = yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1]
                    current_tech  = yf.download('XLK', period='1d', interval='1m')['Close'].iloc[-1]
                    current_market = yf.download('^GSPC', period='1d', interval='1m')['Close'].iloc[-1]
        
                    
                elif datetime.now().hour < 9 and datetime.now().hour >= 0 and datetime.now().minute < 30:
                    current_price = yf.download(symbol, period='1d', interval='1m')['Close'].iloc[-1]
                    current_tech  = yf.download('XLK', period='1d', interval='1m')['Close'].iloc[-1]
                    current_market = yf.download('^GSPC', period='1d', interval='1m')['Close'].iloc[-1]
                else:
                    current_price = yf.download(symbol, period='1d')['Close'].iloc[-1]
                    current_tech  = yf.download('XLK', period='1d')['Close'].iloc[-1]
                    current_market = yf.download('^GSPC', period='1d')['Close'].iloc[-1]
            last_row = self.data.iloc[-1]
            second_last_row = self.data.iloc[-2]
            open_price = self.data['Close'].rolling(10).mean().dropna().iloc[0]
            
            
            # Calculate signal components
            signals = {
                'trend': {
                    'value': current_price > second_last_row['MA_50'],
                    'weight': 0.5
                },
                'momentum': {
                    'value': (last_row['RSI'] < 65) and (last_row['MACD'] > 0),
                    'weight': 0.5
                },
                'volume': {
                    'value': last_row['Volume'] > self.data['Volume'].rolling(20).mean().iloc[-1],
                    'weight': 0.5
                },
                'volatility': {
                    'value': last_row['ATR'] > self.data['ATR'].rolling(14).mean().iloc[-1],
                    'weight': 1
                },
                'mean_reversion': {
                    'value': current_price < last_row['Lower_Bollinger'] + 0.2*last_row['ATR'],
                    'weight': 0.5
                },
                'market_picture_increase': {
                    'value': (second_last_row['SP500'] < current_market),
                    'weight': 1
                },
                "sector_picture_increase": {
                    'value': (second_last_row['Tech'] < current_tech),
                    'weight': 1
                },
            }
             
            # signals = {
            #     "trend": {
            #         "value": (
            #             float(current_price)
            #             > float(predictor.data["Close"].rolling(50).mean().iloc[-1])
            #             if "MA_50" not in predictor.data.columns
            #             else float(current_price) > float(second_last_row["MA_50"])
            #         ),
            #         "weight": 0.1,
            #     },
            #     "mean_reversion": {
            #         "value": predictor.data["RSI"].iloc[-1] < 30,  # Oversold condition
            #         "weight": 1.0,
            #     },
            #     "volatility_breakout": {
            #         "value": predictor.data["ATR"].iloc[-1]
            #         > predictor.data["ATR"].rolling(20).mean().iloc[-1] * 1.5,
            #         "weight": 0.8,
            #     },
            #     "volume_spike": {
            #         "value": predictor.data["Volume"].iloc[-1]
            #         > predictor.data["Volume"].rolling(20).mean().iloc[-1] * 2,
            #         "weight": 0.7,
            #     },
            #     "support_resistance": {
            #         "value": abs(current_price - predictor.data["MA_50"].iloc[-1])
            #         / current_price
            #         < 0.01,  # Price near MA50
            #         "weight": 1.2,
            #     },
            #     "momentum": {
            #         "value": (
            #             float(last_row.get("RSI", 0)) < 65
            #             if "RSI" in predictor.data.columns
            #             else True
            #         ),
            #         "weight": 0.5,
            #     },
            #     "volume": {
            #         "value": (
            #             float(last_row.get("Volume", 0))
            #             > float(predictor.data["Volume"].rolling(20).mean().iloc[-1])
            #             if "Volume" in predictor.data.columns
            #             else True
            #         ),
            #         "weight": 0.5,
            #     },
            #     "volatility": {
            #         "value": (
            #             float(last_row.get("ATR", 0))
            #             > float(predictor.data["ATR"].rolling(14).mean().iloc[-1])
            #             if "ATR" in predictor.data.columns
            #             else True
            #         ),
            #         "weight": 1,
            #     },
            # }

            # # Add optional signals only if the columns exist
            # if "Lower_Bollinger" in predictor.data.columns and "ATR" in predictor.data.columns:
            #     signals["mean_reversion"] = {
            #         "value": current_price
            #         < last_row["Lower_Bollinger"] + 0.2 * last_row["ATR"],
            #         "weight": 0.5,
            #     }

            # if "SP500" in predictor.data.columns:
            #     signals["market_picture_increase"] = {
            #         "value": (second_last_row["SP500"] < last_row["SP500"]),
            #         "weight": 1,
            #     }

            # if "Tech" in predictor.data.columns:
            #     signals["sector_picture_increase"] = {
            #         "value": (second_last_row["Tech"] < last_row["Tech"]),
            #         "weight": 1,
            #     }

            # # Encourage Sell
            # signals["overbought"] = {
            #     "value": predictor.data["RSI"].iloc[-1] > 70,  # Overbought condition
            #     "weight": 1.2,  # Higher weight to encourage selling
            # }

            # signals["price_resistance"] = {
            #     "value": abs(current_price - predictor.data["Upper_Bollinger"].iloc[-1])
            #     / current_price
            #     < 0.01,
            #     "weight": 1.0,
            # }

            # signals["profit_taking"] = {
            #     "value": current_price
            #     > predictor.data["MA_50"].iloc[-1] * 1.1,  # 10% above MA50
            #     "weight": 0.8,
            # }

            
            # Calculate score (0-5 scale)
            score = sum(condition['weight'] for name, condition in signals.items() if condition['value'].all() == True)
            print(f"Score: {score} out of 5")
            max_score = sum(condition['weight'] for name, condition in signals.items())
            confidence = min(100, max(0, int((score / max_score) * 100)))
            
            # Generate rationale
            rationales = []
            if signals['trend']['value'].all() == True:
                rationales.append(f"📈 Price {current_price[0]:.2f} above 50MA ({last_row['MA_50']:.2f})")
            else:
                rationales.append(f"📉 Price {current_price[0]:.2f} below 50MA ({last_row['MA_50']:.2f})")
                
            if signals['momentum']['value'].all() == True:
                rationales.append(f"💪 Strong momentum (RSI {last_row['RSI']:.1f}, MACD {last_row['MACD']:.2f})")
                
            if signals['volume']['value'].all() == True:
                rationales.append(f"📊 Volume surge ({last_row['Volume']/1e6:.1f}M vs 20d avg)")
                
            # Make decision
            # if market today is bullish, then we lower the score needed to buy and increase the score needed to sell
            if open_price < current_price[0]:
                decision = "BUY" if score >= 3 else "SELL" if score <= 2 else "HOLD"
            else:
                decision = "SELL" if score >= 3.5 else "BUY" if score <= 1 else "HOLD"
            
            # Add risk check
            position_size = self._calculate_position_size()
            if position_size < 1:
                decision = "HOLD"
                rationales.append("⚠️ Position size too small")
            
            return (
                decision,
                confidence,
                " | ".join(rationales),
                
                {
                    'current_price': current_price,
                    'stop_loss': current_price * (1 - 2* self.risk_params['stop_loss_pct']),
                    'take_profit': current_price * (1 + self.risk_params['take_profit_pct']),
                }
                if decision == "BUY" or decision == "HOLD" 
                else {
                    'current_price': current_price,
                    'stop_loss': current_price * (1 + 2* self.risk_params['stop_loss_pct']),
                    'take_profit': current_price * (1 - self.risk_params['take_profit_pct']),
                }  
            )
 
####################################################################################################################################################################################



    # def load_data(self):
    #     """Load and prepare stock data with features"""
    #     # Add momentum-specific features
    #     window = 15  # Standard momentum window
    #     self.data = yf.download(
    #         self.symbol,
    #         start=self.start_date,
    #         end=self.end_date,
    #         interval=self.interval,
    #     )
    #     self.data.columns = self.data.columns.get_level_values(0)  # Remove multi-index
    #     self.data.ffill()
    #     self.data.dropna()

    #     ### 1. Add rolling indicators
    #     self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
    #     self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()
    #     self.data["MA_7"] = self.data["Close"].rolling(window=7).mean()
    #     self.data["MA_21"] = self.data["Close"].rolling(window=21).mean()

    #     ### 2. Fourier transform
    #     # data_FT = self.data.copy().reset_index()[["Date", "Close"]]
    #     # close_fft = np.fft.fft(np.asarray(data_FT["Close"].tolist()))
    #     # self.data["FT_real"] = np.real(close_fft)
    #     # self.data["FT_img"] = np.imag(close_fft)

    #     # # Fourier Transformation is not used
    #     # fft_df = pd.DataFrame({'fft': close_fft})
    #     # fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    #     # fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    #     # fft_list = np.asarray(fft_df['fft'].tolist())
    #     # for num_ in [3, 6, 9, 100]:
    #     #     fft_list_m10 = np.copy(fft_list)
    #     #     fft_list_m10[num_:-num_] = 0
    #     #     complex_num = np.fft.ifft(fft_list_m10)
    #     #     self.data[f'Fourier_trans_{num_}_comp_real'] = np.real(complex_num)
    #     #     self.data[f'Fourier_trans_{num_}_comp_img'] = np.imag(complex_num)

    #     # ### Fourier Transformation PCA
    #     # X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
    #     # pca = PCA(n_components=2)  # Keep top 2 components
    #     # X_pca = pca.fit_transform(X_fft)
    #     # for i in range(X_pca.shape[1]):
    #     #     self.data[f"Fourier_PCA_{i}"] = X_pca[:, i]

    #     ### 3. Add rolling statistics
    #     self.data["rolling_std"] = self.data["Close"].rolling(window=50).std()
    #     self.data["rolling_min"] = self.data["Close"].rolling(window=50).min()
    #     # self.data['rolling_max'] = self.data['Close'].rolling(window=window).max()
    #     self.data["rolling_median"] = self.data["Close"].rolling(window=50).median()
    #     self.data["rolling_sum"] = self.data["Close"].rolling(window=50).sum()
    #     self.data["rolling_var"] = self.data["Close"].rolling(window=50).var()
    #     self.data["rolling_ema"] = (
    #         self.data["Close"].ewm(span=50, adjust=False).mean()
    #     )  # Exponential Moving Average
    #     # Add rolling quantiles (25th and 75th percentiles)
    #     self.data["rolling_25p"] = self.data["Close"].rolling(window=50).quantile(0.25)
    #     self.data["rolling_75p"] = self.data["Close"].rolling(window=50).quantile(0.75)
    #     # Drop rows with NaN values (due to rolling window)
    #     self.data.dropna(inplace=True)
    #     stock_data.index.name = "Date"  # Ensure the index is named "Date"

    #     ### 4. Advanced Momentum
    #     self.data["RSI"] = self._compute_rsi(window=14)
    #     self.data["MACD"] = (
    #         self.data["Close"].ewm(span=12).mean()
    #         - self.data["Close"].ewm(span=26).mean()
    #     )
    #     ### 5. Williams %R
    #     high_max = self.data["High"].rolling(window).max()
    #     low_min = self.data["Low"].rolling(window).min()
    #     self.data["Williams_%R"] = (
    #         (high_max - self.data["Close"]) / (high_max - low_min)
    #     ) * -100

    #     ### 6. Stochastic Oscillator
    #     self.data["Stochastic_%K"] = (
    #         (self.data["Close"] - low_min) / (high_max - low_min)
    #     ) * 100
    #     self.data["Stochastic_%D"] = self.data["Stochastic_%K"].rolling(3).mean()

    #     ### 7. Momentum Divergence Detection
    #     self.data["Price_Change"] = self.data["Close"].diff()
    #     self.data["Momentum_Divergence"] = (
    #         (self.data["Price_Change"] * self.data["MACD"].diff()).rolling(5).sum()
    #     )

    #     ### 8. Volatility-adjusted Channels
    #     self.data["ATR"] = self._compute_atr(window=14)
    #     self.data["Upper_Bollinger"] = (
    #         self.data["MA_21"] + 2 * self.data["Close"].rolling(50).std()
    #     )
    #     self.data["Lower_Bollinger"] = (
    #         self.data["MA_21"] - 2 * self.data["Close"].rolling(50).std()
    #     )

    #     ### 9. Volume-based Features
    #     # self.data['OBV'] = self._compute_obv()
    #     if self.data["Volume"].cumsum()[-1] != 0:
    #         self.data["VWAP"] = (
    #             self.data["Volume"]
    #             * (self.data["High"] + self.data["Low"] + self.data["Close"])
    #             / 3
    #         ).cumsum() / self.data["Volume"].cumsum()

    #     ### 10. Economic Indicators
    #     # sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date, interval=self.interval,)["Close"]
    #     # # Fetch S&P 500 Index (GSPC) and Treasury Yield ETF (IEF) from Yahoo Finance
    #     # sp500 = sp500 - sp500.mean()
    #     # tnx = yf.download(
    #     #     "^TNX", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # tnx_len = len(tnx)
    #     # treasury_yield = yf.download(
    #     #     "IEF", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # exchange_rate = yf.download(
    #     #     "USDCAD=X", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # technology_sector = yf.download(
    #     #     "XLK", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # financials_sector = yf.download(
    #     #     "XLF", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # energy_sector = yf.download(
    #     #     "XLE", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]
    #     # vix = yf.download(
    #     #     "^VIX", start=self.start_date, end=self.end_date, interval=self.interval
    #     # )["Close"]

    #     # self.data["SP500"] = sp500
    #     # self.data["TNX"] = tnx
    #     # self.data["Treasury_Yield"] = treasury_yield
    #     # self.data["USDCAD=X"] = exchange_rate
    #     # self.data["Tech"] = technology_sector
    #     # self.data["Fin"] = financials_sector
    #     # self.data["VIX"] = vix
    #     # self.data["Energy"] = energy_sector

    #      # Batch download all economic indicators at once
    #     economic_tickers = ["^GSPC", "^TNX", "IEF", "USDCAD=X", "XLK", "XLF", "XLE", "^VIX"]
    #     economic_data = yf.download(
    #         economic_tickers,
    #         start=self.start_date, 
    #         end=self.end_date,
    #         interval=self.interval,
    #         group_by='ticker',  # Group results by ticker
    #         auto_adjust=True,   # Auto-adjust data
    #         progress=False      # Disable progress bar
    #     )

    #     # Extract each indicator from the batched data
    #     try:
    #         sp500 = economic_data['^GSPC']['Close'] - economic_data['^GSPC']['Close'].mean()
    #         tnx = economic_data['^TNX']['Close']
    #         tnx_len = len(tnx)
    #         treasury_yield = economic_data['IEF']['Close'] 
    #         exchange_rate = economic_data['USDCAD=X']['Close']
    #         technology_sector = economic_data['XLK']['Close']
    #         financials_sector = economic_data['XLF']['Close']
    #         energy_sector = economic_data['XLE']['Close']
    #         vix = economic_data['^VIX']['Close']
            
    #         # Additional defensive code to handle missing data
    #         for series_name, series in [
    #             ("TNX", tnx), 
    #             ("Treasury_Yield", treasury_yield),
    #             ("Exchange Rate", exchange_rate),
    #             ("Technology Sector", technology_sector),
    #             ("Financial Sector", financials_sector),
    #             ("Energy Sector", energy_sector),
    #             ("VIX", vix)
    #         ]:
    #             if series.empty:
    #                 print(f"Warning: {series_name} data is empty, filling with zeros")
    #                 if series_name == "TNX":
    #                     tnx = pd.Series(0, index=self.data.index)
    #                     tnx_len = 0
    #     except KeyError as e:
    #         print(f"Warning: One or more economic indicators missing: {e}")
    #         # Provide fallback values or skip the missing indicators



    #     economic_data = (
    #         pd.concat(
    #             [
    #                 sp500,
    #                 tnx,
    #                 treasury_yield,
    #                 exchange_rate,
    #                 technology_sector,
    #                 financials_sector,
    #                 vix,
    #                 energy_sector,
    #             ],
    #             axis=1,
    #             keys=[
    #                 "SP500",
    #                 "TNX",
    #                 "Treasury_Yield",
    #                 "USDCAD=X",
    #                 "Tech",
    #                 "Fin",
    #                 "VIX",
    #                 "Energy",
    #             ],
    #         )
    #         .reset_index()
    #         .rename(columns={"index": "Date"})
    #         # .dropna()
    #     )
    #     economic_data.columns = economic_data.columns.get_level_values(0)
    #     if self.interval == "1m":

    #         economic_data["Datetime"] = pd.to_datetime(economic_data["Datetime"])
    #         economic_data.set_index("Datetime", inplace=True)
    #     else:
    #         economic_data["Date"] = pd.to_datetime(economic_data["Date"])
    #         economic_data.set_index("Date", inplace=True)
        
    #     # Issue of Yfinance API of USDCAD=X
    #     # Fill missing values with the mean
    #     economic_data["USDCAD=X"] = economic_data["USDCAD=X"].fillna(
    #         economic_data["USDCAD=X"].mean()
    #     )

    #     # 11. Whether the next or previous day is a non-trading day
    #     # nyse = mcal.get_calendar("NYSE")
    #     # schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
    #     # economic_data["is_next_non_trading_day"] = economic_data.index.shift(
    #     #     -1, freq="1d"
    #     # ).isin(schedule.index).astype(int) + economic_data.index.shift(
    #     #     1, freq="1d"
    #     # ).isin(
    #     #     schedule.index
    #     # ).astype(
    #     #     int
    #     # )

    #     # Merge with stock data
    #     if tnx_len < len(self.data):
    #         economic_data = economic_data.drop(columns='TNX')
    #     if self.interval == "1m":
    #         self.data = pd.merge(self.data, economic_data, on="Datetime", how="left")
    #     else:
    #         self.data = pd.merge(self.data, economic_data, on="Date", how="left")

    #     ### 12. Volatility and Momentum
    #     # self.data["Daily Returns"] = self.data["Close"].pct_change() # Percentage change
    #     self.data["Daily Returns"] = (
    #         self.data["Close"].pct_change(window) * 100
    #     )  # Percentage change in the standard window for the momentum
    #     self.data["Volatility"] = self.data["Daily Returns"].rolling(window=20).std()
    #     # Adaptive Momentum Score
    #     vol_weight = self.data["Volatility"] * 100
    #     self.data["Momentum_Score"] = (
    #         self.data["RSI"] * 0.4
    #         + self.data["Daily Returns"] * 0.3
    #         + self.data["Williams_%R"] * 0.3
    #     ) / (1 + vol_weight)
    #     # Drop rows with NaN values
    #     self.data["Momentum_Interaction"] = (
    #         self.data["RSI"] * self.data["Daily Returns"]
    #     )
    #     self.data["Volatility_Adj_Momentum"] = self.data["Momentum_Score"] / (
    #         1 + self.data["Volatility"]
    #     )
    #     self.data["Volatility_Adj_Momentum"] = self.data[
    #         "Volatility_Adj_Momentum"
    #     ].clip(lower=0.1)
    #     self.data["Volatility_Adj_Momentum"] = self.data[
    #         "Volatility_Adj_Momentum"
    #     ].clip(upper=10.0)
    #     self.data["Volatility_Adj_Momentum"] = self.data[
    #         "Volatility_Adj_Momentum"
    #     ].fillna(0.0)

    #     ### 13. Market Regime Detection by HMM
    #     # hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    #     # hmm.fit(self.data["Close"].pct_change().dropna().values.reshape(-1, 1))
    #     # # Predict hidden states
    #     # market_state = hmm.predict(
    #     #     self.data["Close"].pct_change().dropna().values.reshape(-1, 1)
    #     # )
    #     # hmm_sp = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    #     # hmm_sp.fit(self.data["SP500"].pct_change().dropna().values.reshape(-1, 1))
    #     # market_state_sp500 = hmm_sp.predict(
    #     #     self.data["SP500"].pct_change().dropna().values.reshape(-1, 1)
    #     # )
    #     # # Initialize the Market_State column
    #     # self.data["Market_State"] = np.zeros(len(self.data))
    #     # if (
    #     #     len(set(list(market_state))) != 1
    #     #     and len(set(list(market_state_sp500))) != 1
    #     # ):
    #     #     self.data["Market_State"][0] = 0
    #     #     self.data.iloc[1:]["Market_State"] = market_state + market_state_sp500

    #     # ### 14. Sentiment Analysis (Computationally expensive)
    #     # self.data["Market_Sentiment"] = 0.0
    #     # sentimement = MarketSentimentAnalyzer().get_historical_sentiment(
    #     #     self.symbol, self.data.shape[0]
    #     # )
    #     # self.data["Market_Sentiment"] = sentimement

    #     # Final cleaning
    #     # convert timezone to AMErican/New_York
    #     if self.interval == "1m":
    #         self.data.index = self.data.index.tz_convert("America/New_York")
    #     self.data = self.data.dropna()
    #     if len(self.data) < 50:
    #         print("Not enough data to train the model.")
    #         raise ValueError("Not enough data to train the model.")

    #     return self
    def load_data(self):
        """Load and prepare stock data with features"""
        window = 15  # Standard momentum window
        
        # Check if this is a cryptocurrency ticker
        is_crypto = '-USD' in self.symbol or 'USD' in self.symbol or '/USD' in self.symbol or self.symbol.endswith('BTC')
        
        try: 
            yf_symbol = self.symbol.replace("/", "-") if '/' in self.symbol else self.symbol
            data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
            )
            
            if data.empty or len(data) < 5:
                raise Exception(f"Insufficient data from Yahoo Finance for {yf_symbol}")
                
            data.columns = data.columns.get_level_values(0)  # Remove multi-index
            self.data = data
            print(f"Successfully loaded data from Yahoo Finance for {yf_symbol}")
            
        except Exception as e:
            print(f"Failed to get data from Yahoo Finance for {self.symbol}: {str(e)}")
            if is_crypto:
                print(f"Attempting to get crypto data from Alpaca for {self.symbol}")
                self._load_crypto_data_from_alpaca()
            else:
                print(f"Attempting to get stock data from Alpaca for {self.symbol}")
                self._load_stock_data_from_alpaca()

        if self.data.empty:
            print(f"Warning: Data for {self.symbol} is empty after cleaning")
            return self

        self._add_technical_indicators(window)
        
        # Modularized feature loading based on asset type
        if is_crypto:
            # Add crypto-specific indicators
            self._add_crypto_liquidity_indicators()
            self._add_crypto_volatility_indicators()
            self._add_crypto_market_structure_indicators()
            self._add_crypto_onchain_indicators()
        else:
            # For stocks, use economic indicators
            self._add_economic_indicators()

        # Final cleaning
        if self.interval in ["1m", "5m"]:
            self.data.index = self.data.index.tz_convert("America/New_York")
        
        self.data = self.data.dropna(axis=1, thresh=len(self.data) * 0.1)
        self.data = self.data.dropna()
        
        if len(self.data) < 50:
            raise ValueError("Not enough data to train the model. Number of rows: {}".format(len(self.data)))
        else:
            print(f"Data loaded for {self.symbol} with {len(self.data)} rows.")

        return self
    

      
    def _add_technical_indicators(self, window=15):
        """Add all technical indicators"""
        ### 1. Add rolling indicators
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()
        self.data["MA_200"] = self.data["Close"].rolling(window=200).mean()
        self.data["MA_7"] = self.data["Close"].rolling(window=7).mean()
        self.data["MA_21"] = self.data["Close"].rolling(window=21).mean()

        ### 2. Fourier transform
        # data_FT = self.data.copy().reset_index()[["Date", "Close"]]
        # close_fft = np.fft.fft(np.asarray(data_FT["Close"].tolist()))
        # self.data["FT_real"] = np.real(close_fft)
        # self.data["FT_img"] = np.imag(close_fft)

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

        # ### Fourier Transformation PCA
        # X_fft = np.column_stack([np.real(close_fft), np.imag(close_fft)])
        # pca = PCA(n_components=2)  # Keep top 2 components
        # X_pca = pca.fit_transform(X_fft)
        # for i in range(X_pca.shape[1]):
        #     self.data[f"Fourier_PCA_{i}"] = X_pca[:, i]

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
        # stock_data.index.name = "Date"  # Ensure the index is named "Date"

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
        if self.data["Volume"].cumsum()[-1] != 0:
            self.data["VWAP"] = (
                self.data["Volume"]
                * (self.data["High"] + self.data["Low"] + self.data["Close"])
                / 3
            ).cumsum() / self.data["Volume"].cumsum()
         # 11. Whether the next or previous day is a non-trading day
        # nyse = mcal.get_calendar("NYSE")
        # schedule = nyse.schedule(start_date=self.start_date, end_date=self.end_date)
        # economic_data["is_next_non_trading_day"] = economic_data.index.shift(
        #     -1, freq="1d"
        # ).isin(schedule.index).astype(int) + economic_data.index.shift(
        #     1, freq="1d"
        # ).isin(
        #     schedule.index
        # ).astype(
        #     int
        # )
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


       
    def get_cached_data(self, tickers, start_date, end_date, interval, cache_dir="data_cache"):
        """Cache economic data to avoid repeated downloads"""
        import hashlib
        import os
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a unique cache key based on parameters
        cache_key = f"{','.join(sorted(tickers))}_{start_date}_{end_date}_{interval}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_hash}.pkl")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            try:
                cached_data = joblib.load(cache_file)
                print(f"Loaded cached economic data for {len(tickers)} tickers")
                return cached_data
            except Exception as e:
                print(f"Failed to load cached data: {e}")
        
        # Download fresh data
        print(f"Downloading fresh economic data for {len(tickers)} tickers")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            prepost=True
        )
        
        # Cache the data
        try:
            joblib.dump(data, cache_file)
            print(f"Cached economic data to {cache_file}")
        except Exception as e:
            print(f"Failed to cache data: {e}")
        
        return data

    def _add_economic_indicators(self):
        """Add economic indicators for stock trading"""
        # Batch download all economic indicators at once
        economic_tickers = ["^GSPC", "^TNX", "IEF", "USDCAD=X", "XLK", "XLF", "XLE", "^VIX"]
        try:
            print("Trying to fetch economic data from cache...")
            economic_data = self.get_cached_data(
                economic_tickers,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
                cache_dir="data_cache"
            )
        except Exception as e:
            print(f"Cache fetch failed: {str(e)}. Downloading fresh data...")
            economic_data = yf.download(
                economic_tickers,
                start=self.start_date, 
                end=self.end_date,
                interval=self.interval,
                group_by='ticker',  # Group results by ticker
                auto_adjust=True,   # Auto-adjust data
                progress=False,      # Disable progress bar
                prepost=True  # Include pre/post market data
            )

        # Extract each indicator from the batched data
        try:
            sp500 = economic_data['^GSPC']['Close'] - economic_data['^GSPC']['Close'].mean()
            tnx = economic_data['^TNX']['Close']
            tnx_len = len(tnx)
            treasury_yield = economic_data['IEF']['Close'] 
            exchange_rate = economic_data['USDCAD=X']['Close']
            technology_sector = economic_data['XLK']['Close']
            financials_sector = economic_data['XLF']['Close']
            energy_sector = economic_data['XLE']['Close']
            vix = economic_data['^VIX']['Close']
            
            # Additional defensive code to handle missing data
            for series_name, series in [
                ("TNX", tnx), 
                ("Treasury_Yield", treasury_yield),
                ("Exchange Rate", exchange_rate),
                ("Technology Sector", technology_sector),
                ("Financial Sector", financials_sector),
                ("Energy Sector", energy_sector),
                ("VIX", vix)
            ]:
                if series.empty:
                    print(f"Warning: {series_name} data is empty, filling with zeros")
                    if series_name == "TNX":
                        tnx = pd.Series(0, index=self.data.index)
                        tnx_len = 0
        except KeyError as e:
            print(f"Warning: One or more economic indicators missing: {e}")
            # Provide fallback values or skip the missing indicators


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
        if self.interval == "1m" or self.interval == "5m":
        # or self.interval == "15m" or self.interval == "30m" or self.interval == "60m" or self.interval == "90m":

            economic_data["Datetime"] = pd.to_datetime(economic_data["Datetime"])
            economic_data.set_index("Datetime", inplace=True)
        else:
            economic_data["Date"] = pd.to_datetime(economic_data["Date"])
            economic_data.set_index("Date", inplace=True)
        
        # Issue of Yfinance API of USDCAD=X
        # Fill missing values with the mean
        economic_data["USDCAD=X"] = economic_data["USDCAD=X"].fillna(
            economic_data["USDCAD=X"].mean()
        )
        # Merge with stock data
        if tnx_len < len(self.data):
            economic_data = economic_data.drop(columns='TNX')
        if self.interval in ["1m", "5m"]:
            self.data = pd.merge(self.data, economic_data, on="Datetime", how="left")
        else:
            self.data = pd.merge(self.data, economic_data, on="Date", how="left")

    
    def _add_crypto_liquidity_indicators(self):
        """Add crypto-specific liquidity indicators"""
        
        # Volume-based liquidity metrics
        self.data['Volume_MA'] = self.data['Volume'].rolling(24).mean()
        self.data['Relative_Volume'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Liquidity ratio (higher values indicate better liquidity)
        self.data['Liquidity_Ratio'] = self.data['Volume'] / (self.data['High'] - self.data['Low']).replace(0, 0.001)
        
        # Volume-weighted volatility (measures how efficiently price moves with volume)
        self.data['Vol_Weighted_Volatility'] = (self.data['Close'].pct_change().abs() * 
                                            self.data['Volume'] / self.data['Volume_MA'])
        
        # VWAP and deviation from VWAP (measure of buying/selling pressure)
        self.data['VWAP'] = (self.data['Volume'] * self.data['Close']).rolling(24).sum() / self.data['Volume'].rolling(24).sum()
        self.data['VWAP_Deviation'] = ((self.data['Close'] - self.data['VWAP']) / self.data['VWAP']) * 100
        
        # Flash crash detector (sudden volume spike with price drop)
        vol_spike = self.data['Volume'] > (self.data['Volume'].rolling(12).mean() * 3)
        price_drop = self.data['Close'].pct_change() < -0.03
        self.data['Flash_Crash_Signal'] = vol_spike & price_drop
            
    
    def _add_crypto_volatility_indicators(self):
        """Add specialized crypto volatility indicators"""
        
        # True Range and ATR variations
        self.data['TR'] = np.maximum(
            self.data['High'] - self.data['Low'],
            np.maximum(
                abs(self.data['High'] - self.data['Close'].shift(1)),
                abs(self.data['Low'] - self.data['Close'].shift(1))
            )
        )
        self.data['ATR_1h'] = self.data['TR'].rolling(60).mean()  # 1-hour ATR (for minute data)
        self.data['ATR_24h'] = self.data['TR'].rolling(1440).mean()  # 24-hour ATR
        
        # Volatility ratio (short-term vs long-term)
        self.data['Volatility_Ratio'] = self.data['ATR_1h'] / self.data['ATR_24h']
        
        # Bollinger Band Width (normalized)
        bb_period = 20
        std_dev = 2
        self.data['BB_Middle'] = self.data['Close'].rolling(bb_period).mean()
        self.data['BB_Width'] = ((self.data['Close'].rolling(bb_period).std() * std_dev * 2) / 
                            self.data['BB_Middle']) * 100
        
        # Historical Volatility (annualized)
        self.data['HV_1h'] = self.data['Close'].pct_change().rolling(60).std() * np.sqrt(525600)  # Minutes in a year
        self.data['HV_24h'] = self.data['Close'].pct_change().rolling(1440).std() * np.sqrt(525600)
        
        # Volatility Regime (1=low, 2=medium, 3=high)
        self.data['Volatility_Regime'] = 1  # Default to low
        vol_75th = self.data['HV_24h'].quantile(0.75)
        vol_25th = self.data['HV_24h'].quantile(0.25)
        self.data.loc[self.data['HV_24h'] > vol_25th, 'Volatility_Regime'] = 2  # Medium
        self.data.loc[self.data['HV_24h'] > vol_75th, 'Volatility_Regime'] = 3  # High
        
        # Guppy Multiple Moving Average (GMMA) Compression/Expansion
        # A measure of trend strength and potential volatility expansion
        ema_short = [3, 5, 8, 10, 12, 15]
        ema_long = [30, 35, 40, 45, 50, 60]
        
        for period in ema_short + ema_long:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        
        # Calculate average distance between short EMAs
        short_emas = [self.data[f'EMA_{p}'] for p in ema_short]
        long_emas = [self.data[f'EMA_{p}'] for p in ema_long]
        
        self.data['GMMA_Short_Spread'] = (max([short_emas[i].iloc[-1] for i in range(len(short_emas))]) - 
                                        min([short_emas[i].iloc[-1] for i in range(len(short_emas))])) / self.data['Close']
        self.data['GMMA_Long_Spread'] = (max([long_emas[i].iloc[-1] for i in range(len(long_emas))]) - 
                                    min([long_emas[i].iloc[-1] for i in range(len(long_emas))])) / self.data['Close']


    def _add_crypto_market_structure_indicators(self):
        """Add market structure indicators specifically for crypto"""
        
        # Detect pivot points (swing highs and lows)
        pivot_length = 5
        self.data['Pivot_High'] = 0
        self.data['Pivot_Low'] = 0
        
        for i in range(pivot_length, len(self.data) - pivot_length):
            # Check for pivot high
            if (self.data['High'].iloc[i] > self.data['High'].iloc[i-pivot_length:i].max() and 
                self.data['High'].iloc[i] > self.data['High'].iloc[i+1:i+pivot_length+1].max()):
                self.data.loc[self.data.index[i], 'Pivot_High'] = 1
                
            # Check for pivot low
            if (self.data['Low'].iloc[i] < self.data['Low'].iloc[i-pivot_length:i].min() and 
                self.data['Low'].iloc[i] < self.data['Low'].iloc[i+1:i+pivot_length+1].min()):
                self.data.loc[self.data.index[i], 'Pivot_Low'] = 1
        
        # Market Structure Shift Detection
        self.data['Structure_Bullish'] = 0
        self.data['Structure_Bearish'] = 0
        
        pivot_highs = self.data[self.data['Pivot_High'] == 1]
        pivot_lows = self.data[self.data['Pivot_Low'] == 1]
        
        if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
            # Check for Higher Highs & Higher Lows (Bullish Structure)
            last_two_highs = pivot_highs.iloc[-2:]['High'].values
            last_two_lows = pivot_lows.iloc[-2:]['Low'].values
            
            if len(last_two_highs) == 2 and len(last_two_lows) == 2:
                if last_two_highs[1] > last_two_highs[0] and last_two_lows[1] > last_two_lows[0]:
                    self.data.loc[self.data.index[-1], 'Structure_Bullish'] = 1
                    
                # Check for Lower Highs & Lower Lows (Bearish Structure)
                if last_two_highs[1] < last_two_highs[0] and last_two_lows[1] < last_two_lows[0]:
                    self.data.loc[self.data.index[-1], 'Structure_Bearish'] = 1
        
        # Ichimoku Cloud for trend structure and support/resistance
        high_9 = self.data['High'].rolling(window=9).max()
        low_9 = self.data['Low'].rolling(window=9).min()
        self.data['Tenkan_Sen'] = (high_9 + low_9) / 2  # Conversion Line
        
        high_26 = self.data['High'].rolling(window=26).max()
        low_26 = self.data['Low'].rolling(window=26).min()
        self.data['Kijun_Sen'] = (high_26 + low_26) / 2  # Base Line
        
        self.data['Senkou_Span_A'] = ((self.data['Tenkan_Sen'] + self.data['Kijun_Sen']) / 2).shift(26)  # Leading Span A
        self.data['Senkou_Span_B'] = ((self.data['High'].rolling(window=52).max() + 
                                    self.data['Low'].rolling(window=52).min()) / 2).shift(26)  # Leading Span B
        
        # Cloud state (Above/Below/In cloud)
        self.data['Cloud_State'] = 0  # 0 = in cloud, 1 = above cloud (bullish), -1 = below cloud (bearish)
        
        for i in range(len(self.data)):
            if i > 26:  # Ensure we have cloud data
                if self.data['Close'].iloc[i] > max(self.data['Senkou_Span_A'].iloc[i], self.data['Senkou_Span_B'].iloc[i]):
                    self.data.loc[self.data.index[i], 'Cloud_State'] = 1
                elif self.data['Close'].iloc[i] < min(self.data['Senkou_Span_A'].iloc[i], self.data['Senkou_Span_B'].iloc[i]):
                    self.data.loc[self.data.index[i], 'Cloud_State'] = -1


    def _add_crypto_onchain_indicators(self):
        """Add crypto-specific indicators that mimic on-chain and exchange data"""
        
        # Simulate exchange inflow/outflow with price and volume
        self.data['Exchange_Flow'] = 0
        price_change = self.data['Close'].pct_change()
        volume_change = self.data['Volume'].pct_change()
        
        # When price drops but volume increases = potential exchange inflow (selling pressure)
        self.data.loc[(price_change < -0.01) & (volume_change > 0.2), 'Exchange_Flow'] = -1
        
        # When price increases with volume = potential exchange outflow (buying pressure)
        self.data.loc[(price_change > 0.01) & (volume_change > 0.2), 'Exchange_Flow'] = 1
        
        # Average True Range Volatility Bands (wider during high volatility)
        self.data['ATR_5'] = self.data['TR'].rolling(5).mean()
        self.data['Upper_Band'] = self.data['Close'] + (self.data['ATR_5'] * 2)
        self.data['Lower_Band'] = self.data['Close'] - (self.data['ATR_5'] * 2)
        
        # Whale activity detection (large volume spikes)
        median_vol = self.data['Volume'].rolling(50).median()
        self.data['Whale_Activity'] = (self.data['Volume'] > median_vol * 3).astype(int)
        
        # Create Buy/Sell imbalance ratio
        close_change = self.data['Close'].diff()
        self.data['Buy_Volume'] = self.data['Volume'] 
        self.data['Sell_Volume'] = self.data['Volume']
        
        self.data.loc[close_change > 0, 'Sell_Volume'] = self.data.loc[close_change > 0, 'Volume'] * 0.4
        self.data.loc[close_change < 0, 'Buy_Volume'] = self.data.loc[close_change < 0, 'Volume'] * 0.4
        
        # Replace the problematic line with this more robust implementation
        buy_vol_sum = self.data['Buy_Volume'].rolling(24).sum()
        sell_vol_sum = self.data['Sell_Volume'].rolling(24).sum()
        # Handle potential zeros in denominator and handle NaNs
        self.data['Buy_Sell_Ratio'] = buy_vol_sum / sell_vol_sum.replace(0, 0.001)
        # Fill NaN values with a neutral ratio of 1.0
        self.data['Buy_Sell_Ratio'] = self.data['Buy_Sell_Ratio'].fillna(1.0)



    def _load_crypto_data_from_alpaca(self):
        """Get crypto data from Alpaca API when yfinance fails"""
        try:
            from alpaca.data import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            # Initialize crypto data client
            crypto_client = CryptoHistoricalDataClient()
            
            # Convert interval to Alpaca timeframe
            timeframe_map = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame(5, TimeFrameUnit.Minute),
                # "15m": TimeFrame.Minute(15),
                # "30m": TimeFrame.Minute(30),
                # "60m": TimeFrame.Hour,
                # "90m": TimeFrame.Hour,  # Alpaca doesn't have 90m, use Hour
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day
            }
            
            timeframe = timeframe_map.get(self.interval, TimeFrame.Day)
            
            # Prepare symbol for Alpaca (ensure it has the right format)
            alpaca_symbol = self.symbol
            if '-' in alpaca_symbol and '/' not in alpaca_symbol:
                alpaca_symbol = alpaca_symbol.replace('-', '/')
            
            # Create request for crypto bars
            request_params = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol,
                timeframe=timeframe,
                start=pd.Timestamp(self.start_date).tz_localize("America/New_York"),
                end=pd.Timestamp(self.end_date).tz_localize("America/New_York")
            )
            
            # Get the bars
            bars = crypto_client.get_crypto_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            # Reset multi-level index and format similar to yfinance output
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            
            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close', 
                'volume': 'Volume',
                # 'trade_count': 'Trade_Count',
                # 'vwap': 'VWAP'
            })
            
            self.data = df
            # logger.info(f"Successfully loaded crypto data from Alpaca for {alpaca_symbol}")
            return df
            
        except Exception as e:
            print(f"Failed to get crypto data from Alpaca for {self.symbol}: {str(e)}")
            self.data = pd.DataFrame()  # Empty DataFrame
            return self.data

    def _load_stock_data_from_alpaca(self):
        """Get stock data from Alpaca API when yfinance fails"""
        try:
            from alpaca.data import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            # Init stock data client (assuming API keys are available)
            api_key = "PKXPBKCIK15IBA4G84P4"
            secret_key = "aJHuDphvn8S6M69F0Vrc0EAudEgob2xc5ltXc0bA"
            stock_client = StockHistoricalDataClient(api_key, secret_key)
            
            # Convert interval to Alpaca timeframe
            timeframe_map = {
                "1m": TimeFrame.Minute,
                "5m": TimeFrame(5, TimeFrameUnit.Minute),
                # "15m": TimeFrame.Minute(15),
                # "30m": TimeFrame.Minute(30),
                # "60m": TimeFrame.Hour,
                # "90m": TimeFrame.Hour,  # Alpaca doesn't have 90m, use Hour
                "1h": TimeFrame.Hour,
                "1d": TimeFrame.Day
            }
            
            timeframe = timeframe_map.get(self.interval, TimeFrame.Day)
            
            # Create request for stock bars
            request_params = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=timeframe,
                start=pd.Timestamp(self.start_date).tz_localize("America/New_York"),
                end=pd.Timestamp(self.end_date).tz_localize("America/New_York")
            )
            
            # Get the bars
            bars = stock_client.get_stock_bars(request_params)
            
            # Convert to DataFrame
            df = bars.df
            
            # Reset multi-level index and format similar to yfinance output
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
                
            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close', 
                'volume': 'Volume',
                # 'trade_count': 'Trade_Count',
                # 'vwap': 'VWAP'
            })
            
            self.data = df
            # print(f"Successfully loaded stock data from Alpaca for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"Failed to get stock data from Alpaca for {self.symbol}: {str(e)}")
            self.data = pd.DataFrame()  # Empty DataFrame
            return self.data




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
            if cached_model and not needs_retrain:
            # if cached_model:
                print(f"Using cached model for {predictor}")
                self.models[predictor] = cached_model
            else:
                print(f"Training new model for {predictor}")

              
                # Select features excluding the current predictor
                features = [col for col in predictors if col != predictor]

                # Prepare data
                X = self.data[features].iloc[:-horizon,]
                y = self.data[predictor].iloc[:-horizon,]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
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
                    # adj_r2 = 1 - (1 - r2_score(y_true=y_test, y_pred=y_pred)) * (
                    #     X_test.shape[0] - 1
                    # ) / (X_test.shape[0] - X_test.shape[1] - 1)

                    # Compute metrics
                    rmse = root_mean_squared_error(y_test, y_pred)
                    result[name] = {"rmse": rmse, "r2": r2}

                    print(f"{predictor} - {name.capitalize()} Model:")
                    print(f"  Test Mean Squared Error: {rmse:.4f}")
                    print(f"  R² Score: {r2:.4f}")
                    # if "arimaxgb" in result:
                    #     if result["arimaxgb"]["r2"] != max(
                    #         [result[model]["r2"] for model in result]
                    #     ):
                    #         if predictor == "Close" and (result["arimaxgb"]["r2"] < 0.8):
                    #             os.remove(
                    #                 f"{self.model_cache_dir}/{predictor}.pkl"
                    #             )
                    #             raise ValueError(
                    #                 "ARIMAXGBoost model failed to converge (r2 < 0.8). Please check your data period or model parameters."
                    #             )
                                
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
              
           
                    current_hash = self._get_data_hash()
                    
                    with open(f"{self.model_cache_dir}/{predictor}.hash", "w", encoding='utf-8') as f:
                        f.write(current_hash)
                    

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
        
        # cached_final_prediction = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="forecast")
        # cached_final_backtest = self._load_cached_result(model_type=model_type,horizon=horizon, output_type="backtest")
        # cached_final_raw_prediction = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="raw_forecast")
        # cached_final_raw_backtest = self._load_cached_result(model_type=model_type,horizon=horizon,output_type="raw_backtest")

        # # if cached_model and not needs_retrain: 
        # if self._model_needs_retraining() is False:
        #     print(f"Using cached forecasts and backtests for {self.symbol}")
        #     return cached_final_prediction, cached_final_backtest, cached_final_raw_prediction, cached_final_raw_backtest   

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
            if self.interval == "1m":
                next_pred_date = last_pred_date + pd.Timedelta(minutes=1)
                next_backtest_date = last_backtest_date + pd.Timedelta(minutes=1)
            else:
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


# class Backtester:
#     """Integrated backtesting engine that works with your StockPredictor"""
    
#     def __init__(self, predictor, initial_capital=100000):
#         self.predictor = predictor
#         self.initial_capital = initial_capital
#         self.portfolio = {
#             'cash': initial_capital,
#             'positions': {},
#             'value_history': [],
#             'transactions': []
#         }
#         self.slippage = 0.0005  # 5bps
#         self.commission = 0.0001  # $0.01 per share
    
#     def _calculate_position_size(self, current_price):
#         """Use your existing risk parameters"""
#         risk_per_trade = self.initial_capital * self.predictor.risk_params['per_trade_risk']
#         atr = self.predictor.data['ATR'].iloc[-1]
#         return risk_per_trade / (atr * current_price)
    
#     # def run_backtest(self, start_date, end_date):
#     #     """Walk-forward backtest using your existing signals"""
#     #     nyse = mcal.get_calendar("NYSE")
#     #     dates = nyse.schedule(start_date=start_date, end_date=end_date).index
#     #     for i, date in enumerate(dates):
#     #         if date not in self.predictor.data.index:
#     #             continue
                
#     #         # Get historical data up to current date 
#     #         self.predictor.data = self.predictor.data.loc[:date] # This code is good
            
#     #         # Generate signal using existing code
#     #         signal = self.predictor.generate_trading_signal(self.predictor.symbol)
            
#     #         # Execute trade
#     #         try: #since sometimes yf may have empty data for some date
#     #             current_price = self.predictor.data['Close'].loc[date]
#     #             position_size = self._calculate_position_size(current_price)
#     #         except KeyError:
#     #             print(f"Data not available for {date}. Skipping.")
#     #             continue
            
#     #         # Apply slippage and commission
#     #         executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
#     #             else current_price * (1 - self.slippage)
            
#     #         if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
#     #             self._execute_buy(executed_price, position_size, date)
#     #         elif signal == 'SELL' and self.predictor.symbol in self.portfolio['positions']:
#     #             self._execute_sell(executed_price, date)
            
#     #         # Update portfolio value
#     #         self._update_portfolio_value(date)
            
#     #         # Check risk limits
#     #         if self._check_daily_loss():
#     #             break
                
#     #     return self._generate_report()

#     def run_backtest(self, start_date, end_date):
#         """More robust date handling"""
#         try:
#             nyse = mcal.get_calendar("NYSE")
#             schedule = nyse.schedule(start_date=start_date, end_date=end_date)
#             if schedule.empty:
#                 print(f"No trading days between {start_date} and {end_date}")
#                 return pd.DataFrame(), {'error': 'No trading days'}
                
#             dates = schedule.index.tz_localize(None)
#         except Exception as e:
#             print(f"Date error: {str(e)}")
#             return pd.DataFrame(), {'error': str(e)}

#         for i, date in enumerate(dates):
#             if not (date in self.predictor.data.index):
#                 continue
                
#             # Rest of existing logic
#             # ...
#             #  Get historical data up to current date 
#             self.predictor.data = self.predictor.data.loc[:date] # This code is good
            
#             # Generate signal using existing code
#             signal = self.predictor.generate_trading_signal(self.predictor.symbol)
            
#             # Execute trade
#             try: #since sometimes yf may have empty data for some date
#                 current_price = self.predictor.data['Close'].loc[date]
#                 position_size = self._calculate_position_size(current_price)
#             except KeyError:
#                 print(f"Data not available for {date}. Skipping.")
#                 continue
            
#             # Apply slippage and commission
#             executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
#                 else current_price * (1 - self.slippage)
            
#             if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
#                 self._execute_buy(executed_price, position_size, date)
#             elif signal == 'SELL' and self.predictor.symbol in self.portfolio['positions']:
#                 self._execute_sell(executed_price, date)
            
#             # Update portfolio value
#             self._update_portfolio_value(date)
            
#             # Check risk limits
#             if self._check_daily_loss():
#                 break
                
#         return self._generate_report()
        


#     def _execute_buy(self, price, qty, date):
#         cost = price * qty + self.commission * qty
#         self.portfolio['cash'] -= cost
#         self.portfolio['positions'][self.predictor.symbol] = {
#             'qty': qty, 
#             'entry_price': price,
#             'entry_date': date
#         }
#         self.portfolio['transactions'].append(('BUY', price, qty, date))

#     # def _execute_sell(self, price, date):
#     #     position = self.portfolio['positions'].pop(self.predictor.symbol)
#     #     proceeds = price * position['qty'] - self.commission * position['qty']
#     #     self.portfolio['cash'] += proceeds
#     #     self.portfolio['transactions'].append(('SELL', price, position['qty'], date))



#     def _execute_sell(self, price, date):
#         if self.predictor.symbol not in self.portfolio['positions']:
#             print(f"No position to sell for {self.predictor.symbol}")
#             return
            
#         position = self.portfolio['positions'].pop(self.predictor.symbol)
#         proceeds = price * position['qty'] - self.commission * position['qty']
#         self.portfolio['cash'] += proceeds
#         self.portfolio['transactions'].append(('SELL', price, position['qty'], date))


#     # def _update_portfolio_value(self, date):
#     #     position_value = 0
#     #     for sym, pos in self.portfolio['positions'].items():
#     #         position_value += pos['qty'] * self.predictor.data['Close'].loc[date]
#     #     self.portfolio['value_history'].append({
#     #         'date': date,
#     #         'value': self.portfolio['cash'] + position_value
#     #     })
    
#     def _check_daily_loss(self):
#         """Use your existing risk management"""
#         if len(self.portfolio['value_history']) < 2:
#             return False
#         daily_pct = (self.portfolio['value_history'][-1]['value'] / 
#                     self.portfolio['value_history'][-2]['value']) - 1
#         return daily_pct < self.predictor.risk_params['daily_loss_limit']

#     # def _generate_report(self):
#     #     """Generate performance report using your existing metrics"""
#     #     df = pd.DataFrame(self.portfolio['value_history']).set_index('date')
#     #     returns = df['value'].pct_change().dropna()
        
#     #     report = {
#     #         'sharpe': returns.mean() / returns.std() * np.sqrt(252),
#     #         'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
#     #         'total_return': df['value'].iloc[-1] / self.initial_capital - 1,
#     #         'win_rate': len([t for t in self.portfolio['transactions'] if t[0] == 'SELL' and 
#     #                       (t[1] - self.portfolio['positions'][t[3]]['entry_price']) > 0]) / 
#     #                      max(1, len([t for t in self.portfolio['transactions'] if t[0] == 'SELL']))
#     #     }
#     #     return df, report
#     def _update_portfolio_value(self, date):
#         position_value = 0
#         for sym, pos in self.portfolio['positions'].items():
#             if date in self.predictor.data.index:  # Add validation
#                 position_value += pos['qty'] * self.predictor.data['Close'].loc[date]
        
#         # Ensure consistent data format
#         self.portfolio['value_history'].append({
#             'date': pd.to_datetime(date),
#             'value': self.portfolio['cash'] + position_value
#         })

#     def _generate_report(self):
#         """More robust report generation"""
#         if not self.portfolio['value_history']:
#             return pd.DataFrame(), {'error': 'No trades executed'}
        
#         try:
#             df = pd.DataFrame(self.portfolio['value_history'])
#             df = df.set_index('date').sort_index()
            
#             if df.empty:
#                 return df, {'error': 'Empty portfolio history'}
                
#             returns = df['value'].pct_change().dropna()
            
#             if len(returns) < 2:
#                 return df, {'error': 'Insufficient data for metrics'}
            
#             report = {
#                 'sharpe': returns.mean() / returns.std() * np.sqrt(252),
#                 'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
#                 'total_return': df['value'].iloc[-1] / self.initial_capital - 1,
#                 'num_trades': len(self.portfolio['transactions']),
#                 'win_rate': self._calculate_win_rate()
#             }
#             return df, report
            
#         except KeyError as e:
#             print(f"Report generation error: {str(e)}")
#             return pd.DataFrame(), {'error': str(e)}

#     def _calculate_win_rate(self):
#         """Safer win rate calculation"""
#         sell_trades = [t for t in self.portfolio['transactions'] if t[0] == 'SELL']
#         if not sell_trades:
#             return 0.0
            
#         winning_trades = 0
#         for trade in sell_trades:
#             entry_price = next(
#                 (t[1] for t in self.portfolio['transactions'] 
#                 if t[0] == 'BUY' and t[3] == trade[3]), None)
#             if entry_price and trade[1] > entry_price:
#                 winning_trades += 1
                
#         return winning_trades / len(sell_trades)


class Backtester:
    """Integrated backtesting engine that works with your StockPredictor"""
    
    def __init__(self, predictor, initial_capital=100000):
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'value_history': [],
            'transactions': []
        }
        self.slippage = 0.0005  # 5bps
        self.commission = 0.0001  # $0.01 per share
        self.full_data = None  # Placeholder for full data
    
    def _calculate_position_size(self, current_price):
        """Use your existing risk parameters"""
        # risk_per_trade = self.initial_capital * self.predictor.risk_params['per_trade_risk']
        # atr = self.predictor.data['ATR'].iloc[-1]
        # return risk_per_trade / (atr * current_price)
        risk_per_trade = self.portfolio['cash'] * self.predictor.risk_params['per_trade_risk']
        return risk_per_trade / current_price
         
    # The version using ML model
    # def run_backtest(self, start_date, end_date):
    #     """More robust date handling"""
    #     try:
    #         import pandas_market_calendars as mcal
    #         import pandas as pd
    #         import numpy as np
            
    #         nyse = mcal.get_calendar("NYSE")
    #         schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    #         if schedule.empty:
    #             print(f"No trading days between {start_date} and {end_date}")
    #             return pd.DataFrame(), {'error': 'No trading days'}
                
    #         dates = schedule.index.tz_localize(None)
    #         print('First date:', dates[0])
    #         print('Last date:', dates[-1])
    #     except Exception as e:
    #         print(f"Date error: {str(e)}")
    #         return pd.DataFrame(), {'error': str(e)}
    
    #     # if rebalance_frequency == 'weekly':
    #     #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    #     # elif rebalance_frequency == 'monthly':
    #     #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BM')
    #     # elif rebalance_frequency == 'quarterly':
    #     #     rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='BQ')
    #     # else:
    #     #     raise ValueError("rebalance_frequency must be 'weekly', 'monthly', or 'quarterly'")
        

    #     # Store original full data
    #     full_data = self.predictor.data.copy()# data till today and so no end date is needed for the stock predictor
    #     self.full_data = full_data
        
    #     # Get signal from new model once three days
    #     i = 0
    #     first_date = dates[0]
    #     for date in dates:
    #         # Make sure date exists in our data
    #         if date not in full_data.index:
    #             print(f"Date {date} not in data. Skipping.")
    #             continue

    #         # is_rebalance_day = pd.to_datetime(date) in rebalance_dates

    #         # if is_rebalance_day:
    #         #     print(f"Running model on rebalance date: {date}")



            
    #         if i % 3 == 0 and i != 0: # regenerate signal every 10 days
    #             first_date = date 



                
    #         # self.predictor.end_date = date - pd.Timedelta(days=1)
    #         self.predictor.end_date = first_date
    #         self.predictor.load_data()  # Fresh load with cutoff
    #         # self.predictor.data = self.predictor.data.loc[:date]  
    #         print(f'last data of predictor data: {self.predictor.data.index[-1]}')
    #         i += 1
        
    #         # Filter data up to current date
    #         # self.predictor.data = full_data.loc[:date].copy()
            
    #         # Generate signal using existing code
    #         signal = self.predictor.generate_trading_signal(self.predictor.symbol, horizon = 5)
            
    #         # Execute trade
    #         try:

    #             current_price = full_data['Close'].loc[date]
    #             position_size = self._calculate_position_size(current_price)
    #         except (KeyError, IndexError) as e:
    #             print(f"Data not available for {date}. Error: {e}. Skipping.")
    #             continue
            
    #         # Apply slippage and commission
    #         executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
    #             else current_price * (1 - self.slippage)
            
    #         if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
    #             self._execute_buy(executed_price, position_size, date)
    #         elif signal == 'SELL' and self.predictor.symbol in self.portfolio['positions']:
    #             self._execute_sell(executed_price, date)
    #         else:
    #             print(f"Signal is hold so no trade executed for {self.predictor.symbol} on {date} ")
            
    #         # Update portfolio value
    #         self._update_portfolio_value(date)
            
    #         # Check risk limits
    #         if self._check_daily_loss():
    #             print(f"Daily loss limit hit on {date}. Stopping backtest.")
    #             break
        
    #     # Restore original data
    #     self.predictor.data = full_data
        
    #     return self._generate_report()
        
    # def run_backtest(self, start_date, end_date):
    #     """Backtest using get_entry_signal instead of generate_trading_signal"""
    #     try:
    #         import pandas_market_calendars as mcal
    #         import pandas as pd
    #         import numpy as np
            
    #         nyse = mcal.get_calendar("NYSE")
    #         schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    #         if schedule.empty:
    #             print(f"No trading days between {start_date} and {end_date}")
    #             return pd.DataFrame(), {'error': 'No trading days'}
                
    #         dates = schedule.index.tz_localize(None)
    #         print('First date:', dates[0])
    #         print('Last date:', dates[-1])
    #     except Exception as e:
    #         print(f"Date error: {str(e)}")
    #         return pd.DataFrame(), {'error': str(e)}

    #     # Store original full data
    #     full_data = self.predictor.data.copy()
    #     self.full_data = full_data
        
    #     # Get signal from new model once every three days
    #     i = 0
    #     first_date = dates[0]
    #     for date in dates:
    #         # Make sure date exists in our data
    #         if date not in full_data.index:
    #             print(f"Date {date} not in data. Skipping.")
    #             continue
                
    #         if i % 2 == 0 and i != 0: # regenerate signal every 3 days
    #             first_date = date 
                
    #         # Update predictor to use data up to current date
    #         self.predictor.end_date = first_date
    #         self.predictor.load_data()  
    #         print(f'Last data of predictor data: {self.predictor.data.index[-1]}')
    #         i += 1
        
    #         # Generate signal using get_entry_signal instead of generate_trading_signal
    #         try:
    #             decision, confidence, rationale, levels = self.predictor.get_entry_signal(self.predictor.symbol)
    #             signal = decision  # BUY, SELL, or HOLD
                
    #             print(f"Date: {date}, Signal: {signal}, Confidence: {confidence}%")
    #             print(f"Rationale: {rationale}")
    #         except Exception as e:
    #             print(f"Error generating signal for {date}: {str(e)}")
    #             continue
            
    #         # Execute trade
    #         try:
    #             current_price = full_data['Close'].loc[date]
    #             position_size = self._calculate_position_size(current_price)
    #         except (KeyError, IndexError) as e:
    #             print(f"Data not available for {date}. Error: {e}. Skipping.")
    #             continue
            
    #         # Apply slippage and commission
    #         executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
    #             else current_price * (1 - self.slippage)
            
    #         if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
    #             self._execute_buy(executed_price, position_size, date)
    #         elif signal == 'SELL' and self.predictor.symbol in self.portfolio['positions']:
    #             self._execute_sell(executed_price, date)
    #         else:
    #             print(f"Signal is {signal} so no trade executed for {self.predictor.symbol} on {date}")
            
    #         # Update portfolio value
    #         self._update_portfolio_value(date)
            
    #         # Check risk limits
    #         if self._check_daily_loss():
    #             print(f"Daily loss limit hit on {date}. Stopping backtest.")
    #             break
        
    #     # Restore original data
    #     self.predictor.data = full_data
        
    #     return self._generate_report() 

    def run_backtest(self, start_date, end_date):
        """Backtest using get_entry_signal instead of generate_trading_signal"""
        try:
            import pandas_market_calendars as mcal
            import pandas as pd
            import numpy as np
            
            # Get trading days for the specified period
            nyse = mcal.get_calendar("NYSE")
            schedule = nyse.schedule(start_date=start_date, end_date=end_date)
            if schedule.empty:
                print(f"No trading days between {start_date} and {end_date}")
                return pd.DataFrame(), {'error': 'No trading days'}
                
            dates = schedule.index.tz_localize(None)
            print('First date:', dates[0])
            print('Last date:', dates[-1])
        except Exception as e:
            print(f"Date error: {str(e)}")
            return pd.DataFrame(), {'error': str(e)}

        # Store original full data
        full_data = self.predictor.data.copy()
        self.full_data = full_data
        
        # Reset portfolio for backtest
        self.portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'value_history': [],
            'transactions': []
        }
        
        # Regenerate signals periodically (every 2 days)
        i = 0
        first_date = dates[0]
        for date in dates:
            # Skip dates that don't exist in our data
            if date not in full_data.index:
                print(f"Date {date} not in data. Skipping.")
                continue
                
            # Re-train model every N days to avoid look-ahead bias
            if i % 2 == 0 and i != 0:  # Regenerate signal every 2 days
                first_date = date
                
            # Update predictor to use ONLY data up to current date
            # This is critical to prevent look-ahead bias
            self.predictor.end_date = first_date
            self.predictor.load_data()  # Fresh load with cutoff at first_date
            
            print(f'Training data ends at: {self.predictor.data.index[-1]}')
            i += 1
        
            # Generate signal using get_entry_signal
            try:
                # Use ONLY data available at this point to generate signal
                decision, confidence, rationale, levels = self.predictor.get_entry_signal(self.predictor.symbol)
                signal = decision  # BUY, SELL, or HOLD
                
                print(f"Date: {date}, Signal: {signal}, Confidence: {confidence}%")
                print(f"Rationale: {rationale}")
            except Exception as e:
                print(f"Error generating signal for {date}: {str(e)}")
                continue
            
            # Execute trade based on today's open price (not close - avoid look-ahead bias)
            try:
                # Use opening price for execution to avoid look-ahead bias
                current_price = full_data.loc[date, 'Open'] if 'Open' in full_data.columns else full_data.loc[date, 'Close']
                position_size = self._calculate_position_size(current_price)
            except (KeyError, IndexError) as e:
                print(f"Data not available for {date}. Error: {e}. Skipping.")
                continue
            
            # Apply slippage and commission
            executed_price = current_price * (1 + self.slippage) if signal == 'BUY' \
                else current_price * (1 - self.slippage)
            
            # Execute trades based on signal
            if signal == 'BUY' and self.portfolio['cash'] > executed_price * position_size:
                self._execute_buy(executed_price, position_size, date)
            elif signal == 'SELL':
                if self.predictor.symbol in self.portfolio['positions']:
                    self._execute_sell(executed_price, date)
                else:
                    # Optional: naked shorting if you want to match the Aptos implementation
                    position_size = self._calculate_position_size(executed_price)
                    self.portfolio['cash'] += executed_price * position_size - self.commission * position_size
                    self.portfolio['transactions'].append(('SELL', executed_price, position_size, date))
                    self.portfolio['positions'][self.predictor.symbol] = {
                        'qty': -position_size, 
                        'Avg_entry_price': executed_price,
                    }
                    print(f"SELL (short) executed on {date}: {position_size} shares at ${executed_price:.2f}")
            
            # Update portfolio value using closing price
            closing_price = full_data.loc[date, 'Close']
            self._update_portfolio_value(date)
            
            # Check risk limits
            if self._check_daily_loss():
                print(f"Daily loss limit hit on {date}. Stopping backtest.")
                break
        
        # Restore original data
        self.predictor.data = full_data
        
        return self._generate_report()


# ------------------------------------------------------------------------------------------------------------

    def _execute_buy(self, price, qty, date):
        cost = price * qty + self.commission * qty
        self.portfolio['cash'] -= cost
        # Want to ensure we don't overwrite existing positions but add to them
        if self.predictor.symbol in self.portfolio['positions']:
            self.portfolio['positions'][self.predictor.symbol]['qty'] += qty
            self.portfolio['positions'][self.predictor.symbol]['Avg_entry_price'] = (
                self.portfolio['positions'][self.predictor.symbol]['Avg_entry_price'] *
                self.portfolio['positions'][self.predictor.symbol]['qty'] + price * qty 
            ) / (self.portfolio['positions'][self.predictor.symbol]['qty'] + qty)
        else:
            # Initialize new position
            self.portfolio['positions'][self.predictor.symbol] = {
                'qty': qty, 
                'Avg_entry_price': price,
                # 'entry_date': date
            }



        self.portfolio['transactions'].append(('BUY', price, qty, date))
        print(f"BUY executed on {date}: {qty} shares at ${price:.2f}")



    def _execute_sell(self, price, date):
        if self.predictor.symbol not in self.portfolio['positions']:
            print(f"No position to sell for {self.predictor.symbol} but want to short")
            # Naked shorting
            qty = self._calculate_position_size(price)
            self.portfolio['cash'] += price * qty - self.commission * qty
            self.portfolio['transactions'].append(('SELL', price, qty, date))
            self.portfolio['positions'][self.predictor.symbol] = {
                'qty': -qty, 
                'Avg_entry_price': price,
                # 'entry_date': date
            }
            print(f"SELL executed on {date}: {qty} shares at ${price:.2f}")
            return
        # Option 1: liquidate all positions
        # position = self.portfolio['positions'].pop(self.predictor.symbol)
        # proceeds = price * position['qty'] - self.commission * position['qty']
        # self.portfolio['cash'] += proceeds
        # profit = proceeds - (position['Avg_entry_price'] * position['qty'] + self.commission * position['qty'])
        # self.portfolio['transactions'].append(('SELL', price, position['qty'], date))
        # print(f"SELL executed on {date}: {position['qty']} shares at ${price:.2f}, profit: ${profit:.2f}")
        
        # Option 2: partial liquidation from the postion by amount of shares calculated
        position = self.portfolio['positions'][self.predictor.symbol]
        qty = self._calculate_position_size(price)
        if qty >= position['qty']:
            qty = position['qty']
            self.portfolio['positions'].pop(self.predictor.symbol)

        position['qty'] -= qty
        proceeds = price * qty - self.commission * qty
        self.portfolio['cash'] += proceeds
        profit = proceeds - (position['Avg_entry_price'] * qty + self.commission * qty)
        self.portfolio['transactions'].append(('SELL', price, qty, date))
        print(f"SELL executed on {date}: {qty} shares at ${price:.2f}, profit: ${profit:.2f}")





    def _check_daily_loss(self):
        """Use your existing risk management"""
        if len(self.portfolio['value_history']) < 2:
            return False
        daily_pct = (self.portfolio['value_history'][-1]['value'] / 
                    self.portfolio['value_history'][-2]['value']) - 1
        return daily_pct < self.predictor.risk_params['daily_loss_limit']
 
    def _update_portfolio_value(self, date):
        position_value = 0
        for sym, pos in self.portfolio['positions'].items():
            try:
                if sym == self.predictor.symbol:  # We're only tracking one symbol
                    # current_price = self.predictor.data['Close'].iloc[-1]
                    # the current price at the data, not the last date
                    # current_price = self.predictor.data['Close'].loc[date]
                    current_price = self.full_data['Close'].loc[date]
                    if pos['qty'] < 0:
                        position_value -= -pos['qty'] * current_price
                    else:
                        position_value += pos['qty'] * current_price
            except (KeyError, IndexError) as e:
                print(f"Error updating portfolio value: {e}")
        
        total_value = self.portfolio['cash'] + position_value
        
        # Ensure consistent data format
        self.portfolio['value_history'].append({
            'date': pd.to_datetime(date),
            'value': total_value,
            'cash': self.portfolio['cash'],
        })
        print(f"Portfolio value on {date}: ${total_value:.2f}")

    def _generate_report(self):
        """More robust report generation"""
        import pandas as pd
        import numpy as np
        
        if not self.portfolio['value_history']:
            print("No portfolio history to generate report")
            return pd.DataFrame(), {'error': 'No trades executed'}
        
        try:
            df = pd.DataFrame(self.portfolio['value_history'])
            df = df.set_index('date').sort_index()
            
            if df.empty:
                return df, {'error': 'Empty portfolio history'}
                
            returns = df['value'].pct_change().dropna()
            
            if len(returns) < 2:
                return df, {'error': 'Insufficient data for metrics'}
            
            report = {
                'sharpe': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': (df['value'] / df['value'].cummax() - 1).min(),
                'total_return': df['value'].iloc[-1] / self.initial_capital - 1,
                'num_trades': len(self.portfolio['transactions']),
                'win_rate': self._calculate_win_rate()
            }
            print(f"Report generated: {report}")
            return df, report
            
        except Exception as e:
            print(f"Report generation error: {str(e)}")
            return pd.DataFrame(), {'error': str(e)}

    # def _calculate_win_rate(self, history_df):
    #     """Safer win rate calculation"""
    #     # Definition: Winning trades means the trade makes the value of profolio higher than the previous trade (whether or not the position is liquidated)
    #     # We can use the history_df to calculate the win rate
    #     return (history_df['value'].diff().dropna()>0).astype(int).sum() / len(self.portfolio['transactions']) if len(self.portfolio['transactions']) > 0 else 0.0
    def _calculate_win_rate(self):
        """Calculate win rate from completed trades. Only count winning trades when the position is liquidated"""
        buy_trades = [(p, d, q) for t, p, q, d in self.portfolio['transactions'] if t == 'BUY']
        sell_trades = [(p, d, q) for t, p, q, d in self.portfolio['transactions'] if t == 'SELL']
        
        if not sell_trades:
            return 0.0
        
        winning_trades = 0
        
        # Match buys with sells sequentially (FIFO)
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i][0]
            sell_price = sell_trades[i][0]
            
            if sell_price > buy_price:
                winning_trades += 1
        
        return winning_trades / len(sell_trades) #only count winning
        

        

class StressTester(Backtester):
    """Stress tests using your existing strategy"""
    
    def _apply_market_crash(self, date):
        """Simulate flash crash scenario"""
        if np.random.rand() < 0.05:  # 5% chance daily
            self.predictor.data.loc[date:, 'Close'] *= 0.9  # 10% drop
            self.predictor.data['Volatility'] *= 2  # Spike volatility
            
    def _apply_liquidity_crisis(self, date):
        """Simulate bid-ask spread widening"""
        if np.random.rand() < 0.03:  # 3% chance daily
            self.slippage = 0.01  # 1% slippage
            self.commission = 0.001  # $0.1 per share
            
    def run_stress_test(self, start_date, end_date):
        """Run stress test using your existing strategy"""
        nyse = mcal.get_calendar("NYSE")
        dates = nyse.schedule(start_date=start_date, end_date=end_date).index
        for date in dates:
            # Apply stress events
            self._apply_market_crash(date)
            self._apply_liquidity_crisis(date)
            
            # Run normal backtest
            super().run_backtest(date, date)
            
        return self._generate_report()
    
    def _run_stress_tests(self, history_df):
        """Run stress tests on the strategy"""
        if len(history_df) < 30:  # Need sufficient data
            return {'stress_test': 'Insufficient data'}
        
        results = {}
        returns = history_df['value'].pct_change().dropna()
        
        # Test 1: Worst week performance
        weekly_returns = (history_df['value'].resample('W').last().pct_change().dropna())
        results['worst_week'] = weekly_returns.min()
        
        # Test 2: Performance in high volatility periods
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        high_vol_returns = returns[rolling_vol > rolling_vol.quantile(0.75)]
        results['high_vol_performance'] = high_vol_returns.mean() * 252 if not high_vol_returns.empty else 0
        
        # Test 3: Monte Carlo simulation - 100 paths
        mc_results = self._monte_carlo_simulation(returns, paths=100)
        results['mc_5pct_var'] = mc_results['5pct_var']
        results['mc_worst_drawdown'] = mc_results['worst_drawdown']
        
        return {'stress_tests': results}
    
    def _monte_carlo_simulation(self, returns, paths=100, horizon=252):
        """Run Monte Carlo simulation to test strategy robustness"""
        sim_returns = np.random.choice(
            returns.values,
            size=(paths, horizon),
            replace=True
        )
        
        # Convert returns to paths
        sim_paths = np.cumprod(1 + sim_returns, axis=1)
        
        # Calculate metrics
        final_values = sim_paths[:, -1]
        drawdowns = np.zeros(paths)
        
        for i in range(paths):
            drawdowns[i] = np.min(sim_paths[i] / np.maximum.accumulate(sim_paths[i])) - 1
        
        return {
            '5pct_var': np.percentile(final_values, 5) - 1,  # 5% VaR
            'worst_drawdown': np.min(drawdowns)  # Worst drawdown across all sims
        }






# Example usage
if __name__ == "__main__":
    predictor = StockPredictor("AAPL", start_date="2020-01-01")
    predictor.load_data()
    print(predictor.data.head())

