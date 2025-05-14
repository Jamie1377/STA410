# Import necessary libraries
from stock_prediction.core import StockPredictor
from stock_prediction.utils import optimize_lookback
import schedule
import time
import pandas as pd
from datetime import date, timedelta, datetime
import yfinance as yf
from pandas_market_calendars import get_calendar
from stock_prediction.core import Backtester, StressTester
nyse = get_calendar("NYSE")

# API imports
api_key = "UR_API_KEY"
secret_key = "UR_SECRET_KEY"
paper = True
# DO not change this
trade_api_url = None
trade_api_wss = None
data_api_url = None
stream_data_wss = None

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
    QueryOrderStatus,
    StopLimitOrderRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass


# Initialize the Alpaca trading client
trading_client = TradingClient(
    api_key=api_key, secret_key=secret_key, paper=True, url_override=trade_api_url
)

# Get the list of top companies in the energy and technology sectors
energy_sector = yf.Sector("energy").top_companies.index
technology_sector = yf.Sector("technology").top_companies.index
crypto = list(yf.Lookup(query="USD").cryptocurrency.copy().index)  # ['symbol'].values
volatile_symbols = [
    "CERO",
    "BOWN",
    "CNEY",
    "JZXN",
    "AREB",
    "AGMH",
    "PET",
    "XELB",
    "OMEX",
    "AREN",
    "JYD",
    "CMLS",
    "SWKH",
    "TMC",
    "AGRI",
    "CTHR",
    "INBK",
    "WLGS",
    "AMBP",
    "FFAI",
    "RLMD",
    "TNMG",
    "UOKA",
    "BUJA",
    "CYH",
    "CDIO",
    "VSME",
    "SGMA",
    "FAMI",
    "ABP",
    "GSHD",
    "BSLK",
    "ASGN",
    "SHYF",
    "LIXT",
    "TLSA",
    "PHIO",
    "SHFS",
    "ENSC",
    "MXL",
    "TOI",
    "RZLV",
    "ABTS",
    "FI",
    "ALBT",
    "ABLV",
    "PRPO",
    "APCX",
    "HOLO",
    "AMTB",
    "SISI",
    "CNSP",
    "SBEV",
    "CHDN",
    "VS",
    "SLXN",
    "XTNT",
    "VCIG",
    "DUO",
    "BLZE",
    "INTS",
    "SXTC",
    "PI",
    "BTOG",
    "GNS",
    "XFOR",
    "SRM",
    "OBLG",
    "ANGH",
    "FARO",
    "BW",
    "EPOW",
    "USAR",
    "BCAB",
    "NTRP",
    "CLGN",
    "PLUR",
    "SMX",
    "MODV",
]
energy_symbols = [
    "COP",
    "TPL",
    "APA",
    "AR",
    "EOG",
    "LLY",
    "CHX",
    "BKR",
    "NOV",
]  # Example energy stocks


def market_day_check(current_date=date.today()):
    """
    Returns the next valid trading day using NYSE calendar.
    """
    # Get NYSE calendar
    nyse = get_calendar("NYSE")

    # Convert input to pandas Timestamp if it isn't already
    current_date = pd.Timestamp(current_date)

    # Get valid trading days for a range (using 10 days to be safe)
    schedule = nyse.schedule(
        start_date=current_date, end_date=current_date + pd.Timedelta(days=10)
    )

    # Check if the current date is in the schedule
    return current_date in schedule.index


################################################################### Trading Job


def trading_job():
    """
    Execute trades at market open
    """
    # Check if the market is open
    if not market_day_check():
        print("Market is closed. Exiting...")
        return
    for symbol in energy_sector[:10]:
        predictor = StockPredictor(
            symbol, start_date="2023-06-01", end_date=date.today()
        )

        try:
            signal = predictor.generate_trading_signal(symbol)
            StockPredictor.execute_trade(symbol, signal, trading_client)
            print(f"Executed {signal} for {symbol}")
        except Exception as e:
            print(f"Trade failed: {str(e)}")


def energy_sector_trading():
    """
    Execute trades at market open (sample version: may add arguments later)
    """
    if not market_day_check() and not crypto:
        print("Market is closed. Exiting...")
        return

    symbols_to_trade = StockPredictor.create_hqm_stocks(
        start_date="2023-06-01"
    ).Symbol.values[:20] # Top 20 stocks with highest momentum
    print("The stocks we are working on are", symbols_to_trade)

    predictors = []
    open_orders = list(set([stock.symbol for stock in trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))]))

    # Initialize predictors
    for symbol in symbols_to_trade:
        predictor = StockPredictor(symbol=symbol, start_date="2023-06-01")
        predictor.load_data()
        predictors.append(predictor)

    # Generate signals with correlation check
    signals = []
    hft_signals = []
    reverse_hft_signals = []
    for predictor, symbol in zip(predictors, symbols_to_trade):
        # Check if the symbol is already in open orders
        if symbol in open_orders:
            print(f"Skipping {symbol} as it is already in open orders.")
            continue
        try:
            signal = predictor.generate_trading_signal(symbol=symbol)
        except Exception as e:
            print(f"Signal generation failed for {symbol}: {str(e)}")
            continue
        try:
            hft_signal = predictor.generate_hft_signals(
                symbol=symbol, profit_target=0.002
            )
            reverse_hft_signal = predictor.generate_hft_signals(
                symbol=symbol, profit_target=0.002
            )
        except Exception as e:
            print(f"HFT signal generation failed for {symbol}: {str(e)}")
            continue

        signals.append((predictor.symbol, signal))
        hft_signals.append((predictor.symbol, hft_signal))
        reverse_hft_signals.append((predictor.symbol, reverse_hft_signal))
    for symbol in symbols_to_trade:
        predictor = next(p for p in predictors if p.symbol == symbol)
        try:
            predictor.execute_hft(symbol=symbol, manual=False, crypto=False)
            print(f"Executed HFT for {symbol}")
        except Exception as e:
            print(f"HFT trade failed for {symbol}: {str(e)}")


def check_entry_points():
    """Check entry points for stocks and execute trades"""
    # try:
    #     # Get all positions
    #     positions = trading_client.get_all_positions()

    #     if not positions:
    #         print("No positions found")
    #         return

    #     print(f"Found {len(positions)} total positions")
    #     profitable_count = 0

    #     for p in positions:
    #         try:
    #             # Convert profit values to float and check profitability
    #             unrealized_pl = (
    #                 float(p.unrealized_pl) if hasattr(p, "unrealized_pl") else 0
    #             )
    #             intraday_pl = (
    #                 float(p.unrealized_intraday_pl)
    #                 if hasattr(p, "unrealized_intraday_pl")
    #                 else 0
    #             )

    #             # Print position info for debugging
    #             print(
    #                 f"Position: {p.symbol}, Qty: {p.qty}, Unrealized P&L: ${unrealized_pl:.2f}, Intraday P&L: ${intraday_pl:.2f}"
    #             )

    #             # Check if position is profitable
    #             if unrealized_pl > 0 or intraday_pl > 0:
    #                 profitable_count += 1
    #                 print(
    #                     f"Profitable position found: {p.symbol} with ${max(unrealized_pl, intraday_pl):.2f} profit"
    #                 )

    #                 # Cancel related open orders first
    #                 related_orders = trading_client.get_orders(
    #                     filter=GetOrdersRequest(
    #                         symbols=[p.symbol], status=QueryOrderStatus.OPEN
    #                     )
    #                 )

    #                 for order in related_orders:
    #                     try:
    #                         trading_client.cancel_order_by_id(order.id)
    #                         print(f"Cancelled order {order.id} for {p.symbol}")
    #                     except Exception as e:
    #                         print(f"Error cancelling order {order.id}: {str(e)}")

    #                 # Close the position with proper error handling
    #                 try:
    #                     # Make sure qty is formatted properly
    #                     qty = abs(float(p.qty))
    #                     if qty > 0:
    #                         close_request = ClosePositionRequest(qty=str(qty))
    #                         result = trading_client.close_position(
    #                             symbol_or_asset_id=p.symbol, close_options=close_request
    #                         )
    #                         print(
    #                             f"✓ Successfully liquidated {p.symbol} position of {p.qty} shares"
    #                         )
    #                     else:
    #                         print(f"⚠ Zero quantity for {p.symbol}, skipping")
    #                 except Exception as e:
    #                     print(f"Error closing position for {p.symbol}: {str(e)}")

    #                     # Fallback to market order if close_position fails
    #                     try:
    #                         side = OrderSide.SELL if float(p.qty) > 0 else OrderSide.BUY
    #                         market_order = MarketOrderRequest(
    #                             symbol=p.symbol,
    #                             qty=abs(float(p.qty)),
    #                             side=side,
    #                             time_in_force=TimeInForce.DAY,
    #                             type=OrderType.MARKET,
    #                         )
    #                         order = trading_client.submit_order(market_order)
    #                         print(
    #                             f"Submitted fallback market order to close {p.symbol}: {order.id}"
    #                         )
    #                     except Exception as e2:
    #                         print(
    #                             f"Fallback order also failed for {p.symbol}: {str(e2)}"
    #                         )

    #         except Exception as e:
    #             print(f"Error processing position {p.symbol}: {str(e)}")
    #             continue

    #     print(
    #         f"Found {profitable_count} profitable positions out of {len(positions)} total positions"
    #     )

    # except Exception as e:
    #     print(f"Failed to get positions: {str(e)}")

    symbols = StockPredictor.create_hqm_stocks(
        start_date="2023-06-01"
    ).Symbol.values # Top 120 stocks with highest momentum
    positions = [pos.symbol for pos in trading_client.get_all_positions()]
    for symbol in symbols:
        if (
            symbol
            in trading_client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
            or symbol in positions
        ):
            print(
                f"Skipping {symbol} as it is already in open orders or has positions."
            )
            continue
        try:
            if date.today().weekday() == 0:
                predictor = StockPredictor(
                    symbol=symbol,
                    start_date=date.today() - pd.Timedelta(days=3),
                    end_date=date.today() + pd.Timedelta(days=1),
                    interval="1m",
                )
            else:
                predictor = StockPredictor(
                    symbol=symbol,
                    start_date=date.today() - pd.Timedelta(days=1),
                    end_date=date.today() + pd.Timedelta(days=1),
                    interval="1m",
                )
            predictor.load_data()
            predictor.data = predictor.data[
                predictor.data.index.date == datetime.today().date()
            ]

            # current strat is bad so reverse it
            decision, confidence, rationale, levels = predictor.get_entry_signal(symbol)
            # Reverse the decision
            decision_copy = decision
            if decision == "BUY":
                decision_copy = "SELL"
                levels["stop_loss"][0], levels["take_profit"][0] = levels["take_profit"][0], levels["stop_loss"][0]
                
            elif decision == "SELL":
                decision_copy = "BUY"
                levels["stop_loss"][0], levels["take_profit"][0] = levels["take_profit"][0], levels["stop_loss"][0]
            
            decision = decision_copy
                
       


            print(f"\n🔍 {symbol} Entry Check:")
            print(f"  Decision: {decision} ({confidence}% confidence)")
            print(f"  Rationale: {rationale}")
            print(f"  Key Levels:")
            print(f"    Current: ${levels['current_price'][0]:.2f}")
            print(f"    Stop Loss: ${levels['stop_loss'][0]:.2f}")
            print(f"    Take Profit: ${levels['take_profit'][0]:.2f}")
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
        if symbol in trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        ):
            print(f"Skipping {symbol} as it is already in open orders.")
            continue

        if decision != "HOLD":
            # Execute trade with proper order type
            if decision == "BUY":
                try:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(predictor._calculate_position_size()),
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(
                            limit_price=round(levels["take_profit"][0], 2)
                        ),
                        stop_loss=StopLossRequest(
                            stop_price=round(levels["stop_loss"][0], 2)
                        ),
                    )
                except Exception as e:
                    print(f"Error in order request for {symbol}: {str(e)}")
                    continue
            else:
                try:
                   
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=round(predictor._calculate_position_size()),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(
                            limit_price=round(levels["take_profit"][0], 2)
                        ),
                        stop_loss=StopLossRequest(
                            stop_price=round(levels["stop_loss"][0], 2)
                        ),
                    )
                except Exception as e:
                    print(f"Error in order request for {symbol}: {str(e)}")
                    continue
            # Submit the order
            try:
                trading_client.submit_order(order)
                print(f"  ⚡ Order submitted at {order.limit_price:.2f}")
                print(f"  Order ID: {order.id}")
            except Exception as e:
                print(f"Error in submitting order for {symbol}: {str(e)}")
                continue


def close_the_postions():
    """Execute trades at market open"""
    # liquidate if there is profit for current positions
    try:
        # Get all positions
        positions = trading_client.get_all_positions()

        if not positions:
            print("No positions found")
            return

        print(f"Found {len(positions)} total positions")
        profitable_count = 0

        for p in positions:
            try:
                # Convert profit values to float and check profitability
                unrealized_pl = (
                    float(p.unrealized_pl) if hasattr(p, "unrealized_pl") else 0
                )
                intraday_pl = (
                    float(p.unrealized_intraday_pl)
                    if hasattr(p, "unrealized_intraday_pl")
                    else 0
                )

                # Print position info for debugging
                print(
                    f"Position: {p.symbol}, Qty: {p.qty}, Unrealized P&L: ${unrealized_pl:.2f}, Intraday P&L: ${intraday_pl:.2f}"
                )

                # Check if position is profitable
                if unrealized_pl > 0 or intraday_pl > 0:
                    profitable_count += 1
                    print(
                        f"Profitable position found: {p.symbol} with ${max(unrealized_pl, intraday_pl):.2f} profit"
                    )

                    # Cancel related open orders first
                    related_orders = trading_client.get_orders(
                        filter=GetOrdersRequest(
                            symbols=[p.symbol], status=QueryOrderStatus.OPEN
                        )
                    )

                    for order in related_orders:
                        try:
                            trading_client.cancel_order_by_id(order.id)
                            print(f"Cancelled order {order.id} for {p.symbol}")
                        except Exception as e:
                            print(f"Error cancelling order {order.id}: {str(e)}")

                    # Close the position with proper error handling
                    try:
                        # Make sure qty is formatted properly
                        qty = abs(float(p.qty))
                        if qty > 0:
                            close_request = ClosePositionRequest(qty=str(qty))
                            result = trading_client.close_position(
                                symbol_or_asset_id=p.symbol, close_options=close_request
                            )
                            print(
                                f"✓ Successfully liquidated {p.symbol} position of {p.qty} shares"
                            )
                        else:
                            print(f"⚠ Zero quantity for {p.symbol}, skipping")
                    except Exception as e:
                        print(f"Error closing position for {p.symbol}: {str(e)}")

                        # Fallback to market order if close_position fails
                        try:
                            side = OrderSide.SELL if float(p.qty) > 0 else OrderSide.BUY
                            market_order = MarketOrderRequest(
                                symbol=p.symbol,
                                qty=abs(float(p.qty)),
                                side=side,
                                time_in_force=TimeInForce.DAY,
                                type=OrderType.MARKET,
                            )
                            order = trading_client.submit_order(market_order)
                            print(
                                f"Submitted fallback market order to close {p.symbol}: {order.id}"
                            )
                        except Exception as e2:
                            print(
                                f"Fallback order also failed for {p.symbol}: {str(e2)}"
                            )

            except Exception as e:
                print(f"Error processing  {p.symbol}: {str(e)}")
                continue

        print(
            f"Found {profitable_count} profitable positions out of {len(positions)} total positions"
        )

    except Exception as e:
        print(f"Failed to get positions: {str(e)}")



# autotrade.py
def backtesting_job():
    symbols = ['AAPL', 'MSFT', 'GOOG']  # Your target symbols
    
    for symbol in symbols:
        predictor = StockPredictor(symbol, start_date="2020-01-01", end_date="2022-01-01")
        predictor.load_data()
        
        # Normal backtespositiont
        print(f"Running backtest for {symbol}")
        backtester = Backtester(predictor)
        history, report = backtester.run_backtest("2022-01-01", "2023-01-01")
        print(f"Backtest Results for {symbol}:")
        print(f"Sharpe: {report['sharpe']:.2f}")
        print(f"Max Drawdown: {report['max_drawdown']:.2%}")
        
        # Stress test
        stress_tester = StressTester(predictor)
        _, stress_report = stress_tester.run_stress_test("2022-01-01", "2023-01-01")
        print(f"Stress Test Results for {symbol}:")
        print(f"Stress Sharpe: {stress_report['sharpe']:.2f}")
        print(f"Stress Drawdown: {stress_report['max_drawdown']:.2%}")

# Add to scheduled jobs
# schedule.every().day.at("17:00").do(backtesting_job)  # Run after market close
# backtesting_job()

# Schedule the trading job

# # Run hourly during market hours

schedule.every(1).minute.do(close_the_postions)
schedule.every(1).minutes.do(check_entry_points)
# schedule.every(3).minutes.do(energy_sector_trading)

# # Run immediately on script start
# energy_sector_trading()
close_the_postions()
check_entry_points()
close_the_postions()

while True:
    schedule.run_pending()
    time.sleep(5)
