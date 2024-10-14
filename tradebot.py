import os
import json
import ccxt
import pandas as pd
import numpy as np
from sqlalchemy import func
import asyncio
from web3 import Web3
from web3.exceptions import ContractLogicError
from sqlalchemy import inspect
import subprocess
import time

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# Create an SQLite database connection (it creates a file 'trades.db')
engine = create_engine('sqlite:///trades.db', echo=False)
Base = declarative_base()


class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)               # Unique trade ID
    signal = Column(String)                              # Signal (BUY or SELL)
    price = Column(Float)                                # Price at which the trade was executed
    tp = Column(Float)                                   # Take profit price
    sl = Column(Float)                                   # Stop loss price
    trade_id = Column(Integer)                           # Trade ID from the system
    equity = Column(Float, nullable=False, default=0.0)  # Equity for the trade
    is_open = Column(Boolean, default=True)              # Whether the trade is still open

    def __repr__(self):
        return f"<Trade(id={self.id}, signal='{self.signal}', price={self.price}, is_open={self.is_open})>"

# Create the table if it doesn't exist
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

session.query(Trade).delete()
session.commit()  # Make sure to commit the changes

# Initialize exchange connection
exchange = ccxt.binance({
    'enableRateLimit': True,
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET')
})

# Define the trading pair and indicators
pair = 'ETH/USDT'
timeframe = '30m'
periods = 100

# Contract parameters
TAKE_PROFIT = 1.05  # e.g., 5% profit
STOP_LOSS = 0.95    # e.g., 5% loss
trade_id = None
entry_price = None

# Technical analysis functions
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

async def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def calculate_cci(df, period=10):
    TP = (df['high'] + df['low'] + df['close']) / 3
    sma = TP.rolling(window=period).mean()
    mean_deviation = TP.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (TP - sma) / (0.015 * mean_deviation)
    return cci

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Function to fetch data and calculate signal
async def generate_signal():
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=periods)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['EMA25'] = calculate_ema(df['close'], 25)
    df['RSI4'] = await calculate_rsi(df['close'], 4)
    df['CCI10'] = await calculate_cci(df, 10)
    df['ATR14'] = calculate_atr(df, 14)

    latest = df.iloc[-1]
    signal = None
    tp = None  
    sl = None  

    if latest['RSI4'] < 30 and latest['CCI10'] < -100 and latest['close'] < latest['EMA25']:
        tp = latest['close'] - (1.5 * latest['ATR14'])
        sl = latest['close'] + (1.5 * latest['ATR14'])
        signal = 'SELL'

    elif latest['RSI4'] > 70 and latest['CCI10'] > 100 and latest['close'] > latest['EMA25']:
        tp = latest['close'] + (1.5 * latest['ATR14'])   
        sl = latest['close'] - (1.5 * latest['ATR14'])
        signal = 'BUY'

    return signal, latest['close'], tp, sl


# Main routine to periodically check for signals and execute trades
async def main():
    global trade_id, entry_price

    # take_snapshot()
    while True:
        signal, price, tp, sl = await generate_signal()

        print(f"There is a {signal} signal")

        # Check if there's an open trade
        open_trade_exists = session.query(Trade).filter_by(is_open=True).first()

        if signal is not None and not open_trade_exists:
            # Reset the Hardhat node before proceeding
            # revert_to_snapshot()
            print(f"Executing {signal} at {price}.")
            hash = 0x00000000000000000000000000000000000000

            # Save the trade to the database
            new_trade = Trade(
                signal=signal,
                price=price,
                tp=tp,
                sl=sl,
                trade_id=hash,
                is_open=True
            )
            session.add(new_trade)
            session.commit()  # Save the changes to the database
            print(f"Trade saved with ID {new_trade.id}")
        else:
            print(f"Open trade exists: {open_trade_exists}")

        if open_trade_exists:
            # Assuming you have a way to get the current price in the context of this check
            current_price = price 

            # If the signal is 'BUY' and the current price is greater or less than the stop loss
            if open_trade_exists.signal == 'BUY':
                if current_price > open_trade_exists.tp:
                    print(f"Take Profit for Buy Trade Hit: price:{current_price}.")

                    # Reset the Hardhat node before proceeding
                    # revert_to_snapshot()

                    pl = current_price - open_trade_exists.price

                    open_trade_exists.equity = pl

                    open_trade_exists.is_open = False

                    print("Trade Closed")

                    print(f"Trade Objects are {open_trade_exists}")
                    # # Action: Close the trade, alert the user, etc.
                    # close_trade(open_trade_exists.id)

                    # Query to sum the equity of all trades
                    total_equity = session.query(func.sum(Trade.equity)).scalar()

                    print(f"Total Equity: {total_equity}")

                    # Commit the updated equity to the database
                    session.add(open_trade_exists)
                    session.commit()

                if current_price < open_trade_exists.sl:
                    print(f"Stop Loss for Buy Trade Hit: price:{current_price}.")

                    # Reset the Hardhat node before proceeding
                    # revert_to_snapshot()

                    print(f"Trade Objects are {open_trade_exists}")

                    pl = current_price - open_trade_exists.price

                    open_trade_exists.equity = pl
                    open_trade_exists.is_open = False
                    print("Trade Closed")

                    total_equity = session.query(func.sum(Trade.equity)).scalar()

                    print(f"Total Equity: {total_equity}")
                    # Action: Close the trade, alert the user, etc.
                    # close_trade(open_trade_exists.id)

                    # Commit the updated equity to the database
                    session.add(open_trade_exists)
                    session.commit()
                    # Query to sum the equity of all trades
                    
                else:
                    print(f"Current price {current_price} is within safe range. No action needed.")
            else:
                if current_price < open_trade_exists.tp:
                    print(f"Take Profit for Sell Trade Hit: price:{current_price}.")

                    # Reset the Hardhat node before proceeding
                    # revert_to_snapshot()

                    print(f"Trade Objects are {open_trade_exists}")

                    pl = open_trade_exists.price - current_price

                    open_trade_exists.equity = pl

                    open_trade_exists.is_open = False
                    print("Trade Closed")

                    # Action: Close the trade, alert the user, etc.
                    # close_trade(open_trade_exists.id)

                    
                    # Commit the updated equity to the database
                    session.add(open_trade_exists)
                    session.commit()
                    # Query to sum the equity of all trades
                    total_equity = session.query(func.sum(Trade.equity)).scalar()

                    print(f"Total Equity: {total_equity}")

                if current_price > open_trade_exists.sl:
                    print(f"Stop Loss for Sell Trade Hit: price:{current_price}.")

                    # Reset the Hardhat node before proceeding
                    # revert_to_snapshot()

                    print(f"Trade Objects are {open_trade_exists}")

                    pl = open_trade_exists.price - current_price

                    open_trade_exists.equity = pl
                    open_trade_exists.is_open = False
                    print("Trade Closed")

                    # Action: Close the trade, alert the user, etc.
                    # close_trade(open_trade_exists.id)
                    
                    # Commit the updated equity to the database
                    session.add(open_trade_exists)
                    session.commit()
                    # Query to sum the equity of all trades
                    total_equity = session.query(func.sum(Trade.equity)).scalar()

                    print(f"Total Equity: {total_equity}")
                else:
                    print(f"Current price {current_price} is within safe range. No action needed.")

        await asyncio.sleep(60) 

asyncio.run(main())
