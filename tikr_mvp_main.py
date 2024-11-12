#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 10 15:15:20 2024

@author dhaneor
"""
import logging
import threading
import asyncio
import numpy as np
import pandas as pd
from queue import Queue, Empty
from threading import Event

from src.rawi import ohlcv_repository as repo
from src.analysis import strategy_builder as sb
from src.analysis import strategy_backtest as bt
from src.analysis.backtest import statistics as st
from src.analysis.models.position import extract_positions, Position
from src.analysis.telegram_signal import TelegramSignal
from src.analysis.strategies.definitions import s_breakout
from src.backtest import result_stats as rs

# set up logging
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

# instantiate strategy
strategy = sb.build_strategy(s_breakout)

ohlcv_request = {
    "exchange": "binance",
    "symbol": strategy.symbol,
    "interval": strategy.interval,
}

RISK_LEVEL = 7
MAX_LEVERAGE = 1.5


def display_results(df: pd.DataFrame) -> None:
    # add drawdown information to dataframe
    df = rs.calculate_stats(df, initial_capital=10_000)

    # calculate the statistics (like sharpe ratio, sortino ratio, ...)
    # for the strategy and HODL
    stats = st.calculate_statistics(df["b.value"].to_numpy())
    stats = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in stats.items()}

    stats_hodl = st.calculate_statistics(df["hodl.value"].to_numpy())
    stats_hodl = {
        k: f"{v:.2f}" if isinstance(v, float) else v for k, v in stats_hodl.items()
        }

    # preprocess the dataframe for display on std out
    df["open time utc"] = pd.to_datetime(df["open time"], unit="ms")
    df.index = pd.to_datetime(df["open time utc"])
    df.drop(columns=["open time utc"], inplace=True)

    incl_cols = [
        "open", "high", "low", "close", "volume",
        "position", "leverage", "buy",
        "buy_size", "buy_at", "sell", "sell_size", "sell_at",
        "b.base", "b.quote", "b.value", "b.drawdown.max",
        "hodl.value", "hodl.drawdown.max",
        ]

    df = df[incl_cols]

    # Apply rounding and formatting to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    df.replace(np.nan, "", inplace=True)
    df.replace(False, ".", inplace=True)
    df.replace("0.0000", "", inplace=True)

    df = df.astype(str)
    df.loc[df["buy_size"] != "", "buy"] = "•"
    df.loc[df["sell_size"] != "", "sell"] = "•"
    df.loc[df["position"] == "1.0000", "position"] = "LONG"
    df.loc[df["position"] == "-1.0000", "position"] = "SHORT"

    # display the last 50 rows of the dataframe & the statistics
    print(df.tail(50))
    logger.info("stats strategy: %s" % stats)
    logger.info("stats hodl    : %s" % stats_hodl)


# ====================================================================================
# Asynchronous Function to Fetch OHLCV Data
async def fetch_ohlcv_data(request: dict, queue: Queue, stop_event: Event):
    try:
        while not stop_event.is_set():
            try:
                ohlcv_data = await repo.process_request(request)
                if not ohlcv_data:
                    logger.warning("No OHLCV data received. Retrying...")
                queue.put(ohlcv_data)
            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {e}")
            finally:
                await asyncio.sleep(5)  # Wait before retrying
    except Exception as e:
        logger.error(f"Error in Async Loop: {e}")

    # await repo.exchange_factory(None)
    # logger.info("Async Event Loop stopped.")


# Function to Run the Async Event Loop in a Separate Thread
def start_async_loop(queue, stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(fetch_ohlcv_data(ohlcv_request, queue, stop_event))
    finally:
        loop.run_until_complete(repo.exchange_factory(None))
        loop.close()


def process_data(ohlcv_data):
    logger.debug(ohlcv_data)

    ohlcv_data = bt.run(
                    data=ohlcv_data.to_dict(),
                    strategy=strategy,
                    risk_level=RISK_LEVEL,
                    max_leverage=MAX_LEVERAGE,
                    initial_capital=10_000
                    )

    # build a dataframe from the results
    df = pd.DataFrame.from_dict(ohlcv_data)

    display_results(df)

    return df


def send_signal(df: pd.DataFrame) -> None:
    """Sends a trading signal to Telegram, based on the results DataFrame."""

    def get_current_position(df: pd.DataFrame) -> Position | None:
        position_manager = extract_positions(df, strategy.symbol)
        return position_manager\
            .get_current_position(strategy.symbol, False)\
            .get_signal()

    try:
        signal = TelegramSignal(get_current_position(df))
        signal.send_signal()
    except Exception as e:
        logger.exception("Error sending Telegram signal: %s", e)


# Main Function to Integrate Everything
def main():
    # Create a Queue for Communication
    ohlcv_queue = Queue()

    # Event to Signal the Async Thread to Stop
    stop_event = Event()

    # Start the Async Event Loop in a Separate Thread
    logger.debug("Starting Async Event Loop in thread ...")
    async_thread = threading.Thread(
        target=start_async_loop,
        args=(ohlcv_queue, stop_event),
        daemon=True
        )
    async_thread.start()

    # Main Loop: Process OHLCV Data from the Queue
    counter = 0
    try:
        while counter < 2:
            try:
                # Wait for data with a timeout to allow graceful shutdown
                ohlcv_data = ohlcv_queue.get()

                if not isinstance(ohlcv_data, repo.Response):
                    raise ValueError("Invalid data received")

                if not ohlcv_data.data:
                    raise ValueError("No OHLCV data received")

            except Empty:
                print("No new data received. Waiting...")
            except ValueError as e:
                logger.error("Error processing data: %s", e)
                continue
            else:
                df = process_data(ohlcv_data)
                send_signal(df)
            finally:
                ohlcv_queue.task_done()
                counter += 1

    except KeyboardInterrupt:
        print("Received shutdown signal. Exiting...")
    finally:
        # Signal the async thread to stop
        stop_event.set()
        async_thread.join(timeout=5)


# Run the Main Function
if __name__ == "__main__":
    main()
