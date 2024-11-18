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
import os
import pandas as pd
from queue import Queue, Empty
from threading import Event

from src.rawi import ohlcv_repository as repo
from src.analysis import strategy_builder as sb
from src.analysis import strategy_backtest as bt
from src.analysis.backtest import statistics as st
from src.analysis.models.position import Positions
from src.analysis import telegram_signal as ts
from src.backtest import result_stats as rs
from src.plotting.minerva import TikrChart as Chart
from tikr_mvp_strategy import mvp_strategy

# set up logging
LOG_LEVEL = logging.INFO

logger = logging.getLogger("main")
logger.setLevel(LOG_LEVEL)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
    )
)
logger.addHandler(ch)

# ================================ Configuration =====================================
# instantiate strategy
strategy = sb.build_strategy(mvp_strategy)
strategy.name = "Safe HODL Strategy by Gregorovich"

RISK_LEVEL = 0  # define the risk level for the strategy / position sizing
MAX_LEVERAGE = 1  # define the maximum leverage for the strategy / position sizing
CHAT_ID = os.getenv('CHAT_ID')  # Telegram chat ID (set as environment variable)


MAX_RETRIES = 3  # number of retries when fetching data from the repository
repo.RATE_LIMIT = False  # disable rate limit for the repository


# ============================ Data Display & Processing =============================
def display_results(df: pd.DataFrame) -> None:
    # calculate the statistics (like sharpe ratio, sortino ratio, ...)
    # for the strategy and HODL
    stats = st.calculate_statistics(df["b.value"].to_numpy())
    stats = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in stats.items()}

    stats_hodl = st.calculate_statistics(df["hodl.value"].to_numpy())
    stats_hodl = {
        k: f"{v:.2f}" if isinstance(v, float) else v for k, v in stats_hodl.items()
        }

    # preprocess the dataframe for display on std out
    incl_cols = [
        "open", "high", "low", "close", "volume",
        "position", "leverage", "buy",
        "buy_size", "buy_at", "sell", "sell_size", "sell_at",
        "b.base", "b.quote", "b.value", "b.drawdown.max",
        "hodl.value", "hodl.drawdown.max",
        ]

    df = df[incl_cols].copy()

    # Apply rounding and formatting to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    df.replace(np.nan, "", inplace=True)
    df.replace(False, ".", inplace=True)
    df.replace(0, "", inplace=True)

    df = df.astype(str)
    df.loc[df["buy_size"] != "", "buy"] = "•"
    df.loc[df["sell_size"] != "", "sell"] = "•"
    df.loc[df["position"] == "1.0000", "position"] = "LONG"
    df.loc[df["position"] == "-1.0000", "position"] = "SHORT"

    # display the last 50 rows of the dataframe & the statistics
    logger.info("------------------------ Last 50 rows -------------------------")
    print(df.tail(50))

    logger.info("--------------------------- Statistics ---------------------------")
    logger.info("stats strategy: %s" % stats)
    logger.info("stats hodl    : %s" % stats_hodl)


def process_data(ohlcv_data):
    logger.debug(ohlcv_data)

    ohlcv_data = bt.run(
        data=ohlcv_data.to_dict(),
        strategy=strategy,
        risk_level=RISK_LEVEL,
        max_leverage=MAX_LEVERAGE,
        initial_capital=10_000
    )

    # build a dataframe from the results & set index to be the datetime
    df = pd.DataFrame.from_dict(ohlcv_data)
    df.index = pd.to_datetime(df["open time"], unit="ms")

    # add drawdown information to dataframe
    df = rs.calculate_stats(df, initial_capital=10_000)

    if LOG_LEVEL == logging.DEBUG:
        display_results(df)

    return df


# ================================ Async Functions ===================================
async def fetch_ohlcv_data(request: dict, queue: Queue, stop_event: Event):
    logger.debug("Async Event Loop started ...")

    try:
        while not stop_event.is_set():
            try:
                ohlcv_data = await repo.process_request(request)
                if not ohlcv_data:
                    logger.warning("No OHLCV data received. Retrying...")
                queue.put(ohlcv_data)
            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {e}")
                await asyncio.sleep(5)  # Wait before retrying
            else:
                # we got our data and can stop the loop
                stop_event.set()
                logger.debug("stop_event set.")
    except Exception as e:
        logger.error(f"Error in Async Loop: {e}")

    await repo.exchange_factory(None)
    logger.debug("Async Event Loop stopped.")


async def send_signal(df: pd.DataFrame) -> None:
    """Sends a trading signal to Telegram, based on the results DataFrame."""
    await ts.send_message(
        chat_id=CHAT_ID,
        msg=await ts.create_signal(
            position=Positions(df=df, symbol=strategy.symbol).current().get_signal()
            )
        )


async def send_performance_chart(df: pd.DataFrame) -> None:
    await ts.send_message(
        chat_id=CHAT_ID,
        msg=f"Gregorovich performance since {df.index.min().strftime('%Y-%m-%d')}",
        image=Chart(df, title=strategy.name).get_image_bytes()
    )


async def notify_telegram(df: pd.DataFrame) -> None:
    await send_signal(df)
    await send_performance_chart(df)


# ================================= Sync Functions ===================================
# Function to Run the Async Event Loop in a Separate Thread
def start_async_loop(queue, stop_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # build ohlcv request
    ohlcv_request = {
        "exchange": "binance",
        "symbol": strategy.symbol,
        "interval": strategy.interval,
    }

    try:
        loop.run_until_complete(fetch_ohlcv_data(ohlcv_request, queue, stop_event))
    finally:
        loop.run_until_complete(repo.exchange_factory(None))
        loop.close()


# Main Function to Integrate Everything
def main():
    # Create a Queue for Communication
    ohlcv_queue = Queue()

    # Start the Async Event Loop in a Separate Thread
    #
    # NOTE: We need to run this in a thread because the ohlcv_repository
    #       has been implemented with asnyc functions
    logger.debug("Starting Async Event Loop in thread ...")

    # Event to Signal the Async Thread to Stop
    stop_event = Event()

    async_thread = threading.Thread(
        target=start_async_loop,
        args=(ohlcv_queue, stop_event),
        daemon=True
        )
    async_thread.start()

    # Main Loop: Process OHLCV Data from the Queue
    counter = 0
    try:
        while counter < MAX_RETRIES:
            try:
                ohlcv_data = ohlcv_queue.get(timeout=10)

                if not isinstance(ohlcv_data, repo.Response):
                    raise ValueError("Invalid data received")

                if not ohlcv_data.data:
                    raise ValueError("No OHLCV data received")

            except Empty:
                print("No new data received. Waiting...")
            except ValueError as e:
                logger.error("Error processing data: %s", e)
            else:
                stop_event.set()
                df = process_data(ohlcv_data)
                asyncio.run(notify_telegram(df))
                ohlcv_queue.task_done()
                break
            finally:
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
