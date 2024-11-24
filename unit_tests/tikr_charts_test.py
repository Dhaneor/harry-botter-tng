#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21 12:08:20 2024

@author dhaneor
"""
import logging
import threading
import asyncio
import numpy as np
import os
import pandas as pd
import sys
import time
from queue import Queue, Empty
from threading import Event
from typing import Any

# ------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------------

from src.rawi import ohlcv_repository as repo  # noqa: E402
from src.analysis import strategy_builder as sb  # noqa: E402
from src.analysis import strategy_backtest as bt  # noqa: E402
from src.analysis.backtest import statistics as st  # noqa: E402
from src.analysis import telegram_signal as ts  # noqa: E402
from src.backtest import result_stats as rs  # noqa: E402
from src.plotting.tikr_charts import TikrChart as Chart  # noqa: E402
from tikr_mvp_strategy import mvp_strategy  # noqa: E402

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

start_time = time.time()

# ================================ Configuration =====================================
# instantiate strategy: the mvp_strategy is not available in the
# Github repository, you can implement your own strategy
strategy = sb.build_strategy(mvp_strategy)
strategy.name = "Safe HODL Strategy by Gregorovich"

RISK_LEVEL = 0  # define the risk level for the strategy / position sizing
MAX_LEVERAGE = 1  # define the maximum leverage for the strategy / position sizing
CHAT_ID = os.getenv('CHAT_ID')  # Telegram chat ID (set as environment variable)

TIMEOUT = 60  # time in seconds to wait for the data to be available in the repository
RETRY_AFTER_SECS = 5  # time between retries in seconds
repo.RATE_LIMIT = False  # disable rate limit for the repository
repo.LOG_STATUS = False  # enable logging of server status and server time

DISPLAY_DF_ROWS = 10  # number of rows to display in the dataframe


# ============================ Data Display & Processing =============================
def format_dataframe(df: pd.DataFrame) -> None:
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

    return df


def calculate_stats(value_series: pd.Series) -> dict[str, Any]:
    return {
        k: f"{v:.2f}" if isinstance(v, float) else v
        for k, v in st.calculate_statistics(value_series.to_numpy()).items()
        }


def generate_comparison_table(df: pd.DataFrame) -> str:
    strategy_stats = calculate_stats(df["b.value"])
    hodl_stats = calculate_stats(df["hodl.value"])

    metrics = [
        ("Total Return %", "profit"),
        ("Max Drawdown %", "max_drawdown"),
        ("Sharpe Ratio", "sharpe_ratio"),
        # ("Sortino Ratio", "sortino_ratio"),
        # ("Kalmar Ratio", "kalmar_ratio"),
        # ("Ann. Volatility %", "annualized_volatility")
    ]

    table = "```\n"  # Start of monospace block
    table += "Performance Comparison:\n"
    table += "━━━━━━━━━━━━━━━━━━━━━━━━\n"

    for metric_name, metric_key in metrics:
        strategy_value = strategy_stats[metric_key]
        hodl_value = hodl_stats[metric_key]
        table += f"{metric_name}:\n"
        table += f"  Strategy: {strategy_value}\n"
        table += f"  HODL:     {hodl_value}\n"
        table += "――――――――――――――――――――――――\n"

    table = table.rstrip("―\n")  # Remove last separator
    table += "\n```"  # End of monospace block

    return table


def display_stats(df: pd.DataFrame) -> None:
    logger.info("--------------------------- Statistics ---------------------------")
    logger.info("stats strategy: %s" % calculate_stats(df["b.value"]))
    logger.info("stats hodl    : %s" % calculate_stats(df["hodl.value"]))


def display_results(df: pd.DataFrame) -> None:
    # display the dataframe & statistics
    logger.info("------------------------ Results ---------------------------")
    # display the last 50 rows of the dataframe & the statistics
    print(format_dataframe(df).tail(DISPLAY_DF_ROWS))

    display_stats(df)


def run_backtest(response: repo.Response):
    logger.debug(response)

    backtest_result = bt.run(
        data=response.to_dict(),
        strategy=strategy,
        risk_level=RISK_LEVEL,
        max_leverage=MAX_LEVERAGE,
        initial_capital=10_000
    )

    # build a dataframe from the results & set index to be the datetime
    df = pd.DataFrame.from_dict(backtest_result)
    df.index = pd.to_datetime(df["open time"], unit="ms")

    # add drawdown & other additional information to dataframe
    df = rs.calculate_stats(df, initial_capital=10_000)

    if LOG_LEVEL == logging.DEBUG:
        display_results(df)

    return df


# ================================ Async Functions ===================================
async def fetch_ohlcv_data(request: dict, queue: Queue, stop_event: Event):
    logger.debug("Async Event Loop started ...")

    while not stop_event.is_set():
        try:
            ohlcv_data = await repo.process_request(request)
            if not ohlcv_data:
                logger.warning("No OHLCV data received. Retrying...")
                raise Empty("empty response from OHLCV repository")

        except Empty as e:
            logger.error(f"Fetching OHLCV failed: {e}")
            await asyncio.sleep(RETRY_AFTER_SECS)  # Wait before retrying

        except TimeoutError as e:
            logger.error(f"Fetching OHLCV timed out: {e}")
            await asyncio.sleep(RETRY_AFTER_SECS)  # Wait before retrying

        except Exception as e:
            logger.error(
                "Unknown error while waiting for OHLCV data: %s", e, exc_info=True
                )
            stop_event.set()  # no need to retry in this case

        else:
            queue.put(ohlcv_data)
            stop_event.set()

    logger.debug("sending exchange close request ...")
    await repo.exchange_factory(None)  # this signals the exchange to close


def draw_chart(df: pd.DataFrame, style='night'):
    chart = Chart(df, style=style, title=f"{strategy.symbol} {strategy.interval}")
    chart.draw()
    logger.info("Chart ready: OK")


async def send_chart(df: pd.DataFrame) -> None:
    comparison_table = generate_comparison_table(df)

    msg = (
        f"Gregorovich performance since {df.index.min().strftime('%Y-%m-%d')}\n\n"
        f"{comparison_table}"
    )

    await ts.send_message(
        chat_id=CHAT_ID,
        msg=msg,
        image=Chart(df, style='day', title=strategy.name).get_image_bytes()
    )


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

    try:
        # we just wait here until we get something
        response = ohlcv_queue.get(timeout=TIMEOUT)

        # if any problem occured, the response from the ohlcv_repository
        # will be marked as unsuccessful, and additional imformation
        # about the cause if the issue should be included.
        if not response.success:
            raise ValueError("Response marked as unsuccessful: %s" % response)

    except ValueError as e:
        logger.error(e)
    except (TimeoutError, Empty) as e:
        logger.error("Timeout while waiting for OHLCV data: %s" % e)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Exiting...")
    except Exception as e:
        logger.error("An unexpected error occured: %s", e, exc_info=True)

    else:
        df = run_backtest(response)
        # asyncio.run(send_chart(df))
        draw_chart(df)
    finally:
        ohlcv_queue.task_done()
        stop_event.set()
        async_thread.join(timeout=RETRY_AFTER_SECS + 3)

    logger.info("exection time: %s seconds" % f"{(time.time() - start_time):.2f}")
    logger.info("shutdown complete: OK")


# Run the Main Function
if __name__ == "__main__":
    main()
