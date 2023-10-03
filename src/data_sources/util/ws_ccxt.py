#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 13:58:23 2023

@author_ dhaneor
"""
import asyncio
# import ccxt.async_support as ccxt
import ccxt.pro as ccxt
from ccxt.base.errors import BadSymbol
import logging
from pprint import pprint

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s.%(funcName)s.%(lineno)d  - [%(levelname)s]: %(message)s"
)
ch.setFormatter(formatter)

logger.addHandler(ch)

exchange = ccxt.binance()


async def main(exchange):

    symbols = ["BTC/USDT"]

    if exchange.has['watchTickers']:

        counter = 0

        while True:
            try:

                if counter == 3:
                    new_symbol = "ETHA/USDT"
                    symbols.append(new_symbol)

                if counter == 5:
                    symbols.append("ETH/USDT")

                tickers = await exchange.watch_tickers(symbols=symbols, params={})

                if tickers:
                    logger.info("---------------------------------------------------")
                    logger.info("[%s] %s", counter, tickers)

            except BadSymbol as e:
                logger.error("[%s] %s", counter, e)
                symbols.remove(new_symbol)

            except asyncio.CancelledError:
                logger.info('Cancelled...')
                break

            except Exception as e:
                logger.exception(e)
                break

            finally:
                counter += 1

    else:
        logger.info("watchTickers not supported")
        pprint(exchange.has)

    await exchange.close()
    print("exchange closed: OK")

if __name__ == '__main__':
    try:
        asyncio.run(main(exchange))
    except KeyboardInterrupt:
        print('Interrupted...')
