# cython: language_level=3
# distutils: language = c++

cimport numpy as np
import numpy as np

ctypedef struct MarketData:
    double** open
    double** high
    double** low
    double** close
    double** volume
    double** atr
    double** volatility


ctypedef struct BacktestData:
    MarketData market_data
    double** leverage
    double** cash_balance
    double** equity
    double*** asset_balances
    double** effective_leverage_per_asset
    double* effective_leverage_global


cdef struct ActionData:
    np.int64_t timestamp
    double price
    double qty
    double quote_qty
    double fee
    double slippage