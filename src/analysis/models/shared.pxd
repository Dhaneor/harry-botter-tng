# cython: language_level=3
# distutils: language = c++

cimport numpy as np
import numpy as np
from libcpp.vector cimport vector

ctypedef struct MarketData:
    double** open
    double** high
    double** low
    double** close
    double** volume
    double** atr
    double** volatility

cdef struct ActionData:
    int type
    np.int64_t timestamp
    double price
    double qty
    double quote_qty
    double fee
    double slippage