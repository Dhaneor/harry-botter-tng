# cython: language_level=3
# distutils: language = c++

cimport numpy as np
import numpy as np
from libcpp.vector cimport vector


cdef struct TradeData:
    int type
    long long timestamp
    double price
    double qty
    double quote_qty
    double fee
    double slippage


cdef struct ActionData:
    int type
    np.int64_t timestamp
    double price
    double qty
    double quote_qty
    double fee
    double slippage


# ............................... Trade Action classes .................................
cdef class Trade:
    cdef TradeData data


cdef class ActionInterface:
    cdef public ActionData data


cdef class Buy(ActionInterface):
    cdef void _calculate(self, double amount)


cdef class Sell(ActionInterface):
    cdef void _calculate(self, double amount)


# .................................. Position class ....................................
cdef class Position:
    cdef:
        readonly str symbol
        cdef list[ActionInterface] buys
        cdef list[ActionInterface] sells
        readonly int type
        readonly int is_active
        readonly double current_qty
        readonly double average_entry_price
        readonly double last_price
        readonly double realized_pnl

    cpdef void add_action(self, ActionInterface action)
    cpdef void close(self, np.int64_t timestamp, double price)
    
    cdef void _open(self, np.int64_t timestamp, double size, double price)
    cdef void _close(self, np.int64_t timestamp, double price)
    cdef void _update(self, ActionInterface action) except *
    cdef double _get_average_buy_price(self)
    cdef double _get_average_sell_price(self)