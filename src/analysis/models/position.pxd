# cython: language_level=3
# distutils: language = c++
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np


cdef struct TradeData:
    int type
    long long timestamp
    double price
    double qty
    double gross_quote_qty
    double net_quote_qty
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


cdef struct StopOrder:
    int type
    int trailing 
    double price
    double qty
    int exit_after


cdef struct PositionData:
    int idx
    int type
    int is_active
    int duration
    double avg_entry_price
    double pnl
    vector[TradeData] trades
    vector[StopOrder] stop_orders


cdef double get_fee(double qty, double fee_rate)
cdef double get_slippage(double qty, double slippage_rate)

cdef TradeData build_buy_trade(long long timestamp, double quote_qty, double price)
cdef TradeData build_sell_trade(long long timestamp, double base_qty, double price)

cdef void add_buy(PositionData* pos, long long timestamp, double quote_qty, double price)
cdef void add_sell(PositionData* pos, long long timestamp, double base_qty, double price)

cdef PositionData build_long_position(int index, long long timestamp, double quote_qty, double price)
cdef PositionData build_short_position(int index, long long timestamp, double base_qty, double price)

cpdef void run_func_bench(int iterations)

# ............................... Trade Action classes .................................
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
