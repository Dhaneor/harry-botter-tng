# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
import numpy as np
from abc import ABC, abstractmethod
from libcpp.vector cimport vector

cdef double fee_rate = 0.001
cdef double slippage_rate = 0.001

cdef struct ActionData:
    np.int64_t timestamp
    double price
    double qty
    double quote_qty
    double fee
    double slippage

cdef class Action:
    cdef ActionData data

    def __init__(self, np.int64_t timestamp, double amount, double price):
        self.data.timestamp = timestamp
        self.data.price = price

    def __add__(self, Action other):
        if type(self) != type(other):
            raise TypeError("Can only add actions of the same type")

        new_qty = self.qty + other.qty
        new_fee = self.fee + other.fee
        new_slippage = self.qty + other.qty
        new_quote_qty = self.quote_qty + other.quote_qty

        # Calculate volume-weighted average price
        new_price = (
            self.data.price * self.data.qty + other.data.price * other.data.qty
            ) / new_qty

        if self.type == "BUY":
             return Buy(self.timestamp, new_quote_qty, new_price)

        elif self.type == "SELL":
            return Sell(self.timestamp, new_qty, new_price)

    @property
    def timestamp(self):
        return self.data.timestamp

    @property
    def qty(self):
        return self.data.qty

    @property
    def price(self):
        return self.data.price

    @property
    def quote_qty(self):
        return self.data.quote_qty

    @property
    def fee(self):
        return self.data.fee

    @fee.setter
    def fee(self, value):
        self.data.fee = value

    @property
    def slippage(self):
        return self.data.slippage

    @slippage.setter
    def slippage(self, value):
        self.data.slippage = value


cdef class Buy(Action):
    def __init__(self, np.int64_t timestamp, double amount, double price):
        super().__init__(timestamp, amount, price)
        self._calculate(amount)

    def __repr__(self):
        return (
            f"Buy(timestamp={self.timestamp}, amount={self.data.qty}, "
            f"price={self.data.price})"
        )

    cdef void _calculate(self, double amount):
        self.data.quote_qty = amount
        self.data.fee = amount * fee_rate
        self.data.slippage = amount * slippage_rate
        cdef double net_amount = amount - self.data.fee - self.data.slippage
        self.data.qty = net_amount / self.data.price

    @property
    def type(self) -> str:
        return "BUY"

cdef class Sell(Action):
    def __init__(self, np.int64_t timestamp, double amount, double price):
        super().__init__(timestamp, amount, price)
        self._calculate(amount)

    def __repr__(self):
        return (
            f"Sell(timestamp={self.timestamp}, amount={self.data.qty}, "
            f"price={self.data.price})"
        )

    cdef void _calculate(self, double amount):
        self.data.qty = amount
        cdef double gross_quote = amount * self.data.price
        self.data.fee = gross_quote * fee_rate
        self.data.slippage = gross_quote * slippage_rate
        self.data.quote_qty = gross_quote - self.data.fee - self.data.slippage

    @property
    def type(self) -> str:
        return "BUY"


# .................................. Position class ....................................
cdef class Position:
    """A class to represent a trading position."""

    cdef:
        list buys
        list sells
        readonly str symbol
        readonly double current_qty
        readonly double average_entry_price
        readonly double realized_pnl

    def __cinit__(self, str symbol):
        self.buys = []
        self.sells = []

        self.symbol = symbol
        self.current_qty = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl = 0.0

    def __repr__(self):
        return (
            f"Position(symbol={self.symbol}, current_qty={self.current_qty}, "
            f"average_entry_price={self.average_entry_price:.2f}, "
            f"realized_pnl={self.realized_pnl:.2f})"
        )

    cpdef void add_action(self, Action action):
        if isinstance(action, Buy):
            self.buys.append(action)
        elif isinstance(action, Sell):
            self.sells.append(action)
        self._update_position(action)

    cdef void _update_position(self, Action action):
        print(action)
        print(self.current_qty)
        cdef double old_qty = self.current_qty
        cdef double old_value = old_qty * self.average_entry_price

        if isinstance(action, Buy):
            self.current_qty += action.data.qty
            new_value = old_value + action.qty * action.data.price
            self.average_entry_price = new_value / self.current_qty
        elif isinstance(action, Sell):
            if self.current_qty > 0:
                # Closing long position
                self.realized_pnl += (action.price - self.average_entry_price) * min(self.current_qty, action.qty)
            else:
                # Closing short position
                self.realized_pnl += (self.average_entry_price - action.price) * min(-self.current_qty, action.qty)
            
            self.current_qty -= action.qty
            if self.current_qty == 0:
                self.average_entry_price = 0.0
            elif self.current_qty * old_qty < 0:  # Position direction changed
                self.average_entry_price = action.price

    def get_actions(self):
        return self.buys + self.sells

    def get_average_entry(self):
        if not self.buys:
            return None
        return sum(self.buys, start=self.buys[0])

    def get_average_exit(self):
        if not self.sells:
            return None
        return sum(self.sells, start=self.sells[0])
