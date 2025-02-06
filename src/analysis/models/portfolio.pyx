# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
import numpy as np
from functools import reduce
from libc.stdlib cimport malloc, free

cdef double fee_rate = 0.001
cdef double slippage_rate = 0.001


cdef extern from *:
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


""""
cdef class StopOrder:
    cdef:
        BacktestData* data
        int asset_index

# Example of a concrete StopOrder implementation
cdef class TrailingStopOrder(StopOrder):
    cdef:
        double trail_percent

    def __cinit__(self, BacktestData* data, int asset_index, double trail_percent):
        self.data = data
        self.asset_index = asset_index
        self.trail_percent = trail_percent

    cdef double apply(self, int period):
        # Implementation using self.data and self.asset_index
        pass
"""


# ............................... Trade Action classes .................................
cdef class ActionInterface:
    cdef public ActionData data


cdef class Buy(ActionInterface):

    def __cinit__(self, np.int64_t timestamp, double amount, double price):
        self.data.timestamp = timestamp
        self.data.price = price
        self._calculate(amount)

    def __repr__(self):
        return (
            f"Buy(timestamp={self.timestamp}, amount={self.data.qty}, "
            f"price={self.data.price})"
        )

    def __add__(self, Buy other):
        new_qty = self.qty + other.qty
        new_quote_qty = self.quote_qty + other.quote_qty

        # Calculate volume-weighted average price
        new_price = (
            self.data.price * self.data.qty + other.data.price * other.data.qty
            ) / new_qty

        return Buy(self.timestamp, new_quote_qty, new_price)

    cdef void _calculate(self, double amount):
        self.data.quote_qty = amount
        self.data.fee = amount * fee_rate
        self.data.slippage = amount * slippage_rate
        cdef double net_amount = amount - self.data.fee - self.data.slippage
        self.data.qty = net_amount / self.data.price

    @property
    def type(self) -> str:
        return "BUY"

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


cdef class Sell(ActionInterface):

    def __cinit__(self, np.int64_t timestamp, double amount, double price):
        self.data.timestamp = timestamp
        self.data.price = price
        self._calculate(amount)

    def __repr__(self):
        return (
            f"Sell(timestamp={self.timestamp}, amount={self.data.qty}, "
            f"price={self.data.price})"
        )

    def __add__(self, Sell other):
        new_qty = self.qty + other.qty
        new_quote_qty = self.quote_qty + other.quote_qty

        # Calculate volume-weighted average price
        new_price = (
            self.data.price * self.data.qty + other.data.price * other.data.qty
            ) / new_qty

        return Sell(self.timestamp, new_qty, new_price)

    cdef void _calculate(self, double amount):
        self.data.qty = amount
        cdef double gross_quote = amount * self.data.price
        self.data.fee = gross_quote * fee_rate
        self.data.slippage = gross_quote * slippage_rate
        self.data.quote_qty = gross_quote - self.data.fee - self.data.slippage

    @property
    def type(self) -> str:
        return "BUY"

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


# .................................. Position class ....................................
cdef class Position:
    """A class to represent a trading position."""

    cdef:
        list buys
        list sells
        readonly str symbol
        readonly double current_qty
        readonly double average_entry_price
        readonly double last_price
        readonly double realized_pnl


    def __cinit__(self, str symbol):
        self.buys = []
        self.sells = []

        self.symbol = symbol
        self.current_qty = 0.0
        self.average_entry_price = 0.0
        self.last_price = 0.0
        self.realized_pnl = 0.0

    def __repr__(self):
        return (
            f"Position(symbol={self.symbol}, current_qty={self.current_qty}, "
            f"average_entry_price={self.average_entry_price:.2f}, "
            f"realized_pnl={self.realized_pnl:.2f})"
        )

    @property
    def average_entry(self) -> ActionInterface:
        if not self.buys:
            return None
        return reduce(lambda x, y: x + y, self.buys)

    @property
    def average_exit(self) -> ActionInterface:
        if not self.sells:
            return None
        return reduce(lambda x, y: x + y, self.sells)

    def get_actions(self):
        return sorted(self.buys + self.sells, key=lambda x: x.timestamp)

    cpdef void add_action(self, ActionInterface action):
        """Adds a new buy or sell action to the position.
        
        Python wrapper method which allows to define the actual 
        _add_action method for pure cythonic access. This method
        is mainly intended to be used for tests.
        """
        self._add_action(action)

    cpdef void close(self, np.int64_t timestamp, double price):
        """Adds a new buy or sell action to the position.
        
        Python wrapper method which allows to define the actual 
        _add_action method for pure cythonic access.
        """
        self._close(timestamp, price)

    cdef void _add_action(self, ActionInterface action):
        if isinstance(action, Buy):
            self.buys.append(action)
        elif isinstance(action, Sell):
            self.sells.append(action)
        self._update_position(action)

    cdef void _close(self, np.int64_t timestamp, double price):
        if self.current_qty > 0:
            self.add_action(Sell(timestamp, self.current_qty, price))
        if self.current_qty < 0:
            self.add_action(Buy(timestamp, self.current_qty, price))

    cdef void _update_position(self, ActionInterface action) except *:
        cdef double old_qty = self.current_qty
        cdef double old_value = old_qty * self.average_entry_price

        if isinstance(action, Buy):
            new_qty = self.current_qty + action.data.qty

            if old_qty < 0 and new_qty > 0:
                raise ValueError(
                    f"Buying {action.data.qty} would change this short position to a "
                    f"long position. Use .close() method before opening a new long "
                    f"position, or buy a max amount of {abs(self.current_qty)}."
                )

            self.current_qty = new_qty
            new_value = old_value + action.qty * action.data.price
            self.average_entry_price = new_value / self.current_qty
        
        elif isinstance(action, Sell):
            new_qty = self.current_qty - action.data.qty
        
            if old_qty > 0 and new_qty < 0:
                raise ValueError(
                    f"Selling {action.data.qty} would change this long position to a "
                    f"short position. Use .close() method before opening a new short "
                    f"position, or sell a max amount of {self.current_qty}."
                )

            self.current_qty = new_qty

        # calculate the PNL if the position is closed now
        if self.current_qty == 0:
            self.realized_pnl = abs(
                self.average_entry_price - self.average_exit_price
            ) * old_qty


# ................................. Portfolio class ....................................
cdef class Portfolio:

    cdef BacktestData* data

    def __cinit__(self, BacktestData* data):
        self.data = data

    # cdef void process_period(
    #     self, np.int64_ timestamp, np.ndarray open_price, double close_price
    # ):
    #     pass


# ................................ The BackTestEngine ...................................
"""
cdef class BacktestEngine:
    cdef:
        BacktestData* data
        Portfolio portfolio
        list positions
        int num_periods
        int num_assets

    def __cinit__(self, int num_periods, int num_assets):
        self.num_periods = num_periods
        self.num_assets = num_assets
        
        # Allocate memory for BacktestData
        self.data = <BacktestData*>malloc(sizeof(BacktestData))
        if not self.data:
            raise MemoryError()

        # Initialize arrays
        self.data.market_data.open = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.high = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.low = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.close = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.volume = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.atr = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.market_data.volatility = np.zeros((num_periods, num_assets), dtype=np.float64)

        self.data.leverage = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.cash_balance = np.zeros((num_periods, 1), dtype=np.float64)
        self.data.equity = np.zeros((num_periods, 1), dtype=np.float64)
        self.data.asset_balances = np.zeros((num_periods, num_assets, 1), dtype=np.float64)
        self.data.effective_leverage_per_asset = np.zeros((num_periods, num_assets), dtype=np.float64)
        self.data.effective_leverage_global = np.zeros(num_periods, dtype=np.float64)

        # Initialize Portfolio and Positions
        self.portfolio = Portfolio(self.data)
        self.positions = [Position(self.data, i) for i in range(num_assets)]

    def __dealloc__(self):
        if self.data:
            free(self.data)

    def run_backtest(self):
        # Backtest logic here
        pass
"""

# Usage
# cdef BacktestEngine engine = BacktestEngine(1000, 5)  # 1000 periods, 5 assets
# engine.run_backtest()