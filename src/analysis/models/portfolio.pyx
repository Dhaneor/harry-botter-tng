# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
import numpy as np

from analysis.models.market_data_store cimport MarketDataStore, MarketState
from analysis.models.position cimport Position


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




# ................................. Portfolio class ....................................
cdef class Portfolio:

    cdef:
        list[double] shape
        list[double] a_weights
        list[double] s_weights
        public np.ndarray equity
        public np.ndarray base_qty
        public np.ndarray quote_qty
        public np.ndarray leverage
        public np.ndarray quote_qty_global
        public np.ndarray equity_global
        public np.ndarray leverage_global

    def __cinit__(
        self, 
        list[double] shape, 
        list[double] a_weights, 
        list[double] s_weights
    ):
        self.shape = shape

        self.a_weights = a_weights
        self.s_weights = s_weights

        # arrays for tracking assets per period and strategy
        self.equity = np.zeros(shape, dtype=np.float64)
        self.base_qty = np.zeros(shape, dtype=np.float64)
        self.quote_qty = np.zeros(shape, dtype=np.float64)
        self.leverage = np.zeros(shape, dtype=np.float64)

        # arrays for tracking the portfolio
        self.quote_qty_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.equity_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.leverage_global = np.zeros(self.shape[:1], dtype=np.float64)

    cdef void update(self, MarketState state, double[:] signals):
        pass


# ................................ The BackTestEngine ...................................
cdef class BacktestEngine:
    cdef:
        MarketDataStore market_data
        np.ndarray signals
        Portfolio portfolio

    def __cinit__(self, MarketDataStore market_data, np.ndarray signals):
        self.market_data = market_data
        self.signals = signals
                
        # Initialize Portfolio
        self.portfolios = [Portfolio(self.data) for _ in range(signals.shape[2])]

    def run_backtest(self):
        periods, _, strategies = self.shape

        for p in range(periods):
            state = self.market_data.get_state(p)
            for s in range(strategies):
                self.portfolio[s].update(
                    state,
                    self.signals[p, :, s],
                    self.leverage[p, :]
                )