# cython: language_level=3
# distutils: language = c++
# cython: numpy_api=2

cimport numpy as np
import logging
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

from analysis.models.market_data_store cimport MarketDataStore, MarketState
from analysis.models.position cimport (
    TradeData,
    PositionData,
    StopOrder,
    build_buy_trade,
    build_sell_trade,
    add_buy,
    add_sell,
    build_long_position,
    build_short_position,
    close_position,
)

logger = logging.get_logger(f"main.{__name__}")

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


cdef PositionData* get_current_position(self, Account acc, int m, int s):
    if acc.positions.find(m) == acc.positions.end():
        self.positions[m] = unordered_map[int, vector[PositionData]]()
    
    if acc.positions[m].find(s) == acc.positions[m].end():
        return NULL  # Return NULL if no position exists
    
    if acc.positions[m][s].empty():
        return NULL  # Return NULL if the vector is empty
    
    return &acc.positions[m][s].back()  # Return pointer to the last position


cdef void add_position(self, int m, int s, PositionData position):
    if self.positions[m].find(s) == self.positions[m].end():
        self.positions[m][s] = vector[PositionData]()
    self.positions[m][s].push_back(position)
    


# ................................. Portfolio class ....................................
cdef class Portfolio:

    cdef:
        list shape
        list a_weights
        list s_weights
        unordered_map[int, unordered_map[int, vector[PositionData]]] positions
        public np.ndarray equity
        public np.ndarray base_qty
        public np.ndarray quote_qty
        public np.ndarray leverage
        public np.ndarray quote_qty_global
        public np.ndarray equity_global
        public np.ndarray leverage_global

    def __cinit__(self, list shape, list a_weights, list s_weights):
        self.shape = shape
        self.a_weights = a_weights
        self.s_weights = s_weights
        self.positions = {}

        # Initialize arrays
        self.equity = np.zeros(shape, dtype=np.float64)
        self.base_qty = np.zeros(shape, dtype=np.float64)
        self.quote_qty = np.zeros(shape, dtype=np.float64)
        self.leverage = np.zeros(shape, dtype=np.float64)

        self.quote_qty_global = np.zeros(shape[:1], dtype=np.float64)
        self.equity_global = np.zeros(shape[:1], dtype=np.float64)
        self.leverage_global = np.zeros(shape[:1], dtype=np.float64)

    cdef void update(self, MarketState state, double[:] signals):
        cdef int s, m
        cdef PositionData* current_position

        for s in range(signals.shape[0]):
            for m in range(len(self.a_weights)):
                current_position = self._get_current_position(m, s)
                if current_position == NULL:
                    # No position exists, create a new one if needed
                    if signals[s] != 0:  # Or whatever condition you use to decide to open a position
                        new_position = PositionData()  # Initialize with default or specific values
                        self._add_position(m, s, new_position)
                        current_position = &self.positions[m][s].back()
                
                if current_position != NULL:
                    # Update the position
                    current_position.is_active = (signals[s] != 0)
                    # current_position.size = ...
                    # Add your update logic here
                    pass

    cdef PositionData* _get_current_position(self, int m, int s):
        if self.positions.find(m) == self.positions.end():
            self.positions[m] = unordered_map[int, vector[PositionData]]()
        
        if self.positions[m].find(s) == self.positions[m].end():
            return NULL  # Return NULL if no position exists
        
        if self.positions[m][s].empty():
            return NULL  # Return NULL if the vector is empty
        
        return &self.positions[m][s].back()  # Return pointer to the last position


    cdef void _add_position(self, int m, int s, PositionData position):
        if self.positions[m].find(s) == self.positions[m].end():
            self.positions[m][s] = vector[PositionData]()
        self.positions[m][s].push_back(position)
