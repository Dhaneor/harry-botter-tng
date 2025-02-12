# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
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

cdef struct Account:
    unordered_map[int, unordered_map[int, vector[PositionData]]] positions


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


# ................................ The BackTestEngine ...................................
cdef class BacktestEngine:
    cdef:
        MarketDataStore market_data
        np.ndarray signals
        Portfolio portfolio
        tuple[int, int, int] shape
        public np.ndarray equity
        public np.ndarray base_qty
        public np.ndarray quote_qty
        public np.ndarray leverage
        public np.ndarray quote_qty_global
        public np.ndarray equity_global
        public np.ndarray leverage_global

    def __cinit__(self, MarketDataStore market_data, np.ndarray signals):
        self.market_data = market_data
        self.signals = signals
                
        # Initialize Portfolio
        self.portfolios = [Portfolio(self.data) for _ in range(signals.shape[2])]

        self.shape = (signals.shape[0], signals.shape[1], signals.shape[2],)

        # Initialize arrays
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.base_qty = np.zeros(self.shape, dtype=np.float64)
        self.quote_qty = np.zeros(self.shape, dtype=np.float64)
        self.leverage = np.zeros(self.shape, dtype=np.float64)

        self.quote_qty_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.equity_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.leverage_global = np.zeros(self.shape[:1], dtype=np.float64)

    def run_backtest(self):
        cdef int signal, prev_signal, position
        periods, markets, strategies = self.shape

        for s in range(strategies):
            for p in range(periods):
                state = self.market_data.get_state(p)
                for m in range(markets):
                    signal = self.signals[p, m, s] if p > 0 else 0
                    prev_signal = self.signals[p-2, m, s] if p > 1 else 0

                    open_long = signal > 0 and prev_signal <= 0
                    close_long = signal <= 0 and prev_signal > 0
                    open_short = signal < 0 and prev_signal >= 0
                    close_short = signal >= 0 and prev_signal < 0

                    if self.base_qty[p-1, m, s] > 0:
                        position = 1
                    elif self.base_qty[p-1, m, s] < 0:
                        position = -1
                    else:
                        position = 0

                    if position == 1:
                        if close_long or open_short:
                            self._close_position(p, m, s)
                        else:
                            self._update_position(p, m, s)

                    if position == -1:
                        if close_short or open_long:
                            self._close_position(p, m, s)
                        else:
                            self._update_position(p, m, s)

                    if position == 0:
                        if open_long:
                            self._open_position(p, m, s, 1)
                        elif open_short:
                            self._open_position(p, m, s, -1)
                    
                    self.equity ^= \
                        self.market_data.close[p, m] \
                        * self.positions[p, m, s]["qty"] \
                        + self.positions[p, m, s]["quote_qty"] 

    # ............................ PROCESSINNG POSITIONS ...............................
    def _open_position(self, p: int, m: int, s: int, type: int):
        price = self.market_data.open_[p, m]
        exposure_quote, _ = self._calculate_change_exposure(p, m, s, price)

        self.positions[p, m, s]["position"] = type
        self.positions[p, m, s]["entry_price"] = price
        self.positions[p, m, s]["duration"] = 1

        if exposure_quote > 0:
            self._process_buy(p, m, s, exposure_quote, price)
        elif exposure_quote < 0:
            self._process_sell(p, m, s, -exposure_quote, price)

    def _close_position(self, p, m, s):
        price = self.market_data.close[p, m]
        portfolio = self.positions[p, m, s]
        position = self.positions[p, m, s]["position"]

        if position == 0:
            return
        
        change_quote = portfolio["qty"] * price
        fee, slippage = self._calculate_fee_and_slippage(change_quote)

        self.positions[p, m, s]["quote_qty"] += change_quote - fee - slippage
        
        # Update portfolio
        if position == 1:
            self.positions[p, m, s]["sell_qty"] = portfolio["qty"]
            self.positions[p, m, s]["sell_price"] = price
        elif position == -1:
            self.positions[p, m, s]["buy_qty"] = abs(portfolio["qty"])
            self.positions[p, m, s]["buy_price"] = price

        self.positions[p, m, s]["fee"] = fee
        self.positions[p, m, s]["slippage"] = slippage
        self.positions[p, m, s]["equity"] = 0

        self.positions[p, m, s]["qty"] = 0
        self.positions[p, m, s]["position"] = 0
        self.positions[p, m, s]["duration"] = 0
        self.positions[p, m, s]["entry_price"] = 0

    def _update_position(self, p, m, s):
        position_type = self.positions[p, m, s]["position"]
        price = self.market_data.open_[p, m]
        change_exposure, change_pct = self._calculate_change_exposure(p, m, s, price)

        self.positions[p, m, s]["duration"] += 1

        if not self.config.rebalance_position \
        or change_pct < self.config.minimum_change:
            return
        
        fee , slippage = self._calculate_fee_and_slippage(change_exposure)
        change_qty = (change_exposure - fee - slippage) / price

        if position_type == 1:
            if change_qty > 0 and not self.config.increase_allowed:
                return
            elif change_qty < 0 and not self.config.decrease_allowed:
                return

        if position_type == -1:
            if change_qty < 0 and not self.config.increase_allowed:
                return
            elif change_qty > 0 and not self.config.decrease_allowed:
                return

        if change_qty > 0:
            self._process_buy(p, m, s, change_exposure, price)
        elif change_qty < 0:
            self._process_sell(p, m, s, -change_exposure, price)
    
    # ..................................................................................
    def _process_buy(self, p, m, s, quote_qty, price):
        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = (quote_qty - fee - slippage) / price

        self.positions[p, m, s]["buy_qty"] = base_qty
        self.positions[p, m, s]["buy_price"] = price
        self.positions[p, m, s]["quote_qty"] -= quote_qty

        self.positions[p, m, s]["qty"] += base_qty
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage

    def _process_sell(self, p, m, s, quote_qty, price):
        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = quote_qty / price

        self.positions[p, m, s]["sell_qty"] = abs(base_qty)
        self.positions[p, m, s]["sell_price"] = price
        self.positions[p, m, s]["quote_qty"] += quote_qty - fee - slippage

        self.positions[p, m, s]["qty"] -= base_qty
        self.positions[p, m, s]["fee"] += fee
        self.positions[p, m, s]["slippage"] += slippage

    def _calculate_change_exposure(self, p, m, s, price) -> tuple[float, float]:
        """Calculate the change in exposure for a given position/asset.
        
        Parameters:
        -----------
        p (int): period index
        m (int): market index
        s (int): strategy index
        price (float): buy/sell price

        Returns:
        --------
        tuple (float, float): change in exposure (quote asset) and percentage change
        """
        
        # calculate the current exposure
        current_exposure = self.base_qty[p - 1, m, s] * price

        # calulate the target exposure
        equity = current_exposure + self.positions[p - 1, m, s]["quote_qty"]
        target_exposure =  equity * self.signals[p-1, m, s] * self.leverage[p, m]

        # even if the leverage value is smaller that the max allowed
        # leverage, signals can be >1 or <-1, so we need to check the 
        # effective leverage again and adjust if necessary
        effective_leverage = np.abs(target_exposure / equity)

        if effective_leverage > self.config.max_leverage:
            target_exposure /= (effective_leverage / self.config.max_leverage)

        # calculate the change (absolute & percentage)      
        change = target_exposure - current_exposure

        return (
            change, 
            change / current_exposure if current_exposure > 0 else effective_leverage,
        )

    def _calculate_leverage(self, p, m, s) -> float:
        price = self.market_data.open_[p, m]
        current_exposure = self.positions[p - 1, m, s]["qty"] * price
        equity = current_exposure + self.positions[p - 1, m, s]["quote_qty"]
        
        return abs(current_exposure / equity)  

    def _calculate_fee_and_slippage(self, change_quote):
        # calculate fee & slippage
        fee = self.config.fee_rate * change_quote
        slippage = self.config.slippage_rate * change_quote
        return abs(fee), abs(slippage)
