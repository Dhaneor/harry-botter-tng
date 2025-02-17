# cython: language_level=3
# cython: boundscheck=False
# distutils: language = c++


import numpy as np
cimport numpy as np
from numpy cimport float64_t, float32_t, int16_t, int8_t
from libc.math cimport fabs
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map


from src.analysis.models.market_data_store cimport MarketDataStore
from src.analysis.models.position cimport TradeData, PositionData, StopOrder
from src.analysis.models.account cimport  Account
from src.analysis.dtypes import POSITION_DTYPE, PORTFOLIO_DTYPE

cdef int WARMUP_PERIODS = 200
cdef double SENTINEL = -1.0

cdef double fee_rate = 0.001
cdef double slippage_rate = 0.001


cdef struct PositionsC:
    int8_t position
    float64_t qty
    float64_t quote_qty
    float64_t entry_price
    int16_t duration
    float64_t equity
    float64_t buy_qty
    float64_t buy_price
    float64_t sell_qty
    float64_t sell_price
    float64_t fee
    float64_t slippage
    float32_t asset_weight
    float32_t strategy_weight


cdef class Config:
    cdef:
        public double initial_capital
        public double max_leverage
        public int rebalance_freq
        public int rebalance_position
        public int increase_allowed
        public int decrease_allowed
        public double minimum_change
        public double fee_rate
        public double slippage_rate

    def __init__(
        self, 
        initial_capital,
        max_leverage = 1.0,
        rebalance_freq = 0,  # in trading periods
        rebalance_position = True, 
        increase_allowed = True, 
        decrease_allowed = True,
        minimum_change = 0.1,
        fee_rate = 0.001,
        slippage_rate = 0.001, 
    ):
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq
        self.rebalance_position = rebalance_position
        self.increase_allowed = increase_allowed
        self.decrease_allowed = decrease_allowed
        self.minimum_change = minimum_change
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate


cdef class BackTestCore:
    cdef:
        public MarketDataStore market_data
        public float[:, :] leverage
        public double[:, :, :] signals
        public Config config
        double[:] a_weights
        double[:] s_weights
        Account account
        int periods, markets, strategies, warmup_periods
        long long[:, :] timestamps
        double[:, :] market_open 
        double[:, :] market_close
        double[:, :, :] equity
        double[:, :, :] base_qty
        double[:, :] quote_qty_global
        double[:, :] equity_global
        double[:, :] leverage_global

    def __cinit__(
        self, 
        MarketDataStore market_data, 
        np.ndarray[np.float32_t, ndim=2] leverage, 
        np.ndarray[np.float64_t, ndim=3] signals, 
        Config config
    ):
        self.market_data = market_data
        self.timestamps = market_data.timestamp  # CONVERT TO C-MEMORY VIEW
        self.market_open = market_data.open  # CONVERT TO C-MEMORY VIEW
        self.market_close = market_data.close  # CONVERT TO C-MEMORY VIEW
        self.leverage = leverage
        self.signals = signals
        self.config = config

        # Initialize a_weights and s_weights as memory views
        cdef np.ndarray[np.float64_t, ndim=1] a_weights_np = (
            np.full(signals.shape[1], 1.0 / signals.shape[1], dtype=np.float64)
        )
        cdef np.ndarray[np.float64_t, ndim=1] s_weights_np = (
            np.full(signals.shape[2], 1.0 / signals.shape[2], dtype=np.float64)
        )
        self.a_weights = a_weights_np
        self.s_weights = s_weights_np
        
        self.account = Account(positions={})

        self.periods = signals.shape[0]
        self.markets = signals.shape[1]
        self.strategies = signals.shape[2]
        self.warmup_periods = WARMUP_PERIODS

        # Initialize arrays
        cdef np.ndarray[np.float64_t, ndim=3] equity_np = np.zeros(
            (self.periods, self.markets, self.strategies), 
            dtype=np.float64
        )
        self.equity = equity_np

        cdef np.ndarray[np.float64_t, ndim=3] base_qty_np = np.zeros(
            (self.periods, self.markets, self.strategies), 
            dtype=np.float64
        )
        self.base_qty = base_qty_np

        self.quote_qty_global = np.zeros((self.periods, self.markets), dtype=np.float64)
        self.equity_global = np.zeros((self.periods, self.markets), dtype=np.float64)
        self.leverage_global = np.zeros((self.periods, self.markets), dtype=np.float64)

        self.quote_qty_global[self.warmup_periods - 1, :] = self.config.initial_capital
        
        cdef int s
        
        for s in range(self.strategies):
            self.equity_global[self.warmup_periods - 1, s] = self.config.initial_capital

    def run(self):
        self._run()
        # return self._build_result_array(), self.equity_global

    cdef void _run(self) nogil:
        cdef double signal, prev_signal
        cdef double price = 0.0
        cdef double current
        cdef int s, p, m, open_long, close_long, open_short, close_short

        for s in range(self.strategies):

            for p in range(self.warmup_periods, self.periods):
                self.quote_qty_global[p, s] = self.quote_qty_global[p -1, s]

                for m in range(self.markets):
                    self.base_qty[p, m, s] = self.base_qty[p - 1, m, s]
                    current = self.base_qty[p, m, s]

                    # determine the necessary action, if any
                    signal = self.signals[p-1, m, s] if p > 0 else 0
                    prev_signal = self.signals[p-2, m, s] if p > 1 else 0

                    if signal > 0 and prev_signal <= 0:
                        open_long = 1
                    elif signal <= 0 and prev_signal > 0:
                        close_long = 1
                    elif signal < 0 and prev_signal >= 0:
                        open_short = 1
                    elif  signal >= 0 and prev_signal < 0:
                        close_short = 1

                    if current > 0:  # we are long
                        if close_long == 1 or open_short == 1:
                            self.market_open[p, m]
                            self._close_position(p, m, s, price)
                        else:
                            self._update_position(p, m, s)
                    
                    elif current < 0:  # we are short
                        if close_short == 1 or open_long == 1:
                            self.market_open[p, m]
                            self._close_position(p, m, s, price)
                        else:
                            self._update_position(p, m, s)

                    else:  # we have no position
                        if open_long == 1:
                            self._open_position(p, m, s, 1)
                        elif open_short == 1:
                            self._open_position(p, m, s, -1)

                    self.equity[p, m, s] = self.base_qty[p, m, s] * self.market_close[p, m]

                self._update_portfolio(p, s)

    # ............................ PROCESSINNG POSITIONS ...............................
    cdef inline void _open_position(self, int p, int m, int s, int type) noexcept nogil:
        cdef double price = self.market_open[p, m]
        cdef double quote_qty
        cdef PositionData pos
        cdef TradeData trade

        pos.idx = m
        pos.type = 1
        pos.is_active = 1
        pos.duration = 1
        pos.size = 0.0
        pos.avg_entry_price = price
        pos.pnl = 0.0
        pos.trades = vector[TradeData]()
        pos.stop_orders = vector[StopOrder]()

        quote_qty, _ = self._calculate_change_exposure(p, m, s, price)

        if quote_qty == 0:
            return

        if type == 1:
            trade = self.build_buy_trade(
                timestamp=self.timestamps[p, 0],
                price=price,
                quote_qty=quote_qty,
                base_qty=SENTINEL
            )
            pos.size += trade.qty
            pos.trades.push_back(trade)
            self.quote_qty_global[p, s] -= quote_qty
        
        elif type == -1:
            trade = self.build_sell_trade(
                timestamp=self.timestamps[p, 0],
                price=price,
                quote_qty=quote_qty,
                base_qty=SENTINEL
            )
            pos.size -= trade.qty
            pos.trades.push_back(trade)
            self.quote_qty_global[p, s] += trade.net_quote_qty

        self.base_qty[p, m, s] = pos.size
        self._add_position(pos, m, s)

    cdef inline void _close_position(self, int p, int m, int s, double price) noexcept nogil:
        cdef PositionData* pos = self._get_current_position(m, s)
        cdef TradeData t

        if pos == NULL:
            return

        if pos.type == 1:
            trade = self.build_sell_trade(
                timestamp=self.timestamps[p, 0],
                price=price, 
                base_qty=pos.size, 
                quote_qty=SENTINEL
            )
            pos.size -= trade.qty
            pos.trades.push_back(trade)
            self.quote_qty_global[p, s] += trade.net_quote_qty
        
        elif pos.type == -1:
            trade = self.build_buy_trade(
                timestamp=self.timestamps[p, 0], 
                price=price, 
                base_qty=pos.size, 
                quote_qty=SENTINEL
            )
            pos.size += trade.qty
            pos.trades.push_back(trade)
            self.quote_qty_global[p, s] -= trade.gross_quote_qty
        else:
            return

        self.base_qty[p, m, s] = 0.0

    cdef inline void _update_position(self, int p, int m, int s) noexcept nogil:
        cdef PositionData* pos = self._get_current_position(m, s)
        cdef TradeData trade
        cdef double change_exposure, change_pct
        cdef int action
    
        if pos == NULL:
            return
    
        pos.duration += 1
        change_exposure, change_pct = self._calculate_change_exposure(p, m, s, self.market_open[p, m])
    
        if not self.config.rebalance_position or change_pct < self.config.minimum_change:
            return
    
        if pos.type == 1:
            if change_exposure > 0:
                if not self.config.increase_allowed:
                    return
                action = 1
            elif change_exposure < 0:
                if not self.config.decrease_allowed:
                    return
                action = -1
    
        if pos.type == -1:
            if change_exposure < 0:
                if not self.config.increase_allowed:
                    return
                action = -1
            elif change_exposure > 0:
                if not self.config.decrease_allowed:
                    return
                action =  1
           
        if action == 1:
            trade = self.build_buy_trade(
                timestamp = self.timestamps[p, 0],
                price = self.market_open[p, m],
                quote_qty=change_exposure,
            )            
            pos.size += trade.qty
            pos.trades.push_back(trade)
            self.base_qty[p, m, s] += trade.qty
        
        elif action == -1:
            trade = self.build_sell_trade(
                timestamp = self.timestamps[p, 0],
                price = self.market_open[p, m],
                quote_qty=change_exposure,
            )
            pos.size -= trade.qty
            pos.trades.push_back(trade)
            self.base_qty[p, m, s] -= trade.qty
    
    # ..................................................................................
    cdef inline TradeData build_buy_trade(
        self,
        long long timestamp,
        double price, 
        double quote_qty = SENTINEL, 
        double base_qty = SENTINEL
    ) noexcept nogil:
        """Get a Trade struct for a buy action."""

        cdef TradeData t
        cdef double fee, slippage, net_quote_qty
        cdef int test

        if quote_qty != SENTINEL:
            fee = quote_qty * fee_rate
            slippage = quote_qty * slippage_rate
            net_quote_qty = quote_qty - fee - slippage
            t.qty = net_quote_qty / price
        
        elif base_qty != SENTINEL:
            t.qty = base_qty
            quote_qty = base_qty * price
            fee = quote_qty * fee_rate
            slippage = quote_qty * slippage_rate
            net_quote_qty = quote_qty - fee - slippage

        else:
            gross_quote_qty = 0.0
            net_quote_qty = 0.0
            fee = 0.0
            slippage = 0.0
            t.qty = 0.0

        t.type = 1
        t.timestamp = timestamp
        t.price = price
        t.gross_quote_qty = quote_qty
        t.net_quote_qty = net_quote_qty
        t.fee = fee
        t.slippage = slippage

        return t

    cdef inline TradeData build_sell_trade(
        self,
        long long timestamp, 
        double price, 
        double quote_qty = SENTINEL,
        double base_qty = SENTINEL
    ) noexcept nogil:
        """Fill an existing Trade struct for a sell action."""

        cdef TradeData t
        cdef double gross_quote_qty
        cdef double fee 
        cdef double slippage

        if base_qty != SENTINEL:
            gross_quote_qty = base_qty * price
            fee = gross_quote_qty * fee_rate
            slippage = gross_quote_qty * slippage_rate

        elif quote_qty != SENTINEL:
            gross_quote_qty = quote_qty
            fee = gross_quote_qty * fee_rate
            slippage = gross_quote_qty * slippage_rate
            base_qty = (gross_quote_qty - fee - slippage) / price

        else:
            gross_quote_qty = 0.0
            fee = 0.0
            slippage = 0.0

        t.type = -1
        t.timestamp = timestamp
        t.price = price
        t.qty = base_qty    
        t.gross_quote_qty = gross_quote_qty
        t.net_quote_qty = gross_quote_qty - fee - slippage
        t.fee = fee
        t.slippage = slippage

        return t

    cdef inline PositionData* _get_current_position(self, int m, int s) noexcept nogil:
        if self.account.positions.find(m) == self.account.positions.end():
            return NULL
    
        if self.account.positions[m].find(s) == self.account.positions[m].end():
            return NULL

        if self.account.positions[m][s].empty():
            return NULL

        return &self.account.positions[m][s].back()

    cdef inline void _add_position(self, PositionData p, int m, int s) noexcept nogil:
        if self.account.positions.find(m) == self.account.positions.end():
            self.account.positions[m] = unordered_map[int, vector[PositionData]]()
        if self.account.positions[m].find(s) == self.account.positions[m].end():
            self.account.positions[m][s] = vector[PositionData]()

        self.account.positions[m][s].push_back(p)       

    cdef inline tuple[double, double] _calculate_change_exposure(
        self, int p, int m, int s, double price
    ) noexcept nogil:
        cdef double current_exposure, budget, target_exposure, effective_leverage, change
        cdef double equity_plus_quote, inv_budget, inv_max_leverage
    
        # Calculate the current exposure
        current_exposure = self.base_qty[p - 1, m, s] * price
    
        # Calculate the target exposure
        equity_plus_quote = self.equity_global[p - 1, s] + self.quote_qty_global[p, s]
        budget = equity_plus_quote * self.a_weights[m]
        target_exposure = budget * self.signals[p-1, m, s] * self.leverage[p, m]
    
        # Check and adjust effective leverage
        inv_budget = 1.0 / (budget + 1e-8)
        effective_leverage = fabs(target_exposure * inv_budget)
    
        if effective_leverage > self.config.max_leverage:
            inv_max_leverage = 1.0 / self.config.max_leverage
            target_exposure *= effective_leverage * inv_max_leverage
    
        # Calculate the change
        change = target_exposure - current_exposure
    
        # Return change and change percentage or effective leverage
        if current_exposure > 0:
            return change, change / current_exposure
        else:
            return change, effective_leverage


    """
    def _calculate_leverage(self, p, m, s) -> float:
        price = self.market_data.open_[p, m]
        current_exposure = self.positions[p - 1, m, s]["qty"] * price
        equity = current_exposure + self.positions[p - 1, m, s]["quote_qty"]
        
        return abs(current_exposure / equity)  

    def _calculate_fee_and_slippage(self, change_quote):
        # calculate fee & slippage
        fee = self.config.fee_rate * change_quote
        slippage = self.config.slippage_rate * change_quote
        return np.abs(fee), np.abs(slippage)
    """


    # ..................................................................................
    cdef inline void _update_portfolio(self, int p, int s) noexcept nogil:
        cdef double quote_qty = self.quote_qty_global[p, s]
        cdef double equity = 0.0
        cdef int m

        for m in range(self.markets):
            equity += self.equity[p, m, s]

        self.equity_global[p, s] = equity
        self.leverage_global[p, s] = fabs(equity) / (equity + quote_qty)

    cdef np.ndarray _build_result_array(self):
        cdef np.ndarray[double, ndim=3] base_qty_array = np.asarray(self.base_qty)
        
        out = np.zeros(
            (self.periods, self.markets, self.strategies), 
            dtype=POSITION_DTYPE
        )
        out["qty"] = base_qty_array
        
        out["position"] = np.select(
            [base_qty_array > 0.0, base_qty_array < 0.0],
            [1, -1],
            default=0
        )
        
        return out