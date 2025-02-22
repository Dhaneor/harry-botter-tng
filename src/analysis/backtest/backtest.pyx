# cython: language_level=3
# cython: boundscheck=False
# distutils: language = c++


import numpy as np
cimport numpy as np
from numpy cimport float32_t, uint16_t, int8_t
from libc.math cimport fabs

from ..models.market_data_store cimport MarketDataStore
from ..models.account cimport  Account
from ..dtypes import POSITION_DTYPE, PORTFOLIO_DTYPE

cdef int WARMUP_PERIODS = 200
cdef double SENTINEL = -1.0

cdef double fee_rate = 0.001
cdef double slippage_rate = 0.001


cdef struct PositionsC:
    int8_t position
    double qty
    double quote_qty
    double entry_price
    uint16_t duration
    double equity
    double buy_qty
    double buy_price
    double sell_qty
    double sell_price
    double fee
    double slippage
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
        # PositionsC[:, :, :] positions
        np.ndarray positions
        np.ndarray portfolio
        double[:, :] quote_qty_global
        double[:, :] equity_global
        double[:, :] leverage_global
        double[:, :] total_value
        double[:, :] max_value_global
        double[:, :] drawdown_global

        int8_t[:, :, :] position
        double[:, :, :] qty
        double[:, :, :] quote_qty
        double[:, :, :] entry_price
        uint16_t[:, :, :] duration
        double[:, :, :] equity
        double[:, :, :] buy_qty
        double[:, :, :] buy_price
        double[:, :, :] sell_qty
        double[:, :, :] sell_price
        double[:, :, :] fee
        double[:, :, :] slippage
        float[:, :, :] asset_weight
        float[:, :, :] strategy_weight

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

        self.periods = signals.shape[0]
        self.markets = signals.shape[1]
        self.strategies = signals.shape[2]

        # initialize Position array
        cdef np.ndarray positions_np = np.zeros(
            (self.periods, self.markets, self.strategies), 
            dtype=POSITION_DTYPE
        )
        self.positions = positions_np
        
        self.position = positions_np["position"]
        self.qty = positions_np["qty"]
        self.quote_qty = positions_np["quote_qty"]
        self.entry_price = positions_np["entry_price"]
        self.duration = positions_np["duration"]
        self.equity = positions_np["equity"]
        self.buy_qty = positions_np["buy_qty"]
        self.buy_price = positions_np["buy_price"]
        self.sell_qty = positions_np["sell_qty"]
        self.sell_price = positions_np["sell_price"]
        self.fee = positions_np["fee"]
        self.slippage = positions_np["slippage"]
        self.asset_weight = positions_np["asset_weight"]
        self.strategy_weight = positions_np["strategy_weight"]

        cdef double a_weight = 1 / self.markets
        cdef double s_weight = 1 / self.strategies
        cdef int s, m
        cdef double initial_capital = self.config.initial_capital * a_weight * s_weight

        for s in range(self.strategies):
            for m in range(self.markets):
                self.asset_weight[WARMUP_PERIODS - 1, m, s] = a_weight
                self.strategy_weight[WARMUP_PERIODS - 1, m, s] = s_weight
                self.quote_qty[WARMUP_PERIODS - 1, m, s] = initial_capital
                self.equity[WARMUP_PERIODS - 1, m, s] = initial_capital
        
        # initialize Portfolio array
        self.portfolio = np.zeros(
            shape=(self.periods, self.strategies), 
            dtype=PORTFOLIO_DTYPE
            )
        self.quote_qty_global = self.portfolio["quote_balance"]
        self.equity_global = self.portfolio["equity"]
        self.leverage_global = self.portfolio["leverage"]
        self.total_value = self.portfolio["total_value"]
        self.max_value_global = self.portfolio["max_value"]
        self.drawdown_global = self.portfolio["drawdown"]

        capital_per_strategy = self.config.initial_capital / self.strategies
        
        for s in range(self.strategies):
            self.portfolio[WARMUP_PERIODS - 1, s]["quote_balance"] = capital_per_strategy
            self._update_portfolio(WARMUP_PERIODS - 1, s)

    def run(self):
        self._run()
        return self.positions, self.portfolio

    cdef void _run(self) nogil:
        cdef int p, m, s

        # process the signals for each trading interval (period)
        for p in range(WARMUP_PERIODS, self.periods):
            # copy relevant values from the previous period
            # self.qty[p, :, :] = self.qty[p-1, :, :]
            # self.quote_qty[p, :, :] = self.quote_qty[p-1, :, :]
            # self.duration[p, :, :] = self.duration[p-1, :, :]
            # ... strategy by strategy
            for s in range(self.strategies):
                # market by market
                for m in range(self.markets):
                    self._process_period(p, m, s)
                
                self._update_portfolio(p, s)
    
    # ..................................................................................
    cdef inline void _update_portfolio(self, int p, int s) noexcept nogil:
        cdef double asset_value, quote_balance, total_value, leverage, max_value
        cdef int m

        # Sum the equity values for all markets for the current period and strategy
        # asset_value= np.sum(self.positions[p, :, s]["equity"])
        asset_value = 0.0
        for m in range(self.markets): 
            asset_value += self.equity[p, m, s]
        
        quote_balance = self.quote_qty_global[p, s]
        
        total_value = asset_value + quote_balance
        leverage = (fabs(asset_value) / total_value) if total_value > 0 else 0
         
        self.leverage_global[p, s] = leverage    
        self.equity_global[p, s]= asset_value
        self.total_value[p, s] = total_value

        max_value = self.max_value_global[p-1, s]

        if total_value > max_value:
            self.max_value_global[p, s] = total_value
            max_value = total_value
        else:
            self.max_value_global[p, s] = max_value

        self.drawdown_global[p, s] = 1 - (total_value / max_value)

    cdef void _process_period(self, int p, int m, int s) noexcept nogil:
        cdef double signal, prev_signal
        cdef int open_long, close_long, lopen_short, lclose_short

        position = self.position[p-1, m, s]   

        self.asset_weight[p, m, s] = self.asset_weight[p-1, m, s]
        self.strategy_weight[p, m, s] = self.strategy_weight[p-1, m, s]
        
        signal = self.signals[p-1, m, s]
        prev_signal = self.signals[p-2, m, s] if p > 1 else 0
        
        open_long = 1 if signal > 0 and prev_signal <= 0 else 0
        close_long = 1 if  signal <= 0 and prev_signal > 0 else 0
        open_short = 1 if signal < 0 and prev_signal >= 0 else 0
        close_short = 1 if signal >= 0 and prev_signal < 0 else 0

        if position == 1:
            if close_long == 1 or open_short == 1:
                self._close_position(p, m, s)
            else:
                self._update_position(p, m, s)

        if position == -1:
            if close_short == 1 or open_long == 1:
                self._close_position(p, m, s)
            else:
                self._update_position(p, m, s)

        if position == 0:
            if open_long == 1:
                self._open_position(p, m, s, 1)
            elif open_short == 1:
                self._open_position(p, m, s, -1)
            else:
                self.quote_qty[p, m, s] = self.quote_qty[p-1, m, s] 
        
        self.equity[p, m, s] = (
            self.market_close[p, m]
            * self.qty[p, m, s]
            + self.quote_qty[p, m, s]
        )

    # ............................ PROCESSINNG POSITIONS ...............................
    cdef inline void _open_position(self, int p, int m, int s, int type) noexcept nogil:
        cdef double price = self.market_open[p, m]
        cdef double exposure_quote

        exposure_quote, _ = self._calculate_change_exposure(p, m, s, price)

        self.position[p, m, s] = type
        self.entry_price[p, m, s] = price
        self.duration[p, m, s] = 1

        if exposure_quote > 0:
            self._process_buy(p, m, s, exposure_quote, price)
        elif exposure_quote < 0:
            self._process_sell(p, m, s, exposure_quote, price)

    cdef inline void _close_position(self, int p, int m, int s) noexcept nogil:
        cdef double  price = self.market_open[p, m]
        cdef int position = self.position[p-1, m, s]
        cdef double change_quote
        cdef double fee, slippage

        if position == 0:
            return
        
        change_quote = self.qty[p-1, m, s] * price * -1
        fee, slippage = self._calculate_fee_and_slippage(change_quote)

        self.quote_qty[p, m, s] = self.quote_qty[p-1, m, s] - change_quote - fee - slippage
        
        # Update portfolio
        if position == 1:
            self.sell_qty[p, m, s] = self.qty[p-1, m, s]
            self.sell_price[p, m, s] = price
        elif position == -1:
            self.buy_qty[p, m, s] = fabs(self.qty[p-1, m, s])
            self.buy_price[p, m, s] = price

        self.fee[p, m, s] = fee
        self.slippage[p, m, s] = slippage
        self.equity[p, m, s] = 0

        self.qty[p, m, s] = 0
        self.position[p, m, s] = 0
        self.duration[p, m, s] = 0
        self.entry_price[p, m, s] = 0

    cdef inline void _update_position(self, int p, int m, int s) noexcept nogil:
        self.position[p, m, s] = self.position[p-1, m, s]
        self.duration[p, m, s] = self.duration[p-1, m, s] + 1
        self.entry_price[p, m, s] = self.entry_price[p-1, m, s]

        cdef int position_type = self.position[p, m, s]
        cdef double price = self.market_open[p, m]
        cdef double change_exposure, change_pct, fee, slippage
        cdef double current_qty, change_qty
        cdef int can_rebalance

        change_exposure, change_pct = self._calculate_change_exposure(p, m, s, price)        
        fee , slippage = self._calculate_fee_and_slippage(change_exposure)
        
        change_qty = (change_exposure - fee - slippage) / price
        current_qty = self.qty[p-1, m, s]

        can_rebalance = self._can_rebalance(current_qty, change_qty)

        if can_rebalance == 0:
            self.qty[p, m, s] = self.qty[p-1, m, s]
            self.quote_qty[p, m, s] = self.quote_qty[p-1, m, s]
            return

        if change_qty > 0:
            self._process_buy(p, m, s, change_exposure, price)
        elif change_qty < 0:
            self._process_sell(p, m, s, change_exposure, price)
    
    # ..................................................................................
    cdef inline void _process_buy(
        self, int p, int m, int s, double quote_qty,  double price
    ) noexcept nogil:
        cdef double fee, slippage, base_qty

        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = (quote_qty - fee - slippage) / price

        self.buy_qty[p, m, s] = base_qty
        self.buy_price[p, m, s] = price
        self.quote_qty[p, m, s] = self.quote_qty[p-1, m, s] - quote_qty

        self.qty[p, m, s] = self.qty[p-1, m, s] + base_qty
        self.fee[p, m, s] = fee
        self.slippage[p, m, s] = slippage

    cdef inline void _process_sell(
        self, int  p, int m, int s, double quote_qty, double price
    ) noexcept nogil:
        cdef double fee, slippage, base_qty

        fee , slippage = self._calculate_fee_and_slippage(quote_qty)
        base_qty = quote_qty / price

        self.sell_qty[p, m, s] = abs(base_qty)
        self.sell_price[p, m, s] = price
        self.quote_qty[p, m, s] = self.quote_qty[p-1, m, s] - (quote_qty + fee + slippage)

        self.qty[p, m, s] = self.qty[p-1, m, s] + base_qty
        self.fee[p, m, s] = fee
        self.slippage[p, m, s] = slippage

    cdef inline tuple[double, double] _calculate_change_exposure_old(
        self, int p, int m, int s, double price
    ) noexcept nogil:
        cdef double current_exposure, equity, target_exposure, effective_leverage, change

        # calculate the current exposure
        current_exposure = self.qty[p - 1, m, s] * price

        # calulate the target exposure
        equity = current_exposure + self.quote_qty[p - 1, m, s]
        target_exposure =  equity * self.signals[p-1, m, s] * self.leverage[p, m]

        # even if the leverage value is smaller that the max allowed
        # leverage, signals can be >1 or <-1, so we need to check the 
        # effective leverage again and adjust if necessary
        effective_leverage = fabs(target_exposure / (equity + 1e-8)) 

        if effective_leverage > self.config.max_leverage:
            target_exposure /= (effective_leverage / self.config.max_leverage)

        # calculate the change (absolute & percentage)      
        change = target_exposure - current_exposure

        if current_exposure > 0:
            return change, change / current_exposure
        else:
            return change, effective_leverage

    cdef inline tuple[double, double] _calculate_change_exposure(
        self, int p, int m, int s, double price
    ) noexcept nogil:
        cdef double current_exposure, budget, target_exposure, effective_leverage, change

        # calculate the current exposure
        current_exposure = self.qty[p - 1, m, s] * price

        # calulate the target exposure
        budget = self._get_budget(p, m, s) 
        
        target_exposure = (
            budget
            * self.signals[p-1, m, s] 
            * self.leverage[p-1, m]
        )

        # even if the leverage value is smaller that the max allowed
        # leverage, signals can be >1 or <-1, so we need to check the 
        # effective leverage again and adjust if necessary
        effective_leverage = fabs(target_exposure / (budget + 1e-8)) 

        if effective_leverage > self.config.max_leverage:
            target_exposure /= (effective_leverage / self.config.max_leverage)

        # calculate the change (absolute & percentage)      
        change = target_exposure - current_exposure

        if current_exposure > 0:
            return change, change / current_exposure
        else:
            return change, effective_leverage

    cdef inline double _get_budget(self, int p, int m, int s) noexcept nogil:
        cdef double equity_global, a_weight, strategy_weight

        equity_global = self.equity_global[p-1, s] 
        a_weight = self.asset_weight[p, m, s]
        s_weight = self.strategy_weight[p, m, s]

        return equity_global * a_weight * s_weight

    cdef inline int _can_rebalance(self, double qty, double change_qty) noexcept nogil:
        cdef double change_pct, minimum_change
        cdef int position_type

        if not self.config.rebalance_position:
            return 0

        change_pct = change_qty / qty
        minimum_change = self.config.minimum_change

        if fabs(change_pct) < minimum_change:
            return 0

        position_type = 1 if qty > 0 else -1

        if position_type == 1:
            if (change_qty > 0 and not self.config.increase_allowed) \
            or (change_qty < 0 and not self.config.decrease_allowed) \
            or change_qty == 0:
                return 0 
            
        elif position_type == -1:
            if (change_qty < 0 and not self.config.increase_allowed) \
            or (change_qty > 0 and not self.config.decrease_allowed) \
            or change_qty == 0:
                return 0

        return 1
        
    cdef inline double _calculate_leverage(self, int p, int m, int s):
        cdef double price, current_exposure, equity
        price = self.market_data.open[p, m]
        current_exposure = self.positions[p - 1, m, s].qty * price
        equity = current_exposure + self.positions[p - 1, m, s].quote_qty
        
        return fabs(current_exposure / equity)  

    cdef inline tuple[double, double] _calculate_fee_and_slippage(self, double change_quote) noexcept nogil:
        cdef double fee, slippage

        fee = self.config.fee_rate * change_quote
        slippage = self.config.slippage_rate * change_quote
        return fabs(fee), fabs(slippage)
