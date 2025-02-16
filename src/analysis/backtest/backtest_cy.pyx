# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
import logging
import numpy as np
from libc.stdio cimport printf
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

from src.analysis.models.market_data_store cimport MarketDataStore, MarketState
from src.analysis.models.position cimport (
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
from src.analysis.models.account cimport (
    Account, 
    get_current_position, 
    add_position,
    update_position,
    print_positions, 
)

logger = logging.getLogger(f"main.{__name__}")

WARMUP_PERIODS = 200


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
        public np.ndarray leverage
        public np.ndarray signals
        public Config config
        list[double] a_weights
        list[double] s_weights
        Account account
        tuple[int, int, int] shape
        public np.ndarray equity
        public np.ndarray base_qty
        public np.ndarray quote_qty_global
        public np.ndarray equity_global
        public np.ndarray leverage_global

    def __cinit__(self, MarketDataStore market_data, np.ndarray leverage, np.ndarray signals, Config config):
        self.market_data = market_data
        self.leverage = leverage
        self.signals = signals
        self.config = config

        self.a_weights = [1 / signals.shape[1] for _ in range(signals.shape[1])]
        self.s_weights = [1 / signals.shape[2] for _ in range(signals.shape[2])]

        # Initialize one account for each strategy
        self.account = Account(positions={})

        self.shape = (signals.shape[0], signals.shape[1], signals.shape[2],)

        # Initialize arrays
        self.equity = np.zeros(self.shape, dtype=np.float64)
        self.base_qty = np.zeros(self.shape, dtype=np.float64)

        self.quote_qty_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.equity_global = np.zeros(self.shape[:1], dtype=np.float64)
        self.leverage_global = np.zeros(self.shape[:1], dtype=np.float64)

    cpdef run(self):
        cdef int signal, prev_signal, periods, markets, strategies
        cdef double price

        periods, markets, strategies = self.shape

        for s in range(strategies):

            for p in range(periods):

                for m in range(markets):
                    # determine the necessary action, if any
                    signal = self.signals[p, m, s] if p > 0 else 0
                    prev_signal = self.signals[p-2, m, s] if p > 1 else 0

                    open_long = signal > 0 and prev_signal <= 0
                    close_long = signal <= 0 and prev_signal > 0
                    open_short = signal < 0 and prev_signal >= 0
                    close_short = signal >= 0 and prev_signal < 0

                    if self.base_qty[p-1, m, s] > 0:  # we are long
                        if close_long or open_short:
                            price = self.market_data.open[p, m]
                            self._close_position(p, m, s, price)
                        else:
                            self._update_position(p, m, s)
                    
                    elif self.base_qty[p-1, m, s] < 0:  # we are short
                        if close_short or open_long:
                            price = self.market_data.open[p, m]
                            self._close_position(p, m, s, price)
                        else:
                            self._update_position(p, m, s)

                    else:  # we have no position
                        if open_long:
                            self._open_position(p, m, s, 1)
                        elif open_short:
                            self._open_position(p, m, s, -1)

        # return self.equity, 0

    # ............................ PROCESSINNG POSITIONS ...............................
    cdef _open_position(self, int p, int m, int s, int type):
        cdef double price = self.market_data.open[p, m]
        cdef double quote_qty
        cdef PositionData pos

        quote_qty, _ = self._calculate_change_exposure(p, m, s, price)

        if type == 1:
            pos = build_long_position(
                m,
                self.market_data.timestamp[p, 0],
                quote_qty,
                price
            )
        elif type == -1:
            pos = build_short_position(
                m,
                self.market_data.timestamp[p, 0],
                quote_qty,
                price
            )
        else:
            raise ValueError(
                "Unsupported position type for opening a position: %s" % type
            )

        self.base_qty[p, m, s] = pos.size
        self.equity[p, m, s] = pos.size * self.market_data.close[p, m]
        self.account = add_position(self.account, m, s, pos)

        # print_positions(self.account)

    cdef _close_position(self, int p, int m, int s, double price):
        cdef PositionData* pos = get_current_position(self.account, m, s)

        if pos == NULL:
            logger.warning(
                "[close] Got NULL, expected PositionData struct."
                "Probable mismatch between arrays and position objects."
            )
            logger.warning("last/current base_qty: %s", self.base_qty[p-1, m, s])
            return

        # close_position(pos, self.market_data.timestamp[p, 0], price)
        self.base_qty[p, m, s] = 0.0

    cdef _update_position(self, int p, int m, int s):
        cdef PositionData* pos = get_current_position(self.account, m, s)
        cdef double price = self.market_data.open[p, m]
        cdef double change_exposure, change_pct
        cdef int action

        if pos == NULL:
            logger.warning(
                f"[update] Got NULL, expected PositionData struct."
                f"Probable mismatch between arrays and position objects."
            )
            logger.warning("last/current base_qty: %s", self.base_qty[p-1, m, s])
            return

        logger.debug(f"[update] Position pointer: {<unsigned long>pos}")
        logger.debug(f"[update] Position type: {pos.type}")
        logger.debug(f"[update] Position size: {pos.size}")
        logger.debug(f"[update] Position duration: {pos.duration}")


        pos.duration += 1 # This line causes the segmentation fault
        return

        change_exposure, change_pct = self._calculate_change_exposure(p, m, s, price)

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

        cdef TradeData t
       
        if action == 1:
            t = build_buy_trade(
                timestamp = self.market_data.timestamp[p, 0],
                price = price,
                quote_qty=change_exposure,
            )
            
            add_buy(pos, &t)
        
        elif action == -1:
            t = build_sell_trade(
                timestamp = self.market_data.timestamp[p, 0],
                price = price,
                quote_qty=change_exposure,
            )
            add_sell(pos, &t)

        self.base_qty[p, m, s] = pos.size
    
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

    cdef tuple[double, double] _calculate_change_exposure(
        self, int p, int m, int s, double price
    ):        
        return 20, 10
        """
        # calculate the current exposure
        current_exposure = self.base_qty[p - 1, m, s] * price

        # calulate the target exposure
        budget = self.equity_global[p] * self.a_weights[m] * self.s_weights[s]
        target_exposure = budget * self.signals[p-1, m, s] * self.leverage[p, m]

        logger.debug("budget: %s (%s)", budget, self.equity_global[p])
        logger.debug("target_exposure: %s", target_exposure)

        # even if the leverage value is smaller that the max allowed
        # leverage, signals can be >1 or <-1, so we need to check the 
        # effective leverage again and adjust if necessary
        effective_leverage = np.abs(target_exposure / budget)

        if effective_leverage > self.config.max_leverage:
            target_exposure /= (effective_leverage / self.config.max_leverage)

        # calculate the change (absolute & percentage)      
        change = target_exposure - current_exposure

        return (
            change, 
            change / current_exposure if current_exposure > 0 else effective_leverage,
        )
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
