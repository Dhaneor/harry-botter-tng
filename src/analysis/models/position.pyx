# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cimport numpy as np
import numpy as np
from functools import reduce
from time import time
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared

# define fee and slippage rate. might later be replaced with
# imported values from  a config file, but these are good defualt
# values which do not need to be changed
cdef double fee_rate = 0.001
cdef double slippage_rate = 0.001


# .......................... Functions to process positions ............................
cdef inline double get_fee(double qty, double fee_rate):
    return qty * fee_rate

cdef inline double get_slippage(double qty, double slippage_rate):
    return qty * slippage_rate


cdef inline TradeData build_buy_trade(long long timestamp, double quote_qty, double price):
    """Get a Trade struct for a buy action."""
    cdef TradeData t
    cdef double fee = get_fee(quote_qty, fee_rate)
    cdef double slippage = get_slippage(quote_qty, slippage_rate)
    cdef double net_quote_qty = quote_qty - fee - slippage
    
    t.type = 1
    t.timestamp = timestamp
    t.price = price
    t.qty = net_quote_qty / price
    t.gross_quote_qty = quote_qty
    t.net_quote_qty = net_quote_qty
    t.fee = fee
    t.slippage = slippage

    return t

cdef inline TradeData build_sell_trade(long long timestamp, double base_qty, double price):
    """Fill an existing Trade struct for a sell action."""
    cdef TradeData t
    cdef double gross_quote_qty = base_qty * price
    cdef double fee = get_fee(gross_quote_qty, fee_rate)
    cdef double slippage = get_slippage(gross_quote_qty, slippage_rate)
    
    t.type = -1
    t.timestamp = timestamp
    t.price = price
    t.qty = base_qty    
    t.gross_quote_qty = gross_quote_qty
    t.net_quote_qty =gross_quote_qty - fee - slippage
    t.fee = fee
    t.slippage = slippage

    return t


cdef void add_buy(PositionData* pos, long long timestamp, double quote_qty, double price):
    """Adds a Buy trade to the position"""
    cdef TradeData trade = build_buy_trade(timestamp, quote_qty, price) 
    pos.size += trade.qty
    pos.trades.push_back(trade)

cdef void add_sell(PositionData* pos, long long timestamp, double base_qty, double price):
    """Adds a Sell trade to the position"""
    cdef TradeData trade = build_sell_trade(timestamp, base_qty, price) 
    pos.size -= trade.qty
    pos.trades.push_back(trade)


cdef inline double get_avg_entry_price(PositionData* pos):
    cdef double sum_qty = 0
    cdef double sum_quote_qty = 0
    cdef int i

    for i in range(pos[0].trades.size()):
        # sum buys for long position
        if pos[0].type == 1 and pos[0].trades[i].type == 1:
            sum_qty += pos[0].trades[i].qty
            sum_quote_qty += pos[0].trades[i].gross_quote_qty
        # sum sells for short position
        elif pos[0].type == -1 and pos[0].trades[i].type == -1:
            sum_qty += pos[0].trades[i].qty
            sum_quote_qty += pos[0].trades[i].net_quote_qty

    return (sum_quote_qty * 1e6) / (sum_qty * 1e6)


cdef PositionData build_long_position(int index, long long timestamp, double quote_qty, double price):
    """Builds a long position (PositionData stuct)."""
    cdef PositionData pos

    pos.idx = index
    pos.type = 1
    pos.is_active = 1
    pos.duration = 1
    pos.size = 0.0
    pos.avg_entry_price = price
    pos.pnl = 0.0
    pos.trades = vector[TradeData]()
    pos.stop_orders = vector[StopOrder]()
    
    add_buy(&pos, timestamp, quote_qty, price)

    return pos

cdef PositionData build_short_position(int index, long long timestamp, double base_qty, double price):
    """Builds a short position (PositionData stuct)."""
    cdef PositionData pos

    pos.idx = index
    pos.type = -1
    pos.is_active = 1
    pos.duration = 1
    pos.size = 0.0
    pos.avg_entry_price = price
    pos.pnl = 0.0
    pos.trades = vector[TradeData]()
    pos.stop_orders = vector[StopOrder]()

    add_sell(&pos, timestamp, base_qty, price)

    return pos


# ............... Python accessible versions of the position functions .................
# I couldn't find a working solution to test position functions with 
# pytest when they are defined with cdef. I also did not want to use 
# cpdef to avoid the added call overhead. That's why each of the 
# functions above has a wrapper function.

def _get_fee(double qty, double fee_rate):
    return get_fee(qty, fee_rate)

def _get_slippage(double qty, double slippage_rate):
    return get_slippage(qty, slippage_rate)

def _build_buy_trade(long long timestamp, double quote_qty, double price):
    return build_buy_trade(timestamp, quote_qty, price)

def _build_sell_trade(long long timestamp, double base_qty, double price):
    return build_sell_trade(timestamp, base_qty, price)

def _get_avg_entry_price(PositionData pos):
    return get_avg_entry_price(&pos)

def _add_buy(PositionData pos, long long timestamp, double quote_qty, double price):
    add_buy(&pos, timestamp, quote_qty, price)
    return pos

def _add_sell(PositionData pos, long long timestamp, double base_qty, double price):
    add_sell(&pos, timestamp, base_qty, price)
    return pos

def _build_long_position(int index, long long timestamp, double quote_qty, double price):
    return build_long_position(index, timestamp, quote_qty, price)

def _build_short_position(int index, long long timestamp, double base_qty, double price):
    return build_short_position(index, timestamp, base_qty, price)


cdef class FuncBench:
    cdef int iterations

    def __cinit__(self, int iterations = 1000):
        self.iterations = iterations

    def run(self):
        funcs = {
            "get_fee": self.bench_get_fee,
            "get_slippage": self.bench_get_slippage,
            "build_buy_trade": self.bench_build_buy_trade,
            "build_sell_trade": self.bench_build_sell_trade,
            "add_buy": self.bench_add_buy,
            "add_sell": self.bench_add_sell,
            "avg_entry_price": self.bench_get_avg_entry_price,
            "build_long_position": self.bench_build_long_position,
            "build_short_position": self.bench_build_short_position
        }
        for name, func in funcs.items():
            exc_time, avg_exc_time = func()
            print(f"[{name}] exc time: {exc_time:,.0f}µs")
            print(f"[{name}] avg exc time: {avg_exc_time:,.2f}ns")
            print("-" * 80)

    cdef bench_get_fee(self):
        st = time()
        for i in range(self.iterations):
            get_fee(1000, fee_rate)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_get_slippage(self):
        st = time()
        for i in range(self.iterations):
            get_slippage(1000, fee_rate)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_build_buy_trade(self):
        st = time()
        for i in range(self.iterations):
            build_buy_trade(17345000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_build_sell_trade(self):
        st = time()
        for i in range(self.iterations):
            build_sell_trade(17345000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_add_buy(self):
        pos = build_long_position(1, 17345000000, 1000.0, 100.0)
        st = time()
        for i in range(self.iterations):
            add_buy(&pos, 1735000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_add_sell(self):
        pos = build_short_position(1, 17345000000, 1000.0, 100.0)
        st = time()
        for i in range(self.iterations):
            add_sell(&pos, 1735000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_get_avg_entry_price(self):
        pos = build_short_position(1, 17345000000, 1000.0, 100.0)
        add_sell(&pos, 1736000000, 1000.0, 100.0)
        add_sell(&pos, 1737000000, 1000.0, 100.0)
        add_buy(&pos, 1738000000, 1000.0, 100.0)
        st = time()
        for i in range(self.iterations):
            get_avg_entry_price(&pos)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_build_long_position(self):
        st = time()
        for i in range(self.iterations):
            build_long_position(1, 17345000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

    cdef bench_build_short_position(self):
        st = time()
        for i in range(self.iterations):
            build_short_position(1, 17345000000, 1000.0, 100.0)
        exc_time = (time() - st) * 1e6
        avg_exc_time = (exc_time / self.iterations) * 1_000
        
        return exc_time, avg_exc_time

cpdef void run_func_bench(int iterations):
    fb = FuncBench(iterations)
    fb.run()






















# ======================================================================================
#                                                                                      #
#             ----- BENCHMARK FOR DETERMINING THE BEST DATA STRUCTURE -----            #
#                                                                                      #
#                                                                                      #
#                        Position class with C struct elements                         #
# ======================================================================================
cdef class PositionStruct:
    cdef int idx
    cdef int type
    cdef int is_active
    cdef int duration
    cdef vector[TradeData] trades

    def __cinit__(self, int symbol_idx):
        self.idx = symbol_idx
        self.trades = vector[TradeData]()  # Initialize empty vector

    cdef void add_buy(self, long long timestamp, double quote_qty, double price):
        self.trades.push_back(build_buy_trade(timestamp, quote_qty, price))

    cdef void add_sell(self, long long timestamp, double base_qty, double price):
        self.trades.push_back(build_sell_trade(timestamp, base_qty, price))


cdef class PositionStructPointer:
    cdef int idx
    cdef int type
    cdef int is_active
    cdef int duration
    cdef vector[shared_ptr[TradeData]] trades  # Use shared_ptr instead of raw pointers

    def __cinit__(self, int symbol_idx):
        self.idx = symbol_idx
        self.trades = vector[shared_ptr[TradeData]]()  # Initialize empty vector

    cdef void add_buy(self, long long timestamp, double quote_qty, double price):
        """Store a Buy trade using shared_ptr."""
        self.trades.push_back(make_shared[TradeData](build_buy_trade(timestamp, quote_qty, price)))

    cdef void add_sell(self, long long timestamp, double base_qty, double price):
        """Store a Sell trade using shared_ptr."""
        self.trades.push_back(make_shared[TradeData](build_sell_trade(timestamp, base_qty, price)))

    def __dealloc__(self):
        """Automatic cleanup is handled by shared_ptr, so no need for manual free()."""
        self.trades.clear()


# ======================================================================================
#                    Position class with Cython extension elements                     #
# ======================================================================================

# ............................... Trade Action classes .................................
cdef class Buy(ActionInterface):

    def __cinit__(self, np.int64_t timestamp, double amount, double price):
        self.data.type = 1
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
        self.data.type = -1
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


# ................................ Position class(es) ..................................
cdef class PositionExt:
    """A class to represent a trading position."""

    cdef int idx
    cdef int type
    cdef int is_active
    cdef int duration
    cdef list[ActionInterface] trades

    def __cinit__(self, int symbol_idx):
        self.idx = symbol_idx
        self.trades = []

    cdef void add_buy(self, long long timestamp, double quote_qty, double price):
        self.trades.append(Buy(timestamp, quote_qty, price))

    cdef void add_sell(self, long long timestamp, double base_qty, double price):
        self.trades.append(Sell(timestamp, base_qty, price))


# ======================================================================================
#                    Benchmark that uses the classes defined above                     #
# ======================================================================================
cdef class Benchmark:
    cdef list[PositionStructPointer] pointer
    cdef list[PositionStruct] struct
    cdef list[PositionExt] ext

    def __cinit__(self, int num_positions):
        self.pointer = [PositionStructPointer(pos) for pos in range(num_positions)] 
        self.struct = [PositionStruct(pos) for pos in range(num_positions)]
        self.ext = [PositionExt(pos) for pos in range(num_positions)]

    cdef benchmark_struct(self, int num_trades):
        for i in range(num_trades):
            build_buy_trade(1000, 10.0, 100.0)
            build_sell_trade(1000, 10.0, 100.0)

    cdef benchmark_ext(self, int num_trades):
        for i in range(num_trades):
            Buy(1000, 10.0, 100.0)
            Sell(1000, 10.0, 100)

    cdef benchmark_struct_pointer_class(self, int num_trades):
        cdef PositionStructPointer pos
        for pos in self.pointer:
            for i in range(num_trades):
                pos.add_buy(timestamp=i, quote_qty=100 + i, price=50000 + i)
                pos.add_sell(timestamp=i, base_qty=0.002 + i * 0.0001, price=50000 + i)

    cdef benchmark_struct_class(self, int num_trades):
        cdef PositionStruct pos
        for pos in self.struct:
            for i in range(num_trades):
                pos.add_buy(timestamp=i, quote_qty=100 + i, price=50000 + i)
                pos.add_sell(timestamp=i, base_qty=0.002 + i * 0.0001, price=50000 + i)

    cdef benchmark_ext_class(self, int num_trades):
        cdef PositionExt pos
        for pos in self.ext:
            for i in range(num_trades):
                pos.add_buy(timestamp=i, quote_qty=100 + i, price=50000 + i)
                pos.add_sell(timestamp=i, base_qty=0.002 + i * 0.0001, price=50000 + i)


cpdef void compare_position_classes(int num_positions=1000, int num_trades=1000):
    bm = Benchmark(num_positions)

    st = time()
    bm.benchmark_struct(num_trades * num_positions)

    struct_time = time()- st
    struct_time_avg = struct_time / (2 * num_trades * num_positions) * 1e9

    st = time()
    bm.benchmark_ext(num_trades * num_positions)
    
    ext_time = time() - st
    ext_time_avg = ext_time / (2 * num_trades * num_positions) * 1e9

    print("Benchmark object creation time ...\n")
    print(f"[Struct] overall time: {struct_time:,.2f}s")
    print(f"[Struct] average time: {struct_time_avg:,.0f}ns")

    print(f"[Ext] overall time: {ext_time:,.2f}s")
    print(f"[Ext] average time: {ext_time_avg:,.0f}ns")
    print(f"speedup: {(ext_time_avg / struct_time_avg):.1f}x")
    print("----------------------------------------˜n")

    st = time()
    bm.benchmark_struct_pointer_class(num_trades)

    pointer_time = time()- st
    pointer_time_avg = pointer_time / (2 * num_positions * num_trades) * 1e9 

    st = time()
    bm.benchmark_struct_class(num_trades)

    struct_time = time()- st
    struct_time_avg = struct_time / (2 * num_positions * num_trades) * 1e9 

    st = time()
    bm.benchmark_ext_class(num_trades)
    
    ext_time = time() - st
    ext_time_avg = ext_time / (2 * num_positions * num_trades) * 1e9

    print("Benchmark class method execution time ...\n")

    print(f"[Pointer] overall time: {pointer_time:,.2f}s")
    print(f"[Pointer] average time: {pointer_time_avg:,.0f}ns")

    print(f"[Struct] overall time: {struct_time:,.2f}s")
    print(f"[Struct] average time: {struct_time_avg:,.0f}ns")

    print(f"[Ext] overall time: {ext_time:,.2f}s")
    print(f"[Ext] average time: {ext_time_avg:,.0f}ns")

    print(f"speedup poitner: {(ext_time_avg / pointer_time_avg):.1f}x")
    print(f"speedup struct: {(ext_time_avg / struct_time_avg):.1f}x")















# ======================================================================================
#                    Position class with Cython extension elements                     #
# ======================================================================================
cdef class Position:
    """A class to represent a trading position."""

    def __cinit__(self, str symbol):
        self.buys = []
        self.sells = []
        self.type = 0
        self.is_active = 0

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
        self._update(action)

    cpdef void close(self, np.int64_t timestamp, double price):
        """Adds a new buy or sell action to the position.
        
        Python wrapper method which allows to define the actual 
        _add_action method for pure cythonic access.
        """
        self._close(timestamp, price)

    cdef void _open(self, np.int64_t timestamp, double size, double price) except *:
        if price <= 0:
            raise ValueError("price must be positive")

        if size == 0:
            raise ValueError("size cannot be 0")

        if size > 0:
            action = Buy(timestamp, size, price)
        elif size < 0:
            action = Sell(timestamp, size, size)

    cdef void _close(self, np.int64_t timestamp, double price):
        if self.current_qty > 0:
            self.add_action(Sell(timestamp, self.current_qty, price))
        if self.current_qty < 0:
            self.add_action(Buy(timestamp, self.current_qty, price))

    cdef void _update(self, ActionInterface action) except *:
        cdef double old_qty = self.current_qty
        cdef double old_value = old_qty * self.average_entry_price

        # check if this position was just opened and if yes, set 
        # type/active flags
        if old_qty == 0:
            self.is_active == 1
            self.type = 1 if action.data.qty > 0 else -1

        # ............................ process BUY action ..............................
        # 
        if action.data.type == 1:  # BUY
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

            self.buys.append(action)
        
        # ............................ process SELL action ..............................
        elif action.data.type == -1:  # SELL
            new_qty = self.current_qty - action.data.qty
        
            if old_qty > 0 and new_qty < 0:
                raise ValueError(
                    f"Selling {action.data.qty} would change this long position to a "
                    f"short position. Use .close() method before opening a new short "
                    f"position, or sell a max amount of {self.current_qty}."
                )

            self.current_qty = new_qty
            self.sells.append(action)

        # raise Exception if the action type is not defined as 1 or -1
        else:
            raise ValueError(f"Unknown action type: {action.type}")

        # ..............................................................................
        # calculate the PNL if the position is closed now
        if self.current_qty == 0:
            self.realized_pnl = abs(
                self.average_entry_price - self.average_exit_price
            ) * old_qty
            self.is_active = 0

    cdef double _get_average_buy_price(self):
        cdef double qty_bought = 0.0
        cdef double quote_spent = 0.0

        for buy in self.buys:
            qty_bought += buy.data.qty
            quote_spent += buy.data.quote_qty

        return quote_spent / qty_bought

    cdef double _get_average_sell_price(self):
        cdef double qty_sold = 0.0
        cdef double quote_rcvd = 0.0

        for sell in self.sells:
            qty_sold += sell.data.qty
            quote_rcvd += sell.data.quote_qty

        return quote_rcvd / qty_sold


