from numba import int64, float32
from numba.experimental import jitclass
import numpy as np
import pandas as pd
import math

spec = [
    ('timestamp', int64[:, :]),
    ('open_',     float32[:, :]),
    ('high',      float32[:, :]),
    ('low',       float32[:, :]),
    ('close',     float32[:, :]),
    ('volume',    float32[:, :]),

    # We'll store log_returns / atr / annual_vol only if you want to keep them:
    ('log_returns', float32[:, :]),
    ('atr',         float32[:, :]),
    ('annual_vol',  float32[:, :]),
]

@jitclass(spec)
class MarketDataStore:
    """A class to store raw market data for multiple symbols as 2d arrays."""
    def __init__(self, timestamp, open_, high, low, close, volume):
        self.timestamp = timestamp
        self.open_     = open_
        self.high      = high
        self.low       = low
        self.close     = close
        self.volume    = volume

        rows, cols = close.shape

        # Initialize indicator arrays to zeros or empty
        self.log_returns = np.zeros((rows, cols), dtype=np.float32)
        self.atr         = np.zeros((rows, cols), dtype=np.float32)
        self.annual_vol  = np.zeros((rows, cols), dtype=np.float32)

        self.compute_log_returns()
        self.compute_atr()
        self.compute_annual_vol()

    def compute_log_returns(self):
        """
        Fill self.log_returns with ln(close[i]/close[i-1]) for each column.
        """
        rows, cols = self.close.shape
        if rows == 0:
            return
        # First row = 0
        for j in range(cols):
            self.log_returns[0, j] = 0.0

        for i in range(1, rows):
            for j in range(cols):
                self.log_returns[i, j] = math.log(self.close[i, j] / self.close[i-1, j])

    def compute_atr(self, period=14):
        """
        Compute Wilder's ATR (exponential smoothing) for each column.
        Fills self.atr with shape (rows, cols).

        Steps:
        1) Calculate TR for each bar.
        2) Seed ATR at index (period - 1) by averaging TR[0..period-1].
        3) For i >= period, use: ATR(i) = ((ATR(i-1) * (period - 1)) + TR(i)) / period
        """
        rows, cols = self.close.shape
        if rows == 0:
            return

        # 1) Compute True Range (TR)
        TR = np.zeros((rows, cols), dtype=np.float32)

        # bar 0: TR = high(0) - low(0), because there's no prev close
        for j in range(cols):
            TR[0, j] = self.high[0, j] - self.low[0, j]
        # bars [1..rows-1]
        for i in range(1, rows):
            for j in range(cols):
                prev_close = self.close[i-1, j]
                tr1 = self.high[i, j] - self.low[i, j]
                tr2 = abs(self.high[i, j] - prev_close)
                tr3 = abs(self.low[i, j] - prev_close)
                TR[i, j] = max(tr1, tr2, tr3)

        # 2) Initialize (seed) ATR for the first `period` bars
        #    We'll define ATR(k) for k < period-1 as partial average,
        #    and ATR(period-1) as the average of TR[0..period-1].
        #    Then from bar = period, use Wilder's formula.

        # If the data has fewer bars than `period`, we handle partial availability
        max_seed = min(period, rows)

        for j in range(cols):
            partial_sum = 0.0
            # partial fill up to (period-1) or row-end
            for i in range(max_seed):
                partial_sum += TR[i, j]
                # For i < period, define an average up to i+1 bars
                # This is optional "incremental seed" logic.
                self.atr[i, j] = partial_sum / (i + 1)

        # 3) Wilder's exponential smoothing for i >= period
        for j in range(cols):
            for i in range(period, rows):
                # Formula: ATR(i) = [ATR(i-1)*(period-1) + TR(i)] / period
                prev_atr = self.atr[i-1, j]
                cur_tr   = TR[i, j]
                self.atr[i, j] = (prev_atr * (period - 1) + cur_tr) / period

            # 2) Simple rolling average (naive O(rows*period)):
            for j in range(cols):
                # partial for first period bars
                partial_sum = 0.0
                for i in range(period):
                    if i < rows:
                        partial_sum += TR[i, j]
                        self.atr[i, j] = partial_sum / (i + 1)
                for i in range(period, rows):
                    window_sum = 0.0
                    for k in range(i - period + 1, i+1):
                        window_sum += TR[k, j]
                    self.atr[i, j] = window_sum / period

    def compute_annual_vol(self, period=30):
        """
        Fill self.annual_vol as rolling stdev of log_returns * sqrt(252).
        """
        rows, cols = self.close.shape
        if rows == 0:
            return

        # Need log_returns first:
        # (In reality, you'd ensure compute_log_returns() was called or do it here)
        for j in range(cols):
            # partial fill
            for i in range(period):
                if i < rows:
                    self.annual_vol[i, j] = 0.0
            for i in range(period, rows):
                # compute stdev over the last `period` bars
                sum_lr = 0.0
                for k in range(i - period + 1, i+1):
                    sum_lr += self.log_returns[k, j]
                mean_lr = sum_lr / period

                sum_sq = 0.0
                for k in range(i - period + 1, i+1):
                    diff = self.log_returns[k, j] - mean_lr
                    sum_sq += diff * diff
                var_lr = sum_sq / period
                stdev = math.sqrt(var_lr)
                self.annual_vol[i, j] = stdev * math.sqrt(252.0)


class MarketData:
    def __init__(self, market_data, symbols):
        """
        market_data : MarketData instance (the jitclass)
        symbols     : list of str, e.g. ["BTCUSDT", "ETHUSDT", ...]
        """
        self.mds = market_data  # the jitclass instance
        self.symbols = symbols
        # create a dictionary symbol -> column index
        self.symbol_to_col = {sym: i for i, sym in enumerate(symbols)}


    def __repr__(self):
        """
        Return a string representation of ALL columns in a DataFrame
        with multi-level columns: top level = symbol, second level = field.
        The row index is taken from the first column in timestamp (assuming
        all columns share the same time dimension).
        """
        mds = self.mds  # shorthand
        rows, cols = mds.close.shape

        # We'll treat the first column of 'timestamp' as our index
        # (assuming all symbols have the same timestamps).
        if cols == 0 or rows == 0:
            # No data, return something minimal
            return "MarketDataWrapper: (empty)"

        time_index = mds.timestamp[:, 0]
        # If you want to convert it to an actual DatetimeIndex (in ms),
        # uncomment the line below:
        # time_index = pd.to_datetime(time_index, unit='ms')

        # Define which fields to show in the multi-level columns
        fields = ["open", "high", "low", "close", "volume",
                  "log_ret", "atr", "ann_vol"]

        # Build a MultiIndex for columns: [(symbol, "open"), (symbol, "high"), ...]
        tuples = []
        for sym in self.symbols:
            for f in fields:
                tuples.append((sym, f))
        col_index = pd.MultiIndex.from_tuples(tuples, names=["Symbol", "Field"])

        # Prepare a 2D NumPy array to hold all numeric data
        # shape = (rows, len(symbols) * len(fields))
        data_matrix = np.zeros((rows, len(col_index)), dtype=np.float64)

        # Fill data_matrix by column
        for sym_idx, sym in enumerate(self.symbols):
            c_open  = mds.open_[:,  sym_idx]
            c_high  = mds.high[:,  sym_idx]
            c_low   = mds.low[:,   sym_idx]
            c_close = mds.close[:, sym_idx]
            c_vol   = mds.volume[:, sym_idx]
            c_lr    = mds.log_returns[:, sym_idx]
            c_atr   = mds.atr[:, sym_idx]
            c_av    = mds.annual_vol[:, sym_idx]

            base = sym_idx * len(fields)
            data_matrix[:, base + 0] = c_open
            data_matrix[:, base + 1] = c_high
            data_matrix[:, base + 2] = c_low
            data_matrix[:, base + 3] = c_close
            data_matrix[:, base + 4] = c_vol
            data_matrix[:, base + 5] = c_lr
            data_matrix[:, base + 6] = c_atr
            data_matrix[:, base + 7] = c_av

        # Build the final DataFrame
        df = pd.DataFrame(data_matrix, columns=col_index, index=time_index)

        # Optionally name the index
        df.index.name = "open_time"

        # Return the DataFrame's string representation
        # so that printing the wrapper shows the multi-level columns.
        return df.__repr__()

    def __getitem__(self, symbol):
        """
        Return data for a specific symbol as a Pandas DataFrame.
        E.g. usage: df = wrapper['BTCUSDT']
        """
        col = self.symbol_to_col.get(symbol, None)
        if col is None:
            raise KeyError(f"Symbol '{symbol}' not found in MarketDataWrapper")

        # Extract columns from the jitclass arrays (using normal Python slicing)
        # The arrays themselves are still NumPy arrays.
        timestamps = self.mds.timestamp[:, col]
        opens      = self.mds.open_[:, col]
        highs      = self.mds.high[:, col]
        lows       = self.mds.low[:, col]
        closes     = self.mds.close[:, col]
        volumes    = self.mds.volume[:, col]
        logrets    = self.mds.log_returns[:, col]
        atrvals    = self.mds.atr[:, col]
        volvals    = self.mds.annual_vol[:, col]

        human_open_time = pd.to_datetime(timestamps, unit='ms')

        # Build a DataFrame (time index optional)
        df = pd.DataFrame({
            'open_time': timestamps,
            'human_open_time': human_open_time,
            'open':      opens,
            'high':      highs,
            'low':       lows,
            'close':     closes,
            'volume':    volumes,
            'log_ret':   logrets,
            'atr':       atrvals,
            'ann_vol':   volvals
        })
        # Optionally convert timestamp to DatetimeIndex if you'd like:
        # df['human open time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)

        return df

    def get_symbol_col(self, symbol):
        """
        Return the integer column index for the given symbol, or -1 if not found.
        """
        return self.symbol_to_col.get(symbol, -1)