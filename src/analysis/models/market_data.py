#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 06 01:28:53 2024

@author: dhaneor
"""

import json
import numpy as np
import numpy.typing as npt
import pandas as pd
from datetime import datetime, timedelta
from numba import int64, float32
from numba.experimental import jitclass
from typing import Sequence

from analysis.statistics.statistics import Statistics


spec = [
    ("timestamp", int64[:, :]),
    ("open_", float32[:, :]),
    ("high", float32[:, :]),
    ("low", float32[:, :]),
    ("close", float32[:, :]),
    ("volume", float32[:, :]),
    # ("stats", Statistics.class_type.instance_type),
    ("log_returns", float32[:, :]),
    ("atr", float32[:, :]),
    ("annual_vol", float32[:, :]),
]


@jitclass(spec)
class MarketDataStore:
    """A class to store raw market data for multiple symbols as 2d arrays."""

    def __init__(
        self,
        timestamp: npt.ArrayLike,
        open_: npt.ArrayLike,
        high: npt.ArrayLike,
        low: npt.ArrayLike,
        close: npt.ArrayLike,
        volume: npt.ArrayLike,
    ):
        self.timestamp = timestamp
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        # self.stats = Statistics()
        self.interval_ms = self._calculate_interval(timestamp)

        rows, cols = close.shape

        # Initialize indicator arrays to zeros or empty
        self.log_returns = np.zeros((rows, cols), dtype=np.float32)
        self.atr = np.zeros((rows, cols), dtype=np.float32)
        self.annual_vol = np.zeros((rows, cols), dtype=np.float32)

        self.compute_log_returns()
        self.compute_atr()
        self.compute_annual_vol()

    @property
    def periods_per_year(self) -> int:
        timestamp_diffs = np.diff(self.timestamp, axis=0)
        typical_diff = np.min(timestamp_diffs, axis=0)
        ms_per_year = 356 * 24 * 60 * 60 * 1000  # milliseconds per year
        return int(ms_per_year / typical_diff)

    def get_no_of_periods(self) -> int:
        """
        Return the number of periods in the data.
        """
        return self.timestamp.shape[0]

    def get_no_of_symbols(self) -> int:
        """
        Return the number of symbols in the data.
        """
        return self.close.shape[1]

    def _calculate_interval(self, timestamp):
        return int(np.median(np.diff(timestamp)))

    def compute_log_returns(self):
        self.log_returns = self.stats.log_returns(self.close)

    def compute_annualized_returns(self):
        self.annualized_returns = self.stats.annualized_returns(self.close, self.interval_ms)

    def compute_annualized_volatility(self):
        self.annualized_volatility = self.stats.annualized_volatility(self.close, self.interval_ms)

    def compute_sharpe_ratio(self, risk_free_rate=0.0):
        returns = self.stats.pct_change(self.close)
        self.sharpe_ratio = self.stats.sharpe_ratio(returns, risk_free_rate, self.interval_ms)

    def compute_atr(self, period=14):
        self.atr = self.stats.atr(self.high, self.low, self.close, period)

class MarketData:
    """A class to hold OHLCV data for multiple symbols.

    NOTE: This class assumes that the provided data is for
    the same interval for all symbols.
    """

    def __init__(self, market_data_store: MarketDataStore, symbols: Sequence[str]):
        """
        market_data : MarketData instance (the jitclass)
        symbols     : list of str, e.g. ["BTCUSDT", "ETHUSDT", ...]
        """
        self.mds = market_data_store
        self.symbols = symbols

        # create a dictionary symbol -> column index
        self.symbol_to_col = {sym: i for i, sym in enumerate(symbols)}

        # We define the fields we have in the MarketData
        # The array names in your jitclass might differ
        self.available_fields = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "log_ret",
            "atr",
            "ann_vol",
        ]

        self._interval: str = None

        timestamps = self.mds.timestamp[:, 0]
        self._interval_in_ms = int(np.min(np.diff(timestamps[~np.isnan(timestamps)])))

    def __repr__(self):
        """
        Return a string representation of ALL columns in a DataFrame
        with multi-level columns: top level = symbol, second level = field.
        The row index is taken from the first column in timestamp (assuming
        all columns share the same time dimension).
        """
        return self.dataframe.__repr__()

    def __len__(self):
        """
        Return the number of rows in the MarketData.
        """
        return len(self.mds.close)

    def __getitem__(self, key) -> pd.DataFrame:
        """
        Return a DataFrame by:
          - Symbol (if `key` is in self.symbols),
          - Field  (if `key` is in self.available_fields).

        Examples:
          wrapper['BTCUSDT'] -> all fields for that symbol as single-level columns
          wrapper['open']    -> all symbols for that field as a multi-level column

        If `key` is in both sets, you may need a tie-break rule.
        If `key` is in neither, raise KeyError.
        """
        if key in self.symbol_to_col:
            # User requested a SYMBOL
            return self._build_symbol_df(key)
        elif key in self.available_fields:
            # User requested a FIELD
            return self._build_field_df(key)
        else:
            raise KeyError(f"'{key}' not found in symbols or fields.")

    # .........................PROPERTIES FOR EASY DATA ACCESS .................................
    @property
    def dataframe(self):
        """Returns the data as a DataFrame.

        Returns:
            pd.DataFrame: Returns the data as a DataFrame,
            with multi-level columns as symbols/prices and
            rows as timestamps.
        """
        mds = self.mds  # shorthand
        rows, cols = mds.close.shape

        # We'll treat the first column of 'timestamp' as our index
        # (assuming all symbols have the same timestamps).
        if cols == 0 or rows == 0:
            # No data, return something minimal
            return "MarketDataWrapper: (empty)"

        time_index = pd.to_datetime(mds.timestamp[:, 0], unit="ms")

        # Define which fields to show in the multi-level columns
        fields = ["open", "high", "low", "close", "volume", "log_ret", "atr", "ann_vol"]

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
            c_open = mds.open_[:, sym_idx]
            c_high = mds.high[:, sym_idx]
            c_low = mds.low[:, sym_idx]
            c_close = mds.close[:, sym_idx]
            c_vol = mds.volume[:, sym_idx]
            c_lr = mds.log_returns[:, sym_idx]
            c_atr = mds.atr[:, sym_idx]
            c_av = mds.annual_vol[:, sym_idx]

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
        df.index.name = None

        return df

    @property
    def open(self) -> np.ndarray:
        """Returnsa 2D NumPy array of 'open' prices.

        Returns:
            np.ndarray: 2D array of open prices for all symbols.
        """
        return self.mds.open_

    @property
    def high(self) -> np.ndarray:
        """Returnsa 2D NumPy array of 'high' prices.

        Returns:
            np.ndarray: 2D array of high prices for all symbols.
        """
        return self.mds.high

    @property
    def low(self) -> np.ndarray:
        """Returnsa 2D NumPy array of 'low' prices.

        Returns:
            np.ndarray: 2D array of low prices for all symbols.
        """
        return self.mds.low

    @property
    def close(self) -> np.ndarray:
        """Returnsa 2D NumPy array of 'close' prices.

        Returns:
            np.ndarray: 2D array of close prices for all symbols.
        """
        return self.mds.close

    @property
    def volume(self) -> np.ndarray:
        """Returnsa 2D NumPy array of 'volumne' data.

        Returns:
            np.ndarray: 2D array of volume data for all symbols.
        """
        return self.mds.volume

    def interval(self) -> str:
        return self._interval

    @property
    def interval_in_ms(self) -> int:
        """Return the interval in milliseconds.

        Returns:
            int: interval in milliseconds.
        """
        return self._interval_in_ms

    @property
    def number_of_assets(self) -> int:
        """Return the number of assets in the MarketData.

        Returns:
            int: number of assets.
        """
        return len(self.symbols)

    # ................................ METHODS TO RETRIEVE DATA ...........................
    def get_array(self, field, symbol=None):
        """Returns a Numpy array of the specified field for the given symbol.

        Parameters:
        -----------
        field: str -> e.g. "close"
        symbol: str -> optional symbol to slice
        if None, return the entire 2D array
        if provided, return a 1D slice

        Returns:
        --------
        np.ndarray: 1D or 2D array of specified field for the given symbol.
        """
        
        arr_2d = None
        match field:
            case "close":
                arr_2d = self.mds.close
            case "open":
                arr_2d = self.mds.open_
            case "high":
                arr_2d = self.mds.high
            case "low":
                arr_2d = self.mds.low
            case "volume":
                arr_2d = self.mds.volume
            case _:
                raise KeyError(f"Unknown field: {field}")

        if symbol is not None:
            col = self.symbol_to_col.get(symbol, -1)
            if col == -1:
                raise KeyError(f"Unknown symbol: {symbol}")
            return arr_2d[:, col]  # shape (rows,)
        return arr_2d  # shape (rows, cols)

    def to_dictionary(self):
        """Returns a dictionary of the MarketData object.

        Returns:
        --------
        dict[str, np.ndarray]
            A dictionary with keys 'open time', 'open', 'high', 'low',
            'close', and 'volume', containing the corresponding numpy
            arrays (1D).
        """
        return {
            "open time": self.mds.timestamp[:, 0].reshape(-1),
            "open": self.mds.open_[:, 0].reshape(-1),
            "high": self.mds.high[:, 0].reshape(-1),
            "low": self.mds.low[:, 0].reshape(-1),
            "close": self.mds.close[:, 0].reshape(-1),
            "volume": self.mds.volume[:, 0].reshape(-1),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the data as a dataframe."""
        return self.dataframe

    def to_json(self):
        """Returns the data as a JSON string."""
        return json.dumps(self.to_dictionary(), indent=4)

    # .............. CLASS METHODS TO BUILD AN INSTANCE IN DIFFERENT WAYS .............
    @classmethod
    def from_dictionary(cls, symbol, data: dict):
        """Builds a new MarketData object from an OHLCV dictionary."""
        mds = MarketDataStore(
            open_=data["open"].reshape(-1, 1).astype(np.float32),
            high=data["high"].reshape(-1, 1).astype(np.float32),
            low=data["low"].reshape(-1, 1).astype(np.float32),
            close=data["close"].reshape(-1, 1).astype(np.float32),
            volume=data["volume"].reshape(-1, 1).astype(np.float32),
            timestamp=data["open time"].reshape(-1, 1).astype(np.int64),
        )

        return MarketData(mds, [symbol])

    @classmethod
    def from_dataframe(cls, symbol, df: pd.DataFrame):
        """Builds a new MarketData object from a DataFrame."""

        # Create MarketDataStore
        mds = MarketDataStore(
            open_=df["open"].values.astype(np.float32),
            high=df["high"].values.astype(np.float32),
            low=df["low"].values.astype(np.float32),
            close=df["close"].values.astype(np.float32),
            volume=df["volume"].values.astype(np.float32),
            timestamp=df["open time"].values.astype(np.int64),
        )

        return MarketData(mds, [symbol])

    @classmethod
    def from_random(cls, length: int, no_of_symbols: int, volatility: float = 0.005):
        vol = volatility

        # Generate end timestamp (current date at 00:00:00)
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(hours=length / 4)
    
        # Generate timestamps
        timestamps = pd.date_range(start=start_date, end=end_date, periods=length)
        timestamps_ms = timestamps.astype(np.int64) // 10**6
        timestamps_ms = timestamps_ms.to_numpy()
    
        # Generate random symbols
        symbols = [
            "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3)) + "USDT"
            for _ in range(no_of_symbols)
        ]
    
        # Initialize arrays
        open_prices = np.zeros((length, no_of_symbols), dtype=np.float32)
        high_prices = np.zeros((length, no_of_symbols), dtype=np.float32)
        low_prices = np.zeros((length, no_of_symbols), dtype=np.float32)
        close_prices = np.zeros((length, no_of_symbols), dtype=np.float32)
        volumes = np.zeros((length, no_of_symbols), dtype=np.float32)
        timestamps = np.tile(timestamps_ms.reshape(-1, 1), (1, no_of_symbols))
    
        for i in range(no_of_symbols):
            # Generate initial price (between 1 and 1000)
            initial_price = np.random.uniform(1, 1000)
    
            # Generate price changes using random walk
            changes = np.random.normal(0, vol, length)  # 2% daily volatility
    
            # Calculate prices
            prices = initial_price * np.exp(np.cumsum(changes))
    
            # Generate open, high, low, close
            close_prices[:, i] = prices
            open_prices[0, i] = initial_price  # First open price
            open_prices[1:, i] = close_prices[:-1, i]  # Subsequent open prices
    
            # Generate intraday price movements
            intraday_high = np.random.uniform(0, vol, length)  # Up to 1% higher
            intraday_low = np.random.uniform(0, vol, length)   # Up to 1% lower
    
            high_prices[:, i] = np.maximum(open_prices[:, i], close_prices[:, i]) * (1 + intraday_high)
            low_prices[:, i] = np.minimum(open_prices[:, i], close_prices[:, i]) * (1 - intraday_low)
    
            # Ensure high is always highest and low is always lowest
            high_prices[:, i] = np.maximum(high_prices[:, i], np.maximum(open_prices[:, i], close_prices[:, i]))
            low_prices[:, i] = np.minimum(low_prices[:, i], np.minimum(open_prices[:, i], close_prices[:, i]))
    
            # Generate volumes (between 1000 and 100000)
            volumes[:, i] = np.random.uniform(1000, 100000, length)
    
            timestamps[:, i] = timestamps_ms
    
        # Create MarketDataStore
        mds = MarketDataStore(
            open_=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volumes,
            timestamp=timestamps,
        )
    
        return cls(mds, symbols)

    # .................................................................................
    def _build_symbol_df(self, symbol):
        """
        Return a DataFrame with columns = all fields, rows = time,
        for the given symbol.
        """
        col = self.symbol_to_col[symbol]

        # For convenience, define a small dictionary to map field_name -> actual array
        mds = self.mds
        field_arrays = {
            "open": mds.open_[:, col],
            "high": mds.high[:, col],
            "low": mds.low[:, col],
            "close": mds.close[:, col],
            "volume": mds.volume[:, col],
            "log_ret": mds.log_returns[:, col],
            "atr": mds.atr[:, col],
            "ann_vol": mds.annual_vol[:, col],
        }

        time_index = mds.timestamp[:, col]  # or mds.timestamp[:, 0] if aligned
        df = pd.DataFrame(
            {f: field_arrays[f] for f in self.available_fields}, index=time_index
        )
        df.index.name = None
        return df

    def _build_field_df(self, field):
        """
        Return a DataFrame with columns = all symbols, rows = time,
        but only for the specified field.

        We'll use a multi-level column index with (symbol, field).
        If you want single-level with just the symbol,
        you could do something else.
        """
        mds = self.mds
        rows, cols = mds.close.shape

        # Prepare a 2D array for all symbols
        data_matrix = np.zeros((rows, len(self.symbols)), dtype=np.float64)

        # Map the user-friendly name to the actual array in the jitclass
        # Notice that we have `mds.open_`, `mds.close`, etc. are separate arrays
        # if field == "open":
        #     arr = mds.open_
        # elif field == "high":
        #     arr = mds.high
        # elif field == "low":
        #     arr = mds.low
        # elif field == "close":
        #     arr = mds.close
        # elif field == "volume":
        #     arr = mds.volume
        # elif field == "log_ret":
        #     arr = mds.log_returns
        # elif field == "atr":
        #     arr = mds.atr
        # elif field == "ann_vol":
        #     arr = mds.annual_vol
        # else:
        #     raise KeyError(f"Unsupported field '{field}'")

        field_arrays = {
            "open": mds.open_,
            "high": mds.high,
            "low": mds.low,
            "close": mds.close,
            "volume": mds.volume,
            "log_ret": mds.log_returns,
            "atr": mds.atr,
            "ann_vol": mds.annual_vol,
        }

        arr = field_arrays.get(field, None)

        if arr is None:
            raise KeyError(f"Unsupported field '{field}'")

        for sym_idx, sym in enumerate(self.symbols):
            data_matrix[:, sym_idx] = arr[:, sym_idx]

        # Build a multi-level column index: (symbol, field)
        col_tuples = [(sym, field) for sym in self.symbols]
        col_index = pd.MultiIndex.from_tuples(col_tuples, names=["Symbol", ""])

        # Use, e.g., the first column of timestamp as a universal index
        time_index = pd.to_datetime(mds.timestamp[:, 0], unit="ms")
        df = pd.DataFrame(data_matrix, columns=col_index, index=time_index)
        df.index.name = None
        return df
