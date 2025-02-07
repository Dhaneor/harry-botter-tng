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
from math import exp, cos, pi
from numba import int64, float64
from numba.experimental import jitclass
from typing import Sequence

from .market_data_store import MarketDataStore
from analysis.statistics.statistics import Statistics
from analysis.chart.plot_definition import SubPlot, Candlestick, Line
from misc.mixins import PlottingMixin


spec = [
    ("timestamp", int64[:, :]),
    ("open_", float64[:, :]),
    ("high", float64[:, :]),
    ("low", float64[:, :]),
    ("close", float64[:, :]),
    ("volume", float64[:, :]),
    ("stats", Statistics.class_type.instance_type),
    ("interval_ms", int64),
    ("lookback", int64),
    ("atr", float64[:, :]),
    ("signal_scale_factor", float64[:, :]),
    ("annual_vol", float64[:, :]),
    ("annual_sr", float64[:, :]),
]

@jitclass(spec)
class MarketDataStoreJIT:
    """A class to store raw market data for multiple symbols as 2d arrays."""

    def __init__(
        self,
        timestamp: npt.ArrayLike,
        open_: npt.ArrayLike,
        high: npt.ArrayLike,
        low: npt.ArrayLike,
        close: npt.ArrayLike,
        volume: npt.ArrayLike,
        lookback: int = 20
    ):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.stats = Statistics()
        # self.interval_ms = self._calculate_interval(timestamp)

        self.lookback = lookback

        rows, cols = close.shape

        self.atr = np.full_like(close, np.nan)
        self.annual_vol = np.zeros((rows, cols), dtype=np.float64)
        self.annual_sr = np.ones((rows, cols), dtype=np.float64)
        self.signal_scale_factor = np.ones(close.shape, dtype=np.float64)

        self.compute_atr()
        self.compute_annualized_volatility()

        self.annual_sr = self.stats.annualized_sharpe_ratio(
            self.close.astype(np.float64),
            self.periods_per_year,
            self.lookback
        )

        # scale_factor = np.abs(
        #     self.stats.sharpe_ratio(
        #         self.close.astype(np.float64), 
        #         self.lookback
        #     )
        # )

        scale_factor = self.stats.sharpe_ratio(
            self.close.astype(np.float64), 
            self.lookback
        ) + 1

        # scale_factor = np.abs(
        #     self.stats.annualized_sharpe_ratio(
        #         self.close.astype(np.float64),
        #         self.periods_per_year,
        #         self.lookback
        #     )
        # )

        # scale_factor = 1 + self.stats.annualized_volatility(
        #     self.close.astype(np.float64),
        #     self.periods_per_year, 
        #     self.lookback
        # )

        self.signal_scale_factor = self.smooth_it(scale_factor, 40)

    # ..................................................................................
    @property
    def periods_per_year(self) -> int:
        ts = self.timestamp.astype(np.float64).reshape(-1,)
        timestamp_diffs = np.diff(ts)
        mask = timestamp_diffs != 0
        typical_diff = np.median(timestamp_diffs[mask])
        ms_per_year = 356 * 24 * 60 * 60 * 1000
        return int(ms_per_year / typical_diff)

    @property
    def periods(self) -> int:
        """
        Return the number of periods in the data.
        """
        return self.timestamp.shape[0]

    @property
    def symbols(self) -> int:
        """
        Return the number of symbols in the data.
        """
        return self.close.shape[1]

    # ..................................................................................
    def compute_annualized_volatility(self):
        self.annual_vol = self.stats.annualized_volatility(
            self.close.astype(np.float64), 
            self.periods_per_year,
            self.lookback
        )

    def compute_atr(self, period=14):
        markets, periods = self.atr.shape

        for m in range(markets):
            for p in range(period, periods):
                tr = max(
                    self.high[m, p] - self.low[m, p],
                    abs(self.high[m, p] - self.open[m, p]),
                    abs(self.low[m, p] - self.open[m, p])
                )
                self.atr[m, p] = ((self.atr[m, p - 1] * (period - 1) + tr) / period)

    def smooth_it(self, arr: np.ndarray, factor: int = 3):
        for market in range(arr.shape[1]):
            self._apply_smoothing_1D(arr[:, market], factor)
        return arr

    # ..................................................................................
    def _apply_smoothing_1D(self, data: np.ndarray, length: int) -> np.ndarray:
        """
        Calculate the Ehlers Ultimate Smoother for a given data series.

        Parameters:
        - data (array-like): The input data series (e.g., prices).
        - length (int): The smoothing period.

        Returns:
        - us (np.ndarray): The smoothed data series.
        """
        n: np.int64 = len(data)
        us: np.ndarray = np.zeros(n, dtype=np.float64)
        
        # Check if data length is sufficient
        if n < 3:
            raise ValueError("Data array must have at least 3 elements.")
        
        # Initialize the smoothed series with zeros
        us = np.zeros(n)

        f: np.float64 = (1.414 * pi) / length
        a1 = exp(-f)
        c2 = 2 * a1 * cos(f)
        c3 = -a1 ** 2
        c1 = (1 + c2 - c3) / 4
        
        # Initialization:
        us[0] = data[0]
        us[1] = data[1]
        
        # Iterate through the data starting from index 2
        for t in range(2, n):
            us[t] = (
                (1 - c1) * data[t] +
                (2 * c1 - c2) * data[t - 1] +
                (-c1 - c3) * data[t - 2] +
                c2 * us[t - 1] +
                c3 * us[t - 2]
            )
        
        return us


class MarketData(PlottingMixin):
    """A class to hold OHLCV data for multiple symbols.

    This is a wrapper class for MarketDataStore which holds the actual 
    data and is a cythonized class that can be used be other cythonized
    code.

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

        self.display_name = f"Market Data ({self.mds.lookback})"

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
        fields = ["open", "high", "low", "close", "volume", "atr", "ann_vol", "ann_sr", "signal_scale"]

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
            c_open = mds.open[:, sym_idx]
            c_high = mds.high[:, sym_idx]
            c_low = mds.low[:, sym_idx]
            c_close = mds.close[:, sym_idx]
            c_vol = mds.volume[:, sym_idx]
            c_atr = mds.atr[:, sym_idx]
            c_av = mds.annual_vol[:, sym_idx]
            c_asr = mds.annual_sr[:, sym_idx]
            c_scl = mds.signal_scale_factor[:, sym_idx]

            base = sym_idx * len(fields)
            data_matrix[:, base + 0] = c_open
            data_matrix[:, base + 1] = c_high
            data_matrix[:, base + 2] = c_low
            data_matrix[:, base + 3] = c_close
            data_matrix[:, base + 4] = c_vol
            data_matrix[:, base + 5] = c_atr
            data_matrix[:, base + 6] = c_av
            data_matrix[:, base + 7] = c_asr
            data_matrix[:, base + 8] = c_scl

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
        return self.mds.open

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

    @property
    def plot_data(self):
        return self[self.symbols[0]]

    @property
    def subplots(self):
        subplots: list[SubPlot] = [
            SubPlot(
                label="OHLCV",
                is_subplot=False,
                elements=(Candlestick(),),
                level="operand",
            ),
            SubPlot(
                label='scaling factor',
                is_subplot=True,
                elements=(
                    Line(
                        label='scaling factor',
                        column='signal_scale_factor',
                    ),
                )
            ),
            SubPlot(
                label='annualized SR',
                is_subplot=True,
                elements=(
                    Line(
                        label='annualized sharpe ratio',
                        column='ann_sr',
                    ),
                )
            ),
            SubPlot(
                label='annualized VOLA',
                is_subplot=True,
                elements=(
                    Line(
                        label='annualized volatility (stdev)',
                        column='ann_vol',
                    ),
                )
            ),
        ]
        return subplots

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
                arr_2d = self.mds.open
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
            "open": self.mds.open[:, 0].reshape(-1),
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
        start_date = end_date - timedelta(days=length)
    
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
        open_prices = np.zeros((length, no_of_symbols), dtype=np.float64)
        high_prices = np.zeros((length, no_of_symbols), dtype=np.float64)
        low_prices = np.zeros((length, no_of_symbols), dtype=np.float64)
        close_prices = np.zeros((length, no_of_symbols), dtype=np.float64)
        volumes = np.zeros((length, no_of_symbols), dtype=np.float64)
        timestamps = np.tile(timestamps_ms.reshape(-1, 1), (1, no_of_symbols))
    
        for i in range(no_of_symbols):
            # Generate initial price (between 1 and 1000)
            initial_price = np.random.uniform(1, 1000)
    
            # Generate price changes using random walk
            changes = np.random.normal(0, vol, length)
    
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
            timestamp=timestamps,
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            volume=volumes,
        )
    
        return cls(mds, symbols)

    # .................................................................................
    def _build_symbol_df(self, symbol):
        """
        Return a DataFrame with columns = all fields, rows = time,
        for the given symbol.
        """
        col = self.symbol_to_col[symbol]

        ["open", "high", "low", "close", "volume", "atr", "ann_vol", "ann_sr", "signal_scale"]

        # For convenience, define a small dictionary to map field_name -> actual array
        mds = self.mds
        field_arrays = {
            "open": mds.open[:, col],
            "high": mds.high[:, col],
            "low": mds.low[:, col],
            "close": mds.close[:, col],
            "volume": mds.volume[:, col],
            "atr": mds.atr[:, col],
            "ann_vol": mds.annual_vol[:, col],
            "ann_sr": mds.annual_sr[:, col],
            "signal_scale_factor": mds.signal_scale_factor[:, col],
        }

        time_index = mds.timestamp[:, col]  # or mds.timestamp[:, 0] if aligned
        df = pd.DataFrame(
            field_arrays,
            # {f: field_arrays[f] for f in self.available_fields}, 
            index=time_index
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

        field_arrays = {
            "open": mds.open,
            "high": mds.high,
            "low": mds.low,
            "close": mds.close,
            "volume": mds.volume,
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

    

