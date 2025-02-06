# cython: language_level=3
cimport numpy as np
import numpy as np
from math import cos, exp, pi

cdef class MarketDataStore:
    cdef:
        public np.ndarray open, high, low, close, volume
        public np.ndarray atr, volatility
        public int num_assets, num_periods

    def __cinit__(
        self,
        np.ndarray timestamp,
        np.ndarray open,
        np.ndarray high,
        np.ndarray low,
        np.ndarray close,
        np.ndarray volume,
        double lookback = 20
    ):
        self.timestamp = timestamp
        self.open_ = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.lookback = lookback      

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
    cdef compute_annualized_volatility(self):
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
                    abs(self.high[m, p] - self.open_[m, p]),
                    abs(self.low[m, p] - self.open_[m, p])
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