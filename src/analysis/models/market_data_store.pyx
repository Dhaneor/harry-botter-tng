# cython: language_level=3
cimport numpy as cnp
import numpy as np
from math import cos, exp, pi

from .shared cimport MarketData
from analysis.statistics.cython_statistics cimport Statistics

cdef class MarketDataStore:
    cdef:
        public cnp.ndarray timestamp, open, high, low, close, volume
        public cnp.ndarray annual_vol
        public cnp.ndarray annual_sr
        public cnp.ndarray signal_scale_factor
        public cnp.ndarray atr, volatility
        public int lookback, num_assets, num_periods
        public Statistics stats

    def __cinit__(
        self,
        cnp.ndarray timestamp,
        cnp.ndarray open,
        cnp.ndarray high,
        cnp.ndarray low,
        cnp.ndarray close,
        cnp.ndarray volume,
        int lookback = 20
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.lookback = lookback     

        self.stats = Statistics() 

        rows, cols = close.shape[0], close.shape[1]

        self.atr = np.full_like(close, np.nan)
        self.annual_vol = np.zeros((rows, cols), dtype=np.float64)
        self.annual_sr = np.ones((rows, cols), dtype=np.float64)
        self.signal_scale_factor = np.ones((rows, cols), dtype=np.float64)

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

        self.signal_scale_factor = np.asarray(
            self.smooth_it(scale_factor, 40)
        )

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
        """Return the number of periods in the data."""
        return self.timestamp.shape[0]

    @property
    def symbols(self) -> int:
        """ Return the number of symbols in the data."""
        return self.close.shape[1]

    # ..................................................................................
    cdef compute_annualized_volatility(self):
        self.annual_vol = self.stats.annualized_volatility(
            self.close.astype(np.float64), 
            self.periods_per_year,
            self.lookback
        )

    cdef void compute_atr(self, int period=14):
        cdef int markets
        cdef int periods

        markets, periods = self.atr.shape[0], self.atr.shape[1]

        for m in range(markets):
            for p in range(period, periods):
                tr = max(
                    self.high[m, p] - self.low[m, p],
                    abs(self.high[m, p] - self.open[m, p]),
                    abs(self.low[m, p] - self.open[m, p])
                )
                self.atr[m, p] = ((self.atr[m, p - 1] * (period - 1) + tr) / period)

    cdef double[:,:] smooth_it(self, double[:,:] arr, int factor = 3):
        for market in range(arr.shape[1]):
            self._apply_smoothing_1D(arr[:, market], factor)
        return arr

    # ..................................................................................
    cdef double[:] _apply_smoothing_1D(self, double[:] data, int length):
        """
        Calculate the Ehlers Ultimate Smoother for a given data series.

        Parameters:
        - data (array-like): The input data series (e.g., prices).
        - length (int): The smoothing period.

        Returns:
        - us (np.ndarray): The smoothed data series.
        """
        cdef int n = len(data)
        cdef double[:] us = np.zeros(n, dtype=np.float64)
        
        # Check if data length is sufficient
        if n < 3:
            raise ValueError("Data array must have at least 3 elements.")
        
        # Initialize the smoothed series with zeros
        us = np.zeros(n)

        cdef cnp.float64_t f = (1.414 * pi) / length
        a1 = exp(-f)
        c2 = 2 * a1 * cos(f)
        c3 = -a1 ** 2
        c1 = (1 + c2 - c3) / 4
        
        # Initialization:
        us[0] = data[0]
        us[1] = data[1]

        cdef int t
        
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