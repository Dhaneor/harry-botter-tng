# cython: language_level=3
import cython
import logging
cimport numpy as np
import numpy as np
from math import cos, exp, pi

from .shared cimport MarketData
from analysis.statistics.cython_statistics cimport Statistics

logger = logging.getLogger("main.market_data_store")


# ............................... MarketState classes ..................................
cdef class MarketState:
    """A class to represent the state of the marlet at a particular point in time"""

    def __cinit__(self):
        self.timestamp = 0
        # Initialize with empty memoryviews
        self.open = self.high = self.low = self.close = self.volume = \
        self.vola_anno = self.sr_anno = self.atr = \
        np.empty(0, dtype=np.float64)

    cdef void update(
        self, 
        long long timestamp, 
        double[:] open, 
        double[:] high, 
        double[:] low,  
        double[:] close, 
        double[:] volume,
        double[:] vola_anno,
        double[:] sr_anno,
        double[:] atr
    ):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.vola_anno = vola_anno
        self.sr_anno = sr_anno
        self.atr = atr

    def test_update(
        self,
        timestamp: int,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        vola_anno: np.ndarray,
        sr_anno: np.ndarray,
        atr: np.ndarray
    ):
        """Method for testing the .update() method.
        
        The .update() method is intended to be used by other Cython 
        code,  but for tests, this wrapper must be used as .update() 
        is not accessible from Python code.
        """
        self.update(timestamp, open, high, low, close, volume, vola_anno, sr_anno, atr)


cdef class MarketStatePool:
    """A pool if readily available MarketState objects.
    
    As MarketState objects are required at every step in backtests,
    this pool prevents constant instantion and destruction of 
    MarketState objects and (hopefully) saves time / increases
    efficiency and speed.
    """

    def __cinit__(self, int size):
        self._pool = [MarketState() for _ in range(size)]
        self.size = size

    cdef MarketState get(self):
        return self._pool.pop() if self._pool else MarketState()

    cdef void release(self, MarketState state):
        if len(self._pool) < self.size:
            self._pool.append(state)

    @property
    def pool(self):
        return self._pool

    @property
    def size(self):
        return len(self._pool)

    # .................... Python wrapper methods for testing only .....................
    def _get(self) -> MarketState:
        return self.get()

    def _release(self, state: MarketState) -> None:
        self.release(state)


# .................................. MarketDataStore class .............................
cdef class MarketDataStore:
    """A class to hold market data.
    
    This class holds market data (OHLCV) and some related data 
    that is needed repeatedly, like:
    • annualized volatility
    • annualized Sharpe Ratio for each asset
    • ATR

    It also provides a method to get the market state in the form of
    MarketState objects for a given point in time (which is useful 
    for backtests).
    """

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
        # Check dimensionality
        if timestamp.ndim != 1 and timestamp.ndim != 2:
            raise ValueError("timestamp must be a 1- or 2-dimensional array")
        if open.ndim != 2 or high.ndim != 2 or low.ndim != 2 \
        or close.ndim != 2 or volume.ndim != 2:
            raise ValueError(
                "open, high, low, close, and volume must be 2-dimensional arrays"
            )

        # Check that all arrays have the same shape
        markets = close.shape[1]
        if (
            open.shape[1] != markets or high.shape[1] != markets 
            or low.shape[1] != markets or volume.shape[1] != markets
        ):
            raise ValueError(
                "All input arrays must have the same number of markets:\n"
                f"markets timestamp: {timestamp.shape[1]}\n"
                f"markets open: {open.shape[1]}\n"
                f"markets high: {high.shape[1]}\n"
                f"markets low: {low.shape[1]}\n"
                f"markets close: {close.shape[1]}\n"
                f"markets volume: {volume.shape[1]}\n"
                )

        periods = close.shape[0]
        if (
            open.shape[0] != periods or high.shape[0] != periods
            or low.shape[0] != periods or volume.shape[0] != periods
        ):
            raise ValueError(
                "All input arrays must have the same number of periods:\n"
                f"periods timestamp: {timestamp.shape[0]}\n"
                f"periods open: {open.shape[0]}\n"
                f"periods high: {high.shape[0]}\n"
                f"periods low: {low.shape[0]}\n"
                f"periods close: {close.shape[0]}\n"
                f"periods volume: {volume.shape[0]}\n"
                )

        # set attributes
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

        self.lookback = lookback     

        # initialize component classes
        self.stats = Statistics() 
        self._state_pool = MarketStatePool(5)

        rows, cols = close.shape[0], close.shape[1]

        # initialize empty arrays for all additional values
        self.atr = np.full_like(close, np.nan, dtype=np.float64)
        self.vola_anno = np.zeros((rows, cols), dtype=np.float64)
        self.sr_anno = np.ones((rows, cols), dtype=np.float64)
        self.signal_scale_factor = np.ones((rows, cols), dtype=np.float64)

        # compute the additional values
        self.compute_atr()
        self.compute_annualized_volatility()

        self.sr_anno = self.stats.annualized_sharpe_ratio(
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
        ts = self.timestamp[:, 0] if self.timestamp.ndim > 0 else self.timestamp
        timestamp_diffs = np.diff(ts)
        mask = timestamp_diffs != 0
        typical_diff = np.median(timestamp_diffs[mask])
        ms_per_year = 365 * 24 * 60 * 60 * 1000
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
        self.vola_anno = self.stats.annualized_volatility(
            self.close.astype(np.float64), 
            self.periods_per_year,
            self.lookback
        )

    cdef MarketState get_state(self, int index):
        state = self._state_pool.get()
        state.update(
            self.timestamp[index] if self.timestamp.ndim == 1 else self.timestamp[index, 0],
            self.open[index, :],
            self.high[index, :],
            self.low[index, :],
            self.close[index, :],
            self.volume[index, :],
            self.vola_anno[index, :],
            self.sr_anno[index, :],
            self.atr[index, :]
        )
        return state

    cdef void release_state(self, MarketState state):
        self._state_pool.release(state)

    cdef void compute_atr(self, int period=14):
        cdef int markets, periods
        cdef double tr, sum_tr
        cdef int m, p
    
        markets, periods = self.atr.shape[1], self.atr.shape[0]
    
        for m in range(markets):
            # Calculate initial ATR
            sum_tr = 0
            for p in range(1, period + 1):
                tr = max(
                    self.high[p, m] - self.low[p, m],
                    abs(self.high[p, m] - self.close[p-1, m]),
                    abs(self.low[p, m] - self.close[p-1, m])
                )
                sum_tr += tr
    
            # Set initial ATR value
            self.atr[period, m] = sum_tr / period
    
            # Calculate subsequent ATR values
            for p in range(period + 1, periods):
                tr = max(
                    self.high[p, m] - self.low[p, m],
                    abs(self.high[p, m] - self.close[p-1, m]),
                    abs(self.low[p, m] - self.close[p-1, m])
                )
                self.atr[p, m] = (self.atr[p-1, m] * (period - 1) + tr) / period

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

    # ................... Python accessible wrappers for testing .......................
    def _get_state(self, index: int) -> MarketState:
        return self.get_state(index)

    def _smooth_it(self, arr: np.ndarray, factor: int = 3) -> np.ndarray:
        return np.asarray(self.smooth_it(arr, factor))