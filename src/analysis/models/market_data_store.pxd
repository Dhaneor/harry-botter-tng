cimport numpy as cnp
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

    cdef compute_annualized_volatility(self)
    cdef void compute_atr(self, int period=*)
    cdef double[:,:] smooth_it(self, double[:,:] arr, int factor=*)
    cdef double[:] _apply_smoothing_1D(self, double[:] data, int length)