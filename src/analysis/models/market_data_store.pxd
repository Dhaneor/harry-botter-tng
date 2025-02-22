# cython: language_level=3
cimport numpy as cnp
from ..statistics.cython_statistics cimport Statistics


cdef class MarketState:
    cdef:
        public long long timestamp
        public double[:] open, high, low, close, volume, vola_anno, sr_anno, atr

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
        double[:] atr,
    )


cdef class MarketStatePool:
    cdef:
        list[MarketState] _pool
        int size

    cdef MarketState get(self)
    cdef void release(self, MarketState state)


cdef class MarketDataStore:
    cdef:
        public cnp.ndarray timestamp, open, high, low, close, volume
        public cnp.ndarray vola_anno
        public cnp.ndarray sr_anno
        public cnp.ndarray signal_scale_factor
        public cnp.ndarray atr, volatility
        public int lookback, num_assets, num_periods
        Statistics stats
        MarketStatePool _state_pool

    cdef compute_annualized_volatility(self)
    cdef MarketState get_state(self, int index)
    cdef void release_state(self, MarketState state)
    cdef void compute_atr(self, int period=*)
    cdef double[:,:] smooth_it(self, double[:,:] arr, int factor=*)
    cdef double[:] _apply_smoothing_1D(self, double[:] data, int length)