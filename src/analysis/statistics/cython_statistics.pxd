# cython: language_level=3
cimport numpy as cnp

cdef class Statistics:
    cdef:
        double _epsilon
        long periods_per_year
        double risk_free_rate

    cdef _apply_to_columns(self, cnp.ndarray arr, func, int period) except *
    cdef double[:] _apply_rolling(self, double[:] arr, func, int period)
    cdef compute(self, cnp.ndarray arr, func, int period)

    cpdef mean(self, cnp.ndarray arr, int period=*)
    cpdef std(self, cnp.ndarray arr, int period=*)
    cpdef var(self, cnp.ndarray arr, int period=*)
    cpdef min(self, cnp.ndarray arr, int period=*)
    cpdef max(self, cnp.ndarray arr, int period=*)
    cpdef sum(self, cnp.ndarray arr, int period=*)

    cpdef returns(self, cnp.ndarray arr, int period=*)
    cpdef log_returns(self, cnp.ndarray arr, int period=*)
    cpdef volatility(self, cnp.ndarray arr, int period=*)
    cpdef sharpe_ratio(self, cnp.ndarray arr, int period=*)

    cpdef annualized_returns(self, cnp.ndarray arr, int periods_per_year, int period=*)
    cpdef annualized_volatility(self, cnp.ndarray arr, int periods_per_year, int period=*)
    cpdef annualized_sharpe_ratio(self, cnp.ndarray arr, int periods_per_year, int period=*)

    cdef double[:] _returns_fn(self, double[:] arr)
    cdef double[:] _log_returns_fn(self, double[:] arr)
    cdef double _sharpe_ratio_fn(self, double[:] arr)
    cdef double _annualized_returns_fn(self, double[:] arr)
    cdef double _annualized_vol_fn(self, double[:] arr)
    cdef double _annualized_sharpe_ratio_fn(self, double[:] arr)
    
    cdef true_range(self, cnp.ndarray high, cnp.ndarray low, cnp.ndarray close)
    cpdef atr(self, cnp.ndarray high, cnp.ndarray low, cnp.ndarray close, int period=*)
