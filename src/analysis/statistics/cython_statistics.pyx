# cython: language_level=3
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
"""
Provides a (Cython) Statistics class for often needed calculations. 

Created on Tue Feb 06 01:33:23 2025

@author dhaneor
"""

cimport numpy as cnp
import numpy as np

# Ensure NumPy is initialized
cnp.import_array()


cdef class Statistics:

    def __cinit__(self, long periods_per_year = 365, double risk_free_rate = 0.0):
        self._epsilon = 1e-8  # Small value to avoid division by zero
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    cdef _apply_to_columns(self, cnp.ndarray arr, func, int period) except *:
        cdef:
            Py_ssize_t shape_0 = arr.shape[0]
            Py_ssize_t shape_other = 1
            cnp.ndarray[double, ndim=2] arr_reshaped
            tuple original_shape
            int ndim = arr.ndim
            int num_cols = 1
        
        if ndim < 1:
            raise ValueError("Input array must have at least 1 dimension")

        # Calculate the total number of columns across all other dimensions
        for dim in range(1, ndim):
            shape_other *= arr.shape[dim]
        arr_reshaped = arr.reshape(shape_0, shape_other)
        original_shape = tuple([arr.shape[i] for i in range(ndim)])

        # apply rolling calculation if period parameter is specified
        if period > 0:
            for col in range(shape_other):  # Apply the function to each column
                arr_reshaped[:, col] = self._apply_rolling(
                    arr_reshaped[:, col], 
                    func, 
                    period
                )
            return arr_reshaped.reshape(original_shape)  # back to original shape

        # ... otherwise, apply over the whole length of first dimension
        else:  
            # create an empty array with the same shape as the input array,
            # except for first dimension, which now has a length of 1
            out = np.full((1, shape_other), np.nan, dtype=np.float64)
            for col in range(shape_other):
                out[0, col] = func(arr_reshaped[:, col])
            return out.reshape((1,) + original_shape[1:])

    cdef double[:] _apply_rolling(self, double[:] arr, func, int period):
        cdef:
            int i, n = arr.shape[0]
            double[:] result = np.full(n, np.nan)
        for i in range(period - 1, n):
            result[i] = func(arr[i - period + 1 : i + 1])
        return result

    cdef compute(self, cnp.ndarray arr, func, int period):
        return self._apply_to_columns(arr, func, period)

    # ................................. Simple Stats ...................................
    cpdef mean(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.mean, period)

    cpdef std(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.std, period)

    cpdef var(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.var, period)

    cpdef min(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.min, period)

    cpdef max(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.max, period)

    cpdef sum(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.sum, period)

    # ................................. Returns and Volatility .........................
    cpdef returns(self, cnp.ndarray arr, int period=0):
        return np.diff(arr) / arr[:-1]

    cpdef log_returns(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, self._log_returns_fn, period)

    cpdef volatility(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, np.std, period)

    cpdef sharpe_ratio(self, cnp.ndarray arr, int period=0):
        return self.compute(arr, self._sharpe_ratio_fn, period)

    # ......................... Annualized Returns and Volatility ......................
    cpdef annualized_returns(
        self, cnp.ndarray arr, int periods_per_year, int period=0
    ):
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_returns_fn, period)

    cpdef annualized_volatility(
        self, cnp.ndarray arr, int periods_per_year, int period=0
    ):
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_vol_fn, period)

    cpdef annualized_sharpe_ratio(
        self, cnp.ndarray arr, int periods_per_year, int period=0
    ):
        self.periods_per_year = periods_per_year
        return self.compute(arr, self._annualized_sharpe_ratio_fn, period)

    # ................................. Helper methods .................................
    cdef double[:] _returns_fn(self, double[:] arr):
        return np.diff(arr, n=1, axis=0)

    cdef double[:]  _log_returns_fn(self, double[:] arr):
        return np.log(np.diff(arr, axis=0))

    cdef double _sharpe_ratio_fn(self, double[:] arr):
        # Calculate periodic returns
        cdef int n = arr.shape[0]
        cdef int i = 0
        cdef double[:] returns = np.empty(n - 1, dtype=np.float64)
        
        for i in range(n - 1):
            returns[i] = (arr[i + 1] / arr[i]) - 1

        return np.mean(returns) / np.std(returns)

    cdef double _annualized_returns_fn(self, double[:] arr):
        total_return = arr[-1] / arr[0] - 1
        return (1 + total_return) ** (self.periods_per_year / len(arr)) - 1

    cdef double _annualized_vol_fn(self, double[:] arr):
        cdef double[:] ratios = np.empty(len(arr) - 1)
        cdef int i
        for i in range(len(arr) - 1):
            ratios[i] = arr[i+1] / arr[i]
        
        cdef double[:] log_returns = np.log(ratios)
        
        return np.std(log_returns) * np.sqrt(self.periods_per_year)

    cdef double _annualized_sharpe_ratio_fn(self, double[:] arr):
        cdef double annualized_volatility

        if arr.shape[0] < 2:
            return 0.0

        returns = np.subtract(np.divide(arr[1:], arr[:-1]), 1)
        rf_per_period = (1 + self.risk_free_rate) ** (1/self.periods_per_year) - 1
        returns = returns - rf_per_period

        annualized_return = np.mean(returns) * self.periods_per_year
        annualized_volatility = np.std(returns, ddof=1) * np.sqrt(self.periods_per_year)

        return (annualized_return / (annualized_volatility + self._epsilon))

    # ...............................ATR calculation .................................
    cdef true_range(self, cnp.ndarray high, cnp.ndarray low, cnp.ndarray close):
        def tr_func(h, l, c):
            return np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))

        return self.compute(np.stack((high, low, close), axis=-1), tr_func, 0)

    cpdef atr(
        self, cnp.ndarray high, cnp.ndarray low, cnp.ndarray close, 
        int period = 14
    ):
        tr = self.true_range(high, low, close)
        return self.mean(tr, period)
