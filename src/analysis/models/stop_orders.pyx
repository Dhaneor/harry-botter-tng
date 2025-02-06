# cython: language_level=3
# distutils: language = c++

# Add this line to use the newer NumPy API
# cython: numpy_api=2

cdef class StopOrder:
    cdef double apply(self, double price, double high, double low):
        raise NotImplementedError()

cdef class FixedStopOrder(StopOrder):
    cdef double apply(self, double price, double high, double low):
        # Implementation here
        pass

cdef class TrailingStopOrder(StopOrder):
    cdef double apply(self, double price, double high, double low):
        # Implementation here
        pass