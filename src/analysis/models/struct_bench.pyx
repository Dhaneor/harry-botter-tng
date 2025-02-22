# cython: language_level=3
cimport numpy as cnp
import numpy as np
from libc.stdlib cimport malloc, free, rand

cdef struct CArray2D:
    double **data
    int rows, cols

cdef class CStructArray:
    cdef CArray2D arr
    
    def __cinit__(self, int rows, int cols):
        self.arr.rows = rows
        self.arr.cols = cols
        self.arr.data = <double**> malloc(rows * sizeof(double*))
        for i in range(rows):
            self.arr.data[i] = <double*> malloc(cols * sizeof(double))

    def __dealloc__(self):
        for i in range(self.arr.rows):
            free(self.arr.data[i])
        free(self.arr.data)

    cpdef void initialize(self):
        cdef int i, j
        for i in range(self.arr.rows):
            for j in range(self.arr.cols):
                self.arr.data[i][j] = <double> rand() / 32767.0

    cpdef void access_elements(self):
        cdef int i, j
        cdef double _
        for i in range(self.arr.rows):
            for j in range(self.arr.cols):
                _ = self.arr.data[i][j]

    cpdef void set_elements(self, double value):
        cdef int i, j
        for i in range(self.arr.rows):
            for j in range(self.arr.cols):
                self.arr.data[i][j] = value

cdef class NumpyArray:
    cdef object arr  # Use a generic object instead of cnp.ndarray

    def __cinit__(self, int rows, int cols):
        self.arr = np.empty((rows, cols), dtype=np.float64)  # Initialize the NumPy array

    cpdef void initialize(self):
        self.arr[:, :] = np.random.rand(self.arr.shape[0], self.arr.shape[1])

    cpdef void access_elements(self):
        cdef int i, j
        cdef double[:, :] view = self.arr  # Use memoryview for efficient access
        cdef double _
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                _ = view[i, j]

    cpdef void set_elements(self, double value):
        cdef int i, j
        cdef double[:, :] view = self.arr  # Use memoryview for efficient setting
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):
                view[i, j] = value