# cython: language_level=3
# distutils: language = c++
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from analysis.models.position cimport PositionData


cdef struct Account:
    unordered_map[int, unordered_map[int, vector[PositionData]]] positions

cdef PositionData* get_current_position(self, Account acc, int m, int s)