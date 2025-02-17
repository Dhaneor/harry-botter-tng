# cython: language_level=3
# distutils: language = c++
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from src.analysis.models.position cimport PositionData


cdef struct Account:
    unordered_map[int, unordered_map[int, vector[PositionData]]] positions

cpdef get_account()
cdef PositionData* get_current_position(Account acc, int m, int s)
cdef Account add_position(Account acc, int m, int s, PositionData position)
cdef void update_position(Account* acc, int m, int s, PositionData* new_pos)
cpdef void print_positions(Account acc)


cdef class TradingAccount:
    cdef int assets
    cdef int strategies
    cdef unordered_map[int, unordered_map[int, vector[PositionData]]] positions

    cdef void add(self, int m, int s, PositionData pos)
    cdef void replace(self, int m, int s, PositionData* pos)
    cdef PositionData* current(self, int m, int s)

    cpdef void print_positions(self)