# cython: language_level=3
# distutils: language = c++
# cython: numpy_api=2

cimport numpy as np
import logging
import numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

from src.analysis.models.market_data_store cimport MarketDataStore, MarketState
from src.analysis.models.position cimport (
    TradeData,
    PositionData,
    StopOrder,
    build_buy_trade,
    build_sell_trade,
    add_buy,
    add_sell,
    build_long_position,
    build_short_position,
    close_position,
)

logger = logging.getLogger(f"main.{__name__}")


cpdef get_account():
    cdef Account acc = Account(positions={})
    return acc


cdef PositionData* get_current_position(Account acc, int m, int s):
    if acc.positions.find(m) == acc.positions.end():
        return NULL
 
    if acc.positions[m].find(s) == acc.positions[m].end():
        return NULL

    if acc.positions[m][s].empty():
        return NULL

    cdef PositionData* pos = &acc.positions[m][s].back()
    pos.duration += 1 
    logger.debug("[get_current] %s" % str(pos[0]))

    return pos


cdef Account add_position(Account acc, int m, int s, PositionData position):
    # logger.debug(f"Adding Position for market {m}, symbol {s}")

    if acc.positions.find(m) == acc.positions.end():
        # logger.debug(f"Creating new map for market {m}")
        acc.positions[m] = unordered_map[int, vector[PositionData]]()
        # logger.debug("added: %s" % str(acc.positions[m]))
    if acc.positions[m].find(s) == acc.positions[m].end():
        # logger.debug(f"adding sub-map for market {m}, symbol {s}")
        acc.positions[m][s] = vector[PositionData]()

    # logger.debug(f"Adding position ... ")
    acc.positions[m][s].push_back(position)
    # logger.debug("added: %s" % acc.positions[m][s].back())
    return acc


cdef void update_position(Account* acc, int m, int s, const PositionData* new_pos):
    cdef PositionData* pos_ptr = get_current_position(acc[0], m, s)
    if pos_ptr == NULL:
        logger.warning("No position found to update")
        return

    if acc.positions.find(m) == acc.positions.end():
        logger.warning(f"No map for market {m}")
        return
    if acc.positions[m].find(s) == acc.positions[m].end():
        logger.warning(f"No sub-map for market {m}, symbol {s}")
        return

    # Update the existing position in place
    acc.positions[m][s].pop_back()
    acc.positions[m][s].push_back(new_pos[0])


cpdef void print_positions(Account acc):
    """Prints all positions stored in the account."""
    cdef int m  # Market ID
    cdef int s  # Symbol ID
    cdef vector[PositionData] positions
    
    for market_iter in acc.positions:
        m = market_iter.first
        logger.debug(f"[Market] {m}:")
        for symbol_iter in market_iter.second:
            s = symbol_iter.first
            positions = symbol_iter.second
            
            logger.debug(f"  [Symbol] {s}: {positions.size()} positions")
            for pos in positions:
                logger.debug(f"    [PositionData] id={pos.idx}, size={pos.size}")
    

# ................... Python accessible versions of the functions above ................
cdef PositionData deep_copy_position(const PositionData& pos):
    cdef PositionData new_pos
    new_pos = pos  # This copies the basic fields
    new_pos.trades = pos.trades  # This should perform a deep copy of the vector
    new_pos.stop_orders = pos.stop_orders
    return new_pos

# In your _get_current_position function:
def _get_current_position(acc: Account, m: int, s: int) -> PositionData | None:
    result = get_current_position(acc, m, s)
    if result == NULL:
        return None
    return deep_copy_position(result[0])

def _add_position(acc: Account, m: int, s: int, pos: PositionData) -> Account:
    return add_position(acc, m, s, pos)