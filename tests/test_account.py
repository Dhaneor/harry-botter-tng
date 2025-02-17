# cython: language_level=3
# distutils: language = c++
import logging
import pytest
from pprint import pprint
from src.analysis.models.account import (
    get_account, _add_position, _get_current_position,
    TradingAccount
)
from src.analysis.models.position import (
    _build_long_position,
    _build_short_position,
    _close_position,
)
from util import get_logger

logger = get_logger(level="DEBUG")


# ------------------------- Tests for the TradingAccount class -------------------------
def test_initialize():
    ta = TradingAccount("test", 2, 2)

    assert isinstance(ta, TradingAccount)


def test_add_position():
    ta = TradingAccount("test", 2, 2)
    pos = _build_long_position(0, 1735000000, 100.0, 10.0)

    ta._add(0, 0, pos)

def test_get_current_position_with_active():
    ta = TradingAccount("test", 2, 2)
    pos = _build_long_position(0, 1735000000, 100.0, 10.0)

    ta._add(0, 0, pos)
    curr = ta._current(0, 0)

    try:
        assert curr == pos, "Current position is not equal to the one that was added"
    except AssertionError as e:
        print(e)
        print("Expected:")
        pprint(pos)
        print("Got:")
        pprint(curr)


def test_get_current_position_with_inactive():
    ta = TradingAccount("test", 2, 2)
    pos = _build_long_position(0, 1735000000, quote_qty=100.0, price=10.0) 
    pos = _close_position(pos, 1736000000, 12.0)
    ta._add(0, 0, pos)
    curr = ta._current(0, 0)

    assert curr is None, f"Expected: None, but got: {curr}"



def test_update_position():
    ta = TradingAccount("test", 2, 2)
    pos = _build_long_position(0, 1735000000, 100.0, 10.0)

    ta._add(0, 0, pos)

    new_pos = _build_long_position(0, 1735000000, 100.0, 20.0)

    ta._replace(0, 0, new_pos)




