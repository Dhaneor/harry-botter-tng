# cython: language_level=3
# distutils: language = c++
import logging
import pytest
from pprint import pprint
from src.analysis.models.account import get_account, _add_position, _get_current_position
from src.analysis.models.position import (
    _build_long_position,
)
from util import get_logger

logger = get_logger(level="DEBUG")


def test_get_current_position():
    # Create a new account
    acc = get_account()

    # Create a sample position
    # sample_position = PositionData()
    # sample_position.idx = 1
    # sample_position.size = 100.0
    sample_position = _build_long_position(0, 1735000000, 10.0, 100.0)

    # Add the position to the account
    market_id = 1
    symbol_id = 1
    acc = _add_position(acc, market_id, symbol_id, sample_position)

    # Try to get the current position
    result = _get_current_position(acc, market_id, symbol_id)

    # Check if the result is not None
    assert result is not None

    # Check if the returned position matches the one we added
    try:
        assert result == sample_position, f"Expected: {sample_position}\nGot: {result}\n" 
    except AssertionError as e:
        print("Expected:")
        pprint(sample_position)
        print("-" * 80)
        print("Got:")
        pprint(result)
        raise

def test_get_current_position_nonexistent():
    # Create a new account
    acc = get_account()

    # Try to get a position that doesn't exist
    result = _get_current_position(acc, 1, 1)

    # The result should be None
    assert result is None

# Add more tests as needed


