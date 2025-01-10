#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper class (pytest).

Created on Jan 09 00:44:23 2025

@author dhaneor
"""

import numpy as np
import pytest

from wrappers.base_wrapper import BaseWrapper

data = np.random.rand(10, 5)
columns = ["col1", "col2", "col3", "col4", "col5"]
wrapper = BaseWrapper(data, columns)


def test_base_wrapper_init():
    assert wrapper.data.shape == data.shape
    assert isinstance(wrapper.data, np.ndarray)

def test_init_with_wrong_data_type():
    with pytest.raises(ValueError):
        BaseWrapper("not a numpy array", columns)

def test_init_with_wrong_dimensions():
    with pytest.raises(ValueError):
        BaseWrapper(np.random.rand(10, 2, 2, 2), columns)


def test_wrapper_call_method():
    assert wrapper() is data


def test_len_method():
    assert len(wrapper) == data.shape[0]


# def test_iter_method():
#     for col in enumerate(wrapper):
#         assert isinstance(col, np.ndarray)
#         assert col.shape == (10,)
#         assert col.dtype == np.float64
#         assert col == data[:, col]


# .............................. TESTS FOR __getitem__ ................................
def test_get_item_by_string():
    assert np.array_equal(wrapper["col1"], data[:, 0])
    assert np.array_equal(wrapper["col5"], data[:, 4])


def test_get_item_by_slice():
    assert np.array_equal(wrapper[::2], data[::2]), "Expected: {}, got: {}".format(
        data[::2], wrapper[::2]
    )
    assert np.array_equal(wrapper[1:4], data[1:4])

    # Test 2D slicing
    assert np.array_equal(
        wrapper[1:4, 2:4], data[1:4, 2:4]
    ), "Expected: {}, got: {}".format(data[1:4, 2:4], wrapper[1:4, 2:4])


# .............................. TESTS FOR __setitem__ ................................
def test_set_item_by_string():
    wrapper = BaseWrapper(data.copy(), columns)
    new_values = np.random.rand(10)

    wrapper["col1"] = new_values
    assert np.array_equal(wrapper["col1"], new_values)
    assert np.array_equal(wrapper.data[:, 0], new_values)


def test_set_item_by_slice():
    wrapper = BaseWrapper(data.copy(), columns)
    new_values = np.random.rand(5, 5)

    wrapper[::2] = new_values
    assert np.array_equal(wrapper[::2], new_values)
    assert np.array_equal(wrapper.data[::2], new_values)


def test_set_item_2d():
    wrapper = BaseWrapper(data.copy(), columns)
    new_values = np.random.rand(3, 2)

    wrapper[1:4, 2:4] = new_values
    assert np.array_equal(wrapper[1:4, 2:4], new_values)
    assert np.array_equal(wrapper.data[1:4, 2:4], new_values)


def test_set_item_invalid_column():
    wrapper = BaseWrapper(data.copy(), columns)
    with pytest.raises(KeyError):
        wrapper["non_existent_column"] = np.random.rand(10)


def test_set_item_invalid_type():
    wrapper = BaseWrapper(data.copy(), columns)
    with pytest.raises(TypeError):
        wrapper[{}] = 1  # Using a dictionary as an index should raise TypeError


# ......................... TESTS FOR __and__/__or__/__add__ ..........................
def test_and_operator():
    assert np.array_equal(wrapper & wrapper, np.ones((10, 5)))
    # assert np.array_equal(wrapper & 1, wrapper)


def test_or_operator():
    assert np.array_equal(wrapper | wrapper, np.ones((10, 5)))


def test_xor_operator():
    assert np.array_equal(wrapper ^ wrapper, np.zeros((10, 5)))


def test_add_operator():
    assert np.array_equal(wrapper + wrapper, 2 * data)


# .................. TESTS FOR Numpy Array related methods/properties .................
def test_shape():
    assert wrapper.shape == data.shape


def test_ndim():
    assert wrapper.ndim == data.ndim


def test_ffill():
    assert np.array_equal(wrapper.ffill(), data)
    assert isinstance(wrapper.data, np.ndarray)

def test_replace():
    data = np.ndarray([0, 1, 3, 4, 5]).reshape(-1, 1)
    expected = np.ndarray([0, 1, 3, 4, 0]).reshape(-1, 1)

    wrapper = BaseWrapper(data, columns)
    wrapper.replace(5, 0)

    assert np.array_equal(wrapper.data, expected), \
        f"Expected: {expected}, got: {wrapper.data}"

# ........................... TESTS FOR statistical methods ...........................
# def test_mean():
#     assert np.isclose(wrapper.mean(), np.mean(data), atol=1e-5)


# def test_std():
#     assert np.isclose(wrapper.std(), np.std(data), atol=1e-5)


if __name__ == "__main__":
    pytest.main()
