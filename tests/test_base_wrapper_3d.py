#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides test functions for the BaseWrapper3D class (pytest).

Created on Jan 09 00:44:23 2025

@author dhaneor
"""

import numpy as np
import pytest

from misc.base_wrapper import BaseWrapper3D
from misc.exceptions import DimensionMismatchError

shape = (10, 5, 3)
data = np.random.rand(*shape)
columns = ["col1", "col2", "col3", "col4", "col5"]
layers = ["layer1", "layer2", "layer3"]
wrapper = BaseWrapper3D(data, columns, layers)


def test_base_wrapper_init():
    assert wrapper.data.shape == data.shape
    assert isinstance(wrapper.data, np.ndarray)

def test_init_with_wrong_data_type():
    with pytest.raises(ValueError):
        BaseWrapper3D("not a numpy array", columns, layers)

def test_init_with_wrong_dimensions():
    with pytest.raises(ValueError):
        BaseWrapper3D(np.random.rand(10, 2, 2, 2), columns, layers)


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
    assert np.array_equal(wrapper["col1"], data[:, 0, :])
    assert np.array_equal(wrapper["col5"], data[:, 4, :])
    assert np.array_equal(wrapper["layer1"], data[:, :, 0])
    assert np.array_equal(wrapper["layer3"], data[:, :, 2])

def test_get_item_by_slice():
    assert np.array_equal(wrapper[::2], data[::2]), "Expected: {}, got: {}".format(
        data[::2], wrapper[::2]
    )
    assert np.array_equal(wrapper[1:4], data[1:4])

    # Test 3D slicing
    assert np.array_equal(
        wrapper[1:4, 2:4, 1:], data[1:4, 2:4, 1:]
    ), "Expected: {}, got: {}".format(data[1:4, 2:4, 1:], wrapper[1:4, 2:4, 1:])

# .............................. TESTS FOR __setitem__ ................................
def test_set_item_by_string():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    new_values = np.random.rand(10, 3)

    wrapper["col1"] = new_values
    assert np.array_equal(wrapper["col1"], new_values)
    assert np.array_equal(wrapper.data[:, 0, :], new_values)

    new_layer_values = np.random.rand(10, 5)
    wrapper["layer2"] = new_layer_values
    assert np.array_equal(wrapper["layer2"], new_layer_values)
    assert np.array_equal(wrapper.data[:, :, 1], new_layer_values)

def test_set_item_by_slice():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    new_values = np.random.rand(5, 5, 3)

    wrapper[::2] = new_values
    assert np.array_equal(wrapper[::2], new_values)
    assert np.array_equal(wrapper.data[::2], new_values)

def test_set_item_3d():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    new_values = np.random.rand(3, 2, 2)

    wrapper[1:4, 2:4, 1:] = new_values
    assert np.array_equal(wrapper[1:4, 2:4, 1:], new_values)
    assert np.array_equal(wrapper.data[1:4, 2:4, 1:], new_values)

def test_set_item_invalid_column():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    with pytest.raises(KeyError):
        wrapper["non_existent_column"] = np.random.rand(*shape, 3)

def test_set_item_invalid_layer():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    with pytest.raises(KeyError):
        wrapper["non_existent_layer"] = np.random.rand(*shape)

def test_set_item_invalid_type():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    with pytest.raises(TypeError):
        wrapper[{}] = 1  # Using a dictionary as an index should raise TypeError

def test_set_item_invalid_dimensions():
    wrapper = BaseWrapper3D(data.copy(), columns, layers)
    with pytest.raises(DimensionMismatchError):
        wrapper["col1"] = np.random.rand(10, 5)  # Wrong dimensions for a column
    with pytest.raises(DimensionMismatchError):
        wrapper["layer1"] = np.random.rand(10, 3)  # Wrong dimensions for a layer

# ......................... TESTS FOR __and__/__or__/__add__ ..........................
def test_and_operator():
    assert np.array_equal(wrapper & wrapper, np.ones(shape))
    # assert np.array_equal(wrapper & 1, wrapper)


def test_or_operator():
    assert np.array_equal(wrapper | wrapper, np.ones(shape))


def test_xor_operator():
    assert np.array_equal(wrapper ^ wrapper, np.zeros(shape))


def test_add_operator():
    assert np.array_equal(wrapper + wrapper, 2 * data)


# .................. TESTS FOR Numpy Array related methods/properties .................
def test_shape():
    assert wrapper.shape == data.shape


def test_ndim():
    assert wrapper.ndim == data.ndim


# def test_ffill():
#     assert np.array_equal(wrapper.ffill(), data)
#     assert isinstance(wrapper.data, np.ndarray)

# def test_replace():
#     data = np.ndarray([0, 1, 3, 4, 5]).reshape(-1, 1)
#     expected = np.ndarray([0, 1, 3, 4, 0]).reshape(-1, 1)

#     wrapper = BaseWrapper3D(data, columns)
#     wrapper.replace(5, 0)

#     assert np.array_equal(wrapper.data, expected), \
#         f"Expected: {expected}, got: {wrapper.data}"

# ........................... TESTS FOR statistical methods ...........................
# def test_mean():
#     assert np.isclose(wrapper.mean(), np.mean(data), atol=1e-5)


# def test_std():
#     assert np.isclose(wrapper.std(), np.std(data), atol=1e-5)


if __name__ == "__main__":
    pytest.main()
