#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd
import os
import sys

from pprint import pprint

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from find_positions import merge_signals  # noqa: E402


class TestMergeSignals(unittest.TestCase):
    def setUp(self):
        pass

    # def test_merge_signals_open_long(self):
    #     data = {
    #         "open_long": np.array([1, 0.0, 2]),
    #         "open_short": np.zeros(3),
    #         "close_long": np.zeros(3),
    #         "close_short": np.zeros(3)
    #     }
    #     result = merge_signals(data)
    #     self.assertTrue(np.array_equal(result["signal"], np.array([1, 0, 1])))

    # def test_merge_signals_close_long(self):
    #     data = {
    #         "open_long": np.zeros(3),
    #         "open_short": np.zeros(3),
    #         "close_long": np.array([1, 0, 2]),
    #         "close_short": np.zeros(3)
    #     }
    #     result = merge_signals(data)
    #     # print(result)
    #     self.assertTrue(np.array_equal(
    #         result["signal"], np.array([0, np.nan, 0])
    #         ),
    #         f"Expected: {np.array([0, np.nan, 0])}, got: {result['signal']}"
    #     )

    def test_merge_signals_complex_scenario(self):
        data = {
            "open_long":   np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]),
            "open_short":  np.array([0, 1, 0, 0, 0, 1, 0, 0, 0]),
            "close_long":  np.array([0, 0, 1, 0, 0, 0, 1, 0, 0]),
            "close_short": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])
        }
        result = merge_signals(data)

        for k, v in result.items():
            print(f"{k}: \t{v}")

        expected_signal = np.array([1, -1, -1, 0, 1, -1, -1, 0, 0])
        # expected_position = np.array([1, -1, -1, 0, 1, -1, -1, 0, 0])
        # np.testing.assert_array_equal(result["signal"], expected_signal)
        # np.testing.assert_array_equal(result["position"], expected_position)


if __name__ == '__main__':
    unittest.main()
