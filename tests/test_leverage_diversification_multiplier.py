#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generated by CodiumAI
import pytest
import sys
import os
import numpy as np

# ------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# ------------------------------------------------------------------------------

from src.analysis.leverage import diversification_multiplier, DM_MATRIX


class TestDiversificationMultiplier:

    # Tests that the function returns the correct diversification multiplier for a 2D numpy array of close prices with 2 assets and a period of 14
    def test_diversification_multiplier_two_assets_period_14(self):
        close_prices = np.array([[100, 200], [300, 400], [500, 600]])
        expected_multiplier = 1.0
        multiplier = diversification_multiplier(close_prices, period=14)
        assert (multiplier == expected_multiplier)

    # Tests that the function returns the correct diversification multiplier for a 2D numpy array of close prices with more than 50 assets and a period of 14
    def test_diversification_multiplier_many_assets_period_14(self):
        close_prices = np.random.randint(100, 1000, size=(14, 60))
        expected_multiplier = DM_MATRIX[0.5][50]
        multiplier = diversification_multiplier(close_prices, period=14)
        assert (multiplier == expected_multiplier)

    # Tests that the function returns the correct diversification multiplier for a 2D numpy array of close prices with 2 assets and a period of 1
    def test_diversification_multiplier_two_assets_period_1(self):
        close_prices = np.array([[100, 200], [300, 400]])
        expected_multiplier = DM_MATRIX[0.5][2]
        assert diversification_multiplier(close_prices, period=1) == expected_multiplier

    # Tests that the function returns the correct diversification multiplier for a 2D numpy array of close prices with 2 assets and a period of 100
    def test_diversification_multiplier_two_assets_period_100(self):
        close_prices = np.random.randint(100, 1000, size=(100, 2))
        expected_multiplier = DM_MATRIX[0.5][2]
        assert (
            diversification_multiplier(close_prices, period=100) == expected_multiplier
        )

    # Tests that the function returns 1 for a 2D numpy array of close prices with only 1 asset
    def test_diversification_multiplier_one_asset(self):
        close_prices = np.array([[100], [200], [300]])
        expected_multiplier = 1.0
        assert diversification_multiplier(close_prices) == expected_multiplier

    # Tests that the function returns the correct diversification multiplier for a 2D numpy array of close prices with a period of 0
    def test_diversification_multiplier_zero_period(self):
        close_prices = np.random.randint(100, 1000, size=(0, 5))
        expected_multiplier = DM_MATRIX[0.5][5]
        assert diversification_multiplier(close_prices) == expected_multiplier
