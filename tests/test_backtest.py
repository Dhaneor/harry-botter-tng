#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 14 01:22:23 2025

@author dhaneor
"""

import pytest
import numpy as np

from analysis.backtest.backtest import BackTest, Config, run_backtest
from analysis import (
    MarketData,
    MarketDataStore,
    LeverageCalculator,
    signal_generator_factory,
    SignalGeneratorDefinition,
)
from models.enums import COMPARISON
from util.logger_setup import get_logger

logger = get_logger('main', level="DEBUG")


# ==================================== FIXTURES =======================================
@pytest.fixture
def config():
    return Config(
        initial_capital=10_000,
        rebalance_position=True,
        increase_allowed=True,
        decrease_allowed=True,
    )


@pytest.fixture
def market_data():
    def _market_data(
        number_of_periods: int, number_of_assets: int, data_type: str = "random"
    ):
        if data_type == "random":
            market_data = MarketData.from_random(number_of_periods, number_of_assets)
            return market_data.mds
        elif data_type == "fixed":
            # Create synthetic data
            timestamps = (
                np.arange(number_of_periods, dtype=np.int64) * 60000
            )  # 1-minute intervals
            base_price = 100.0
            trend = np.linspace(0, 20, number_of_periods)

            open_prices = np.tile(
                base_price + trend, (number_of_assets, 1)
            ).T + np.random.normal(0, 1, (number_of_periods, number_of_assets))
            close_prices = open_prices + np.random.normal(
                0, 0.5, (number_of_periods, number_of_assets)
            )
            high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(
                0, 0.5, (number_of_periods, number_of_assets)
            )
            low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(
                0, 0.5, (number_of_periods, number_of_assets)
            )
            volumes = np.random.uniform(
                1000, 5000, (number_of_periods, number_of_assets)
            )

            mds = MarketDataStore(
                open_=open_prices.astype(np.float32),
                high=high_prices.astype(np.float32),
                low=low_prices.astype(np.float32),
                close=close_prices.astype(np.float32),
                volume=volumes.astype(np.float32),
                timestamp=timestamps.reshape(-1, 1),
            )
            return MarketData(mds, [f"Asset{i+1}" for i in range(number_of_assets)])
        else:
            raise ValueError("Invalid data_type. Choose 'random' or 'fixed'.")

    return _market_data


@pytest.fixture
def leverage_array():
    def _leverage_array(market_data: MarketData):
        # Create a LeverageCalculator instance
        lc = LeverageCalculator(market_data)

        # Calculate leverage values
        leverage_values = lc.leverage()

        return leverage_values

    return _leverage_array


@pytest.fixture
def signals_array():
    def _signals_array(
        market_data_store: MarketDataStore,
        signal_generator_def: SignalGeneratorDefinition,
        number_of_strategies: int = 1
    ):
        # Create a SignalGenerator instance
        signal_generator = signal_generator_factory(signal_generator_def)

        # Set the market data for the signal generator
        signal_generator.market_data = market_data_store

        # Run the signal generator to produce signals
        base_signals = signal_generator.execute()

        # Extend the signals to the specified number of strategies
        signals = np.repeat(base_signals, number_of_strategies, axis=2)

        return signals

    return _signals_array


# ===================================== TESTS =========================================
def test_fixtures(market_data, leverage_array, signals_array):
    # Test market_data fixture
    md = market_data(number_of_periods=100, number_of_assets=2, data_type="fixed")
    assert isinstance(md, MarketData)
    assert md.mds.open_.shape == (100, 2)
    assert len(md.symbols) == 2

    # Test leverage_array fixture
    leverage = leverage_array(md)
    assert isinstance(leverage, np.ndarray)
    assert leverage.shape == (100, 2)
    assert np.all((leverage >= 0) | np.isnan(leverage))

    # Test signals_array fixture
    # We need to create a simple SignalGeneratorDefinition for this test
    signal_gen_def = SignalGeneratorDefinition(
        name="TestSignalGenerator",
        operands={"sma": ("sma"), "close": "close"},
        conditions={
            "open_long": [("close", COMPARISON.IS_ABOVE, "sma")],
            "open_short": [("close", COMPARISON.IS_BELOW, "sma")],
        },
    )
    signals = signals_array(md, signal_gen_def)
    assert isinstance(signals, np.ndarray)
    assert signals.shape == (100, 2, 1)
    for field in ("open_long", "open_short", "close_long", "close_short"):
        f = signals[field]
        assert np.all((f == 1) | (f == 0))  # Assuming signals are -1, 0, or 1

    print("All fixtures are working as expected.")


def test_backtest_config(config):
    config_ = config
    assert isinstance(config_, Config), "Config instance creation failed."
    assert config_.initial_capital == 10_000.0, "Initial capital mismatch."
    assert config_.rebalance_position == True, "Rebalance position mismatch."  # noqa: E712
    assert config_.increase_allowed == True, "Increase allowed mismatch."  # noqa: E712
    assert config_.decrease_allowed == True, "Decrease allowed mismatch."  # noqa: E712

    print("BacktestConfig fixture is working as expected.")


def test_backtest_init(config, market_data, leverage_array, signals_array):
    md = market_data(number_of_periods=300, number_of_assets=2, data_type="fixed")
    leverage = leverage_array(md)
    signal_gen_def = SignalGeneratorDefinition(
        name="TestSignalGenerator",
        operands={"sma": ("sma"), "close": "close"},
        conditions={
            "open_long": [("close", COMPARISON.IS_ABOVE, "sma")],
            "open_short": [("close", COMPARISON.IS_BELOW, "sma")],
        },
    )
    signals = signals_array(md, signal_gen_def)
    config_ = config

    try:
        bt = BackTest(md.mds, leverage, signals, config_)
    except Exception as e:
        print(f"Error in BackTest init: {str(e)}")
        raise
    else:
        assert isinstance(bt, BackTest), "BackTest instance creation failed."

        # Compare `leverage` arrays correctly
        np.testing.assert_array_equal(
            bt.leverage, leverage, err_msg="Leverage mismatch ..."
        )

        # Compare `signals` arrays correctly
        np.testing.assert_array_equal(
            bt.signals, signals, err_msg="Signals mismatch ..."
        )

        # Compare `market_data` attributes individually
        for attr in ["open_", "close", "high", "low", "volume"]:
            bt_attr = getattr(bt.market_data, attr, None)
            md_attr = getattr(md.mds, attr, None)  # Adjust if `md` structure differs

            assert (
                bt_attr is not None
            ), f"Attribute '{attr}' missing in BackTest.market_data"
            assert md_attr is not None, f"Attribute '{attr}' missing in md.mds"
            np.testing.assert_array_equal(
                bt_attr, md_attr, err_msg=f"MarketData '{attr}' mismatch..."
            )
        # assert bt.config == config
        # assert bt.rebalance_fn is None
        # assert bt.stop_order_fn is None


def test_backtest_run(market_data, leverage_array, signals_array, config):
    periods = 1_000
    assets = 1
    strategies = 1_000

    md = market_data(
        number_of_periods=periods, number_of_assets=assets, data_type="fixed"
        )
    leverage = leverage_array(md)
    
    signal_gen_def = SignalGeneratorDefinition(
        name="TestSignalGenerator",
        operands={"sma": ("sma"), "close": "close"},
        conditions={
            "open_long": [("close", COMPARISON.IS_ABOVE, "sma")],
            "open_short": [("close", COMPARISON.IS_BELOW, "sma")],
        },
    )
    signals = signals_array(md, signal_gen_def, strategies)

    logger.info(
        "shape of signals array: %s (%s backtests)", signals.shape, assets * strategies
        )

    bt = BackTest(md.mds, leverage, signals, config)
    
    try:
        result = bt.run()
    except Exception as e:
        print(f"Error in BackTest run: {str(e)}")
        raise

    assert isinstance(result, np.ndarray), "Positions array creation failed."
    assert result.shape == signals.shape, "Portfolios array shape mismatch."


def test_run_backtest_fn(market_data, leverage_array, signals_array, config):
    periods = 1_000
    assets = 10
    strategies = 10_000

    md = market_data(
        number_of_periods=periods, number_of_assets=assets, data_type="fixed"
        )
    leverage = leverage_array(md)
    
    signal_gen_def = SignalGeneratorDefinition(
        name="TestSignalGenerator",
        operands={"sma": ("sma"), "close": "close"},
        conditions={
            "open_long": [("close", COMPARISON.IS_ABOVE, "sma")],
            "open_short": [("close", COMPARISON.IS_BELOW, "sma")],
        },
    )
    signals = signals_array(md, signal_gen_def, strategies)
    
    try:
        result = run_backtest(md.mds, leverage, signals, config)
    except Exception as e:
        print(f"Error in BackTest run: {str(e)}")
        raise

    assert isinstance(result, np.ndarray), "Positions array creation failed."
    assert result.shape == signals.shape, "Portfolios array shape mismatch."