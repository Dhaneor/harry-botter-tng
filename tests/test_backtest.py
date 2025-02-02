#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on January 14 01:22:23 2025

@author dhaneor
"""

import pytest
import numpy as np
import pandas as pd

from analysis.backtest.backtest import (
    BackTestCore,
    Config,
    run_backtest,
    WARMUP_PERIODS,
)
from analysis import (
    MarketData,
    MarketDataStore,
    LeverageCalculator,
    signal_generator_factory,
    SignalGeneratorDefinition,
    SIGNALS_DTYPE,
    POSITION_DTYPE,
)
from models.enums import COMPARISON
from util.logger_setup import get_logger

logger = get_logger("main", level="DEBUG")

# Set display options to show all columns
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Don't wrap to multiple lines
pd.set_option("display.max_colwidth", None)  # Show full content of each column


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
            market_data = MarketData.from_random(
                number_of_periods, number_of_assets, 0.025
            )
            return market_data
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
                open_=open_prices.astype(np.float64),
                high=high_prices.astype(np.float64),
                low=low_prices.astype(np.float64),
                close=close_prices.astype(np.float64),
                volume=volumes.astype(np.float64),
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
        number_of_strategies: int = 1,
    ):
        # Create a SignalGenerator instance
        signal_generator = signal_generator_factory(signal_generator_def)

        # Set the market data for the signal generator
        signal_generator.market_data = market_data_store

        # Run the signal generator to produce signals
        base_signals = signal_generator.execute(compact=True)

        # Extend the signals to the specified number of strategies
        signals = np.repeat(base_signals, number_of_strategies, axis=2)

        return signals

    return _signals_array


def result_column_to_dataframe(result, symbol_index, strategies_index):
    # Convert the result column to a DataFrame
    df = pd.DataFrame()

    for field in POSITION_DTYPE.names:
        df[f"{field}.{symbol_index + 1}.{strategies_index + 1}"] = result[
            :, symbol_index, strategies_index
        ][field]

    return df


def print_df_for_result_column(
    md, result, leverage, signals, symbol_index, strategies_index
):
    df = result_column_to_dataframe(
        result[WARMUP_PERIODS:], symbol_index, strategies_index
    )
    df.insert(0, "leverage", leverage[WARMUP_PERIODS:, symbol_index])
    df.insert(1, "signals", signals[WARMUP_PERIODS:, symbol_index, strategies_index])
    df.insert(0, "close_price", md.mds.close[WARMUP_PERIODS:, symbol_index])
    # df.close_price = df.close_price.round(2)

    buy_qty_str = f"buy_qty.{symbol_index + 1}.{strategies_index + 1}"
    sell_qty_str = f"sell_qty.{symbol_index + 1}.{strategies_index + 1}"
    fee_str = f"fee.{symbol_index + 1}.{strategies_index + 1}"
    slippage_str = f"slippage.{symbol_index + 1}.{strategies_index + 1}"

    df.insert(15, "change_quote", 0)
    df.change_quote = df.change_quote.astype(np.float64)

    df.loc[df[buy_qty_str] != 0, "change_quote"] = (
        - (df[buy_qty_str] * df.close_price * -1).astype(np.float64)
        - df[fee_str]
        - df[slippage_str]
    )
    df.loc[df[sell_qty_str] != 0, "change_quote"] = (
        (df[sell_qty_str] * df.close_price).astype(np.float64)
        - df[fee_str]
        - df[slippage_str]
    )

    df.insert(16, "check_diff", np.nan)
    df.loc[df[buy_qty_str] != 0, "check_diff"] = (
        df[buy_qty_str] * df.close_price
        + df[fee_str]
        + df[slippage_str]
        + df["change_quote"]
    )
    df.loc[df[sell_qty_str] != 0, "check_diff"] = (
        df[sell_qty_str] * df.close_price
        -  df[fee_str]
        -  df[slippage_str]
        - df["change_quote"]
    )

    # Convert columns to float
    for col in df.columns:
        if col != "duration":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    df = df.round(3)
    df = df.replace(0.0, ".", inplace=False)
    df = df.replace(np.nan, "", inplace=False)
    print(df)


# ===================================== TESTS =========================================
def test_fixtures(market_data, leverage_array, signals_array):
    periods = 250

    # Test market_data fixture
    md = market_data(number_of_periods=periods, number_of_assets=2, data_type="fixed")
    assert isinstance(md, MarketData)
    assert md.mds.open_.shape == (periods, 2)
    assert len(md.symbols) == 2

    # Test leverage_array fixture
    leverage = leverage_array(md)
    assert isinstance(leverage, np.ndarray)
    assert leverage.shape == (periods, 2)
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
    assert signals.shape == (periods, 2, 1)
    if signals.dtype == SIGNALS_DTYPE:
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
        bt = BackTestCore(md.mds, leverage, signals, config_)
    except Exception as e:
        print(f"Error in BackTestCore init: {str(e)}")
        print("leverage dtype: ", leverage.dtype)
        print("signals dtype: ", signals.dtype)
        raise
    else:
        print("BacktTestCore initialized: OK")
        assert isinstance(bt, BackTestCore), "BackTestCore instance creation failed."

        try:
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
                ), f"Attribute '{attr}' missing in BackTestCore.market_data"
                assert md_attr is not None, f"Attribute '{attr}' missing in md.mds"
                np.testing.assert_array_equal(
                    bt_attr, md_attr, err_msg=f"MarketData '{attr}' mismatch..."
                )
        except AssertionError as e:
            print("assertion error:", str(e))
            raise
        # assert bt.config == config
        # assert bt.rebalance_fn is None
        # assert bt.stop_order_fn is None


def test_backtest_run(market_data, leverage_array, signals_array, config):
    periods = 1_000
    assets = 10
    strategies = 10

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

    bt = BackTestCore(md.mds, leverage, signals, config)

    try:
        result, _ = bt.run()
    except Exception as e:
        print(f"Error in BackTestCore run: {str(e)}")
        raise

    assert isinstance(result, np.ndarray), "Positions array creation failed."
    assert result.shape == signals.shape, "Portfolios array shape mismatch."


def test_run_backtest_fn(market_data, leverage_array, signals_array, config):
    periods = 1_000
    assets = 10
    strategies = 10

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
        result, _ = run_backtest(md.mds, leverage, signals, config)
    except Exception as e:
        print(f"Error in BackTestCore run: {str(e)}")
        raise

    assert isinstance(result, np.ndarray), "Positions array creation failed."
    assert result.shape == signals.shape, "Portfolios array shape mismatch."


def test_backtest_run_correctness(market_data, leverage_array, config):
    periods = 220  # Ensure we have enough periods after WARMUP_PERIODS
    assets = 2
    strategies = 1

    long_start = WARMUP_PERIODS + 1
    long_end = long_start + 5
    short_start = WARMUP_PERIODS + 6
    short_end = short_start + 5

    md = market_data(
        number_of_periods=periods, number_of_assets=assets, data_type="fixed"
    )
    leverage = leverage_array(md)

    # Create a simple signal array for testing
    signals = np.zeros((periods, assets, strategies), dtype=np.float64)
    signals[long_start:long_end, 0, 0] = 1  # Long position for asset 0
    signals[short_start:short_end, 1, 0] = -1  # Short position for asset 1

    bt = BackTestCore(md.mds, leverage, signals, config)
    result, _ = bt.run()

    # Check the shape and dtype of the result
    assert result.shape == (periods, assets, strategies)
    assert (
        result.dtype == POSITION_DTYPE
    ), f"Result dtype mismatch. Exepcted POSITION_DTYPE, got: {result.dtype}"

    # Check that no positions are opened before WARMUP_PERIODS
    assert np.all(
        result[:WARMUP_PERIODS]["position"] == 0
    ), "Positions opened before end of warmup period"

    # ---------------------- Check long position for asset 0 ---------------------------
    try:
        assert np.all(
            result[long_start + 1 : long_end + 1, 0, 0]["position"] == 1
        ), "Long position for asset 0 not opened for correct period"
        assert np.all(
            result[long_start + 1 : long_end + 1, 0, 0]["qty"] > 0
        ), "Long position for asset 0 -qty not positive for correct period"
        assert np.any(
            result["qty"][long_start + 1 : long_end + 1, 0, 0] > 0
        ), "Long position qty is 0 for all periods for asset 0"
        assert np.all(
            result[: long_start + 1, 0, 0]["position"] == 0
        ), "Long position for asset 0 opened before signal"
        assert np.all(
            result[long_end + 1 :, 0, 0]["position"] == 0
        ), "Long position for asset 0 closed after signal"
    except AssertionError as e:
        print(f"Error: {str(e)}")
        print_df_for_result_column(md, result, leverage, signals, 0, 0)
        raise e

    # ----------------------- Check short position for asset 1 -------------------------
    try:
        assert np.all(
            result[short_start + 1 : short_end + 1, 1, 0]["position"] == -1
        ), "Short position for asset 1 not opened for correct period"
        assert np.all(
            result[short_start + 1 : short_end + 1, 1, 0]["qty"] < 0
        ), "Short position for asset 1 qty not negative for correct period"
        assert np.any(
            result[short_start + 1 : short_end + 1, 1, 0]["qty"] < 0
        ), "Short position qty is 0 for all periods for asset 1"
        assert np.all(
            result[: short_start + 1, 1, 0]["position"] == 0
        ), "Short position for asset 1 opened before signal"
        assert np.all(
            result[short_end + 1 :, 1, 0]["position"] == 0
        ), "Short position for asset 1 closed after signal"
    except AssertionError as e:
        print(f"Error: {str(e)}")
        print_df_for_result_column(md, result, leverage, signals, 1, 0)
        raise e

def test_backtest_run_with_leverage(market_data, leverage_array, config):
    periods = 220  # Ensure we have enough periods after WARMUP_PERIODS
    assets = 2
    strategies = 1

    long_start = WARMUP_PERIODS + 1
    long_end = long_start + 5

    md = market_data(
        number_of_periods=periods, number_of_assets=assets, data_type="fixed"
    )
    leverage = np.full((periods, assets), 2.0).astype(
        np.float32
    )  # 2x leverage for all periods

    signals = np.zeros((periods, assets, strategies), dtype=np.float64)
    signals[long_start:long_end, 0, 0] = 1  # Long position

    bt = BackTestCore(md.mds, leverage, signals, config)
    result, _ = bt.run()

    try:
        # Check that positions are doubled due to leverage
        assert np.all(
            result[long_start + 1 : long_end + 1, 0, 0]["position"] == 1
        ), "Long position for asset 0 not opened for correct period"
        assert np.all(
            result[long_start + 1 : long_end + 1, 0, 0]["qty"] > 0
        ), "Long position for asset 0 - qty not positive for correct period"
        max_qty = np.max(result[long_start + 1 : long_end + 1, 0, 0]["qty"])
        assert (
            max_qty > 1.9
        ), "Long positions too small"  # Allow for some floating-point imprecision
    except AssertionError as e:
        print(f"Error: {str(e)}")
        print_df_for_result_column(md, result, leverage, signals, 0, 0)
        raise e


# def test_backtest_run_with_multiple_strategies(market_data, leverage_array, config):
#     periods = 400
#     assets = 1
#     strategies = 2

#     md = market_data(
#         number_of_periods=periods, number_of_assets=assets, data_type="fixed"
#     )
#     leverage = leverage_array(md)

#     signals = np.zeros((periods, assets, strategies), dtype=np.float32)
#     signals[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 0, 0] = (
#         1  # Long position for strategy 0
#     )
#     signals[
#         WARMUP_PERIODS + 150 : WARMUP_PERIODS + 200, 0, 1
#     ] = -1  # Short position for strategy 1

#     bt = BackTestCore(md.mds, leverage, signals, config)
#     result = bt.run()

#     # Check positions for strategy 0
#     assert np.all(
#         result[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 0, 0]["position"] == 1
#     )
#     assert np.all(result[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 0, 0]["qty"] > 0)
#     assert np.all(result[: WARMUP_PERIODS + 50, 0, 0]["position"] == 0)
#     assert np.all(result[WARMUP_PERIODS + 100 :, 0, 0]["position"] == 0)

#     # Check positions for strategy 1
#     assert np.all(
#         result[WARMUP_PERIODS + 150 : WARMUP_PERIODS + 200, 0, 1]["position"] == -1
#     )
#     assert np.all(result[WARMUP_PERIODS + 150 : WARMUP_PERIODS + 200, 0, 1]["qty"] < 0)
#     assert np.all(result[: WARMUP_PERIODS + 150, 0, 1]["position"] == 0)
#     assert np.all(result[WARMUP_PERIODS + 200 :, 0, 1]["position"] == 0)


# def test_backtest_run_with_rebalancing(market_data, leverage_array, config):
#     periods = 400
#     assets = 2
#     strategies = 1

#     md = market_data(
#         number_of_periods=periods, number_of_assets=assets, data_type="fixed"
#     )
#     leverage = leverage_array(md)

#     signals = np.zeros((periods, assets, strategies), dtype=np.float32)
#     signals[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 0, 0] = (
#         0.5  # 50% long position for asset 0
#     )
#     signals[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 1, 0] = (
#         0.5  # 50% long position for asset 1
#     )

#     config.rebalance_position = True
#     bt = BackTestCore(md.mds, leverage, signals, config)
#     result = bt.run()

#     # Check that positions are balanced between the two assets
#     qty_0 = result[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 0, 0]["qty"]
#     qty_1 = result[WARMUP_PERIODS + 50 : WARMUP_PERIODS + 100, 1, 0]["qty"]
#     assert np.allclose(qty_0, qty_1, rtol=1e-2)


def test_backtest_run_fields(market_data, leverage_array, config):
    periods = 220
    assets = 2
    strategies = 1

    md = market_data(
        number_of_periods=periods, number_of_assets=assets, data_type="random"
    )
    leverage = leverage_array(md)

    md.mds.close = np.full_like(md.mds.close, 100, dtype=np.float64)
    md.mds.open_ = np.full_like(md.mds.close, 100, dtype=np.float64)

    signals = np.zeros((periods, assets, strategies), dtype=np.float64)
    signals[WARMUP_PERIODS + 1: WARMUP_PERIODS + 18, 0, 0] = 1  # Long position
    signals[WARMUP_PERIODS + 1: WARMUP_PERIODS + 18, 1, 0] = -1  # Short position

    bt = BackTestCore(md.mds, leverage, signals, config)
    result, _ = bt.run()

    assert result.dtype == POSITION_DTYPE

    # Check some specific fields for correctness
    # ... for long position for asset 0
    active_period = result[WARMUP_PERIODS + 7, 0, 0]

    try:
        assert active_period["position"] == 1, "Position is not 1"
        assert active_period["qty"] > 0, "Quantity is not positive"
        assert active_period["entry_price"] > 0, "Entry price is not > 0"
        assert active_period["duration"] > 0, "Duration is not > 0"
        assert active_period["equity"] > 0, "Equity is not > 0"
        # assert active_period["fee"] > 0, "Fee is not > 0"
    except AssertionError as e:
        print(f"Error: {str(e)}")
        print_df_for_result_column(md, result, leverage, signals, 0, 0)
        raise e
    
    # ...same for the short position for asset 1
    active_period = result[WARMUP_PERIODS + 7, 1, 0]

    try:
        assert active_period["position"] == -1, "Position is not -1"
        assert active_period["qty"] < 0, "Quantity is not negative"
        assert active_period["entry_price"] > 0, "Entry price is not > 0"
        assert active_period["duration"] > 0, "Duration is not > 0"
        assert active_period["equity"] > 0, "Equity is not > 0"
        # assert active_period["position"] == 0, "... just checking"
    except AssertionError as e:
        print(f"Error: {str(e)}")
        print_df_for_result_column(md, result, leverage, signals, 1, 0)
        raise e

    # check correct calculation of amounts for changes
    column = result[:, 0, 0]
    for idx in range(WARMUP_PERIODS, periods):
        if column[idx]["buy_qty"] > 0:
            qty = column[idx]["buy_qty"]
            price = column[idx]["buy_price"]
            fee = column[idx]["fee"]
            slippage = column[idx]["slippage"]

            change_quote_expected = qty * price + fee + slippage
            change_quote_real = column[idx - 1]["quote_qty"] - column[idx]["quote_qty"]
            assert change_quote_expected == change_quote_real, \
                f"{change_quote_expected=} != {change_quote_real=}"
            
