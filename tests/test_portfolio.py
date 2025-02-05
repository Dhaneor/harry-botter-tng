#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 06:09:23 2025

@author: dhaneor
"""

import pytest
from analysis.models.portfolio import Buy, Sell

def test_buy_calculation():
    buy = Buy(1000, 100, 10)  # timestamp, amount, price

    try:
        assert buy.qty == pytest.approx(9.98)  # Assuming 1% total for fees and slippage
        assert buy.quote_qty == 100
        assert buy.fee == pytest.approx(0.1)  # Assuming 0.1% fee
        assert buy.slippage == pytest.approx(0.1)  # Assuming 0.1% slippage
        assert buy.price == 10
    except AssertionError as e:
        print(e)
        print(buy)
        print(f"fee: {buy.fee}")
        print(f"slippage: {buy.slippage}")
        raise

def test_sell_calculation():
    sell = Sell(1000, 10, 10)  # timestamp, amount, price
    assert sell.qty == 10
    assert sell.quote_qty == pytest.approx(99.8)  # 100 - 0.2% fees and slippage
    assert sell.fee == pytest.approx(0.1)
    assert sell.slippage == pytest.approx(0.1)
    assert sell.price == 10

def test_buy_addition():
    buy1 = Buy(1000, 100, 10)
    buy2 = Buy(2000, 200, 11)
    combined = buy1 + buy2

    assert combined.qty == pytest.approx(buy1.qty + buy2.qty)
    assert combined.quote_qty == pytest.approx(buy1.quote_qty + buy2.quote_qty)
    assert combined.fee == pytest.approx(buy1.fee + buy2.fee)
    assert combined.slippage == pytest.approx(buy1.slippage + buy2.slippage)
    
    # Check volume-weighted average price
    expected_price = (buy1.price * buy1.qty + buy2.price * buy2.qty) / (buy1.qty + buy2.qty)
    assert combined.price == pytest.approx(expected_price)

def test_sell_addition():
    sell1 = Sell(1000, 10, 10)
    sell2 = Sell(2000, 20, 11)
    combined = sell1 + sell2

    try:
        assert combined.qty == pytest.approx(sell1.qty + sell2.qty)
        assert combined.quote_qty == pytest.approx(sell1.quote_qty + sell2.quote_qty)
        assert combined.fee == pytest.approx(sell1.fee + sell2.fee)
        assert combined.slippage == pytest.approx(sell1.slippage + sell2.slippage)
    
        # Check volume-weighted average price
        expected_price = (sell1.price * sell1.qty + sell2.price * sell2.qty) / (sell1.qty + sell2.qty)
        assert combined.price == pytest.approx(expected_price)
    except AssertionError as e:
        print(e)
        print(sell1)
        print(sell2)
        print('-' * 50)
        print(combined)

def test_invalid_addition():
    buy = Buy(1000, 100, 10)
    sell = Sell(2000, 10, 11)
    
    with pytest.raises(TypeError):
        _ = buy + sell

def test_repr():
    buy = Buy(1000, 100, 10)
    assert repr(buy) == "Buy(timestamp=1000, amount=9.98, price=10.0)"

    sell = Sell(2000, 10, 11)
    assert repr(sell) == "Sell(timestamp=2000, amount=10.0, price=11.0)"