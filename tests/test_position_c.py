# cython: language_level=3
# distutils: language = c++
import pytest
from src.analysis.models.position import (
    _get_fee,
    _get_slippage,
    _build_buy_trade,
    _build_sell_trade,
    _add_buy,
    _add_sell,
    _get_avg_entry_price,
    _build_long_position,
    _build_short_position,
)

fee_rate = 0.001
slippage_rate = 0.001


def test_get_fee():
    fee = _get_fee(1000, fee_rate)
    assert fee == 1


def test_get_slippage():
    assert _get_slippage(1000, slippage_rate) == pytest.approx(1.0)


def test_build_buy_trade():
    t = _build_buy_trade(timestamp=1750000000, quote_qty=1000.0, price=100.0)
    fee = _get_slippage(1000, slippage_rate)
    slippage = _get_slippage(1000, slippage_rate)
    base_qty = 10.0 * (1 - (fee_rate + slippage_rate))
    
    assert t["type"] == 1
    assert t["timestamp"] == 1750000000
    assert t["price"] == 100.0
    assert t["qty"] == base_qty
    assert t["gross_quote_qty"] == 1000.0
    assert t["net_quote_qty"] == 1000.0 - fee - slippage
    assert t["fee"] == fee
    assert t["slippage"] == slippage


def test_build_sell_trade():
    base_qty = 1000.0
    price = 100 
    gross_quote_qty = base_qty * price
    fee = _get_fee(gross_quote_qty, fee_rate)
    slippage = _get_slippage(gross_quote_qty, slippage_rate)
    net_quote_qty = gross_quote_qty - fee - slippage
    
    t = _build_sell_trade(timestamp=1750000000, base_qty=1000.0, price=100.0)
    
    assert t["type"] == -1
    assert t["timestamp"] == 1750000000
    assert t["price"] == price
    assert t["qty"] == base_qty
    assert t["net_quote_qty"] == net_quote_qty
    assert t["fee"] == fee
    assert t["slippage"] == slippage


def test_add_buy():
    quote_qty = 1000.0
    price = 100.0
    p = _build_long_position(0, 1735000000, quote_qty=quote_qty, price=price)
    p = _add_buy(p, 1736000000, quote_qty=quote_qty, price=price)

    assert len(p["trades"]) == 2


def test_add_sell():
    quote_qty = 1000.0
    price = 100.0
    p = _build_short_position(0, 1735000000, base_qty=quote_qty, price=price)
    p = _add_sell(p, 1736000000, base_qty=quote_qty, price=price)

    assert len(p["trades"]) == 2


precision = 1e-04


def test_get_avg_entry_price_long():
    quote_qty = 1000.0
    price = 100.0
    price_1 = 200.0
    p = _build_long_position(0, 1735000000, quote_qty=quote_qty, price=price)
    p = _add_buy(p, 1736000000, quote_qty=quote_qty, price=price)

    calculated = _get_avg_entry_price(p)
    expected  = price * (1 + fee_rate + slippage_rate)
    assert calculated == pytest.approx(expected, abs=precision)
    
    p = _add_buy(p, 1736700000, quote_qty=quote_qty * 4, price=price_1)
    
    calculated = _get_avg_entry_price(p)
    expected = (price * 1.5) * (1 + fee_rate + slippage_rate)
    assert calculated == pytest.approx(expected, abs=precision)


def test_get_avg_entry_price_short():
    base_qty = 10.0
    price = 100.0
    price_2 = 50.0
    p = _build_short_position(0, 1735000000, base_qty=base_qty, price=price)
    p = _add_sell(p, 1736000000, base_qty=base_qty, price=price)

    calculated = _get_avg_entry_price(p)
    expected = price * (1 - fee_rate - slippage_rate)
    assert calculated == pytest.approx(expected, abs=precision)

    p = _add_sell(p, 1736700000, base_qty=base_qty * 2, price=price_2)
    calculated = _get_avg_entry_price(p)
    expected = ((price + price_2) / 2) * (1 - fee_rate - slippage_rate)
    assert calculated == pytest.approx(expected, abs=precision)


def test_build_long_position():
    quote_qty = 1000.0
    price = 100.0
    fee = quote_qty * fee_rate
    slippage = quote_qty * slippage_rate
    quote_qty_net = quote_qty - fee - slippage
    base_qty = quote_qty_net / price
    p = _build_long_position(0, 1735000000, quote_qty=quote_qty, price=price)

    assert p["type"] == 1
    assert p["is_active"] == 1
    assert p["duration"] == 1
    assert p["size"] == base_qty
    assert p["avg_entry_price"] == price
    assert p["pnl"] == 0
    assert len(p["trades"]) == 1
    assert len(p["stop_orders"]) == 0


def test_build_short_position():
    base_qty = 1000.0
    price = 100.0
    p = _build_short_position(0, 1735000000, base_qty=base_qty, price=price)

    assert p["type"] == -1
    assert p["is_active"] == 1
    assert p["duration"] == 1
    assert p["size"] == -base_qty
    assert p["avg_entry_price"] == price
    assert p["pnl"] == 0
    assert len(p["trades"]) == 1
    assert len(p["stop_orders"]) == 0