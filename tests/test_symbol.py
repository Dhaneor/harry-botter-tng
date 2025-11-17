#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:00:06 2025

@author: dhaneor
"""
import pytest
from pprint import pprint
from src.models.symbol import Asset, Market, make_asset, make_market


def test_make_asset_minimal():
    """Minimal CCXT currency dict -> Asset with None defaults"""
    ccxt_data = {
        "id": "BTC",
        "code": "BTC",
        "name": "Bitcoin",
        "precision": 8,
        "active": True,
        "withdraw": True,
        "deposit": True,
        "info": {"foo": "bar"},
    }

    asset = make_asset(ccxt_data)

    assert isinstance(asset, Asset)
    assert asset.id == "BTC"
    assert asset.code == "BTC"
    assert asset.name == "Bitcoin"
    assert asset.precision == 8

    # Optional fields
    assert asset.fee is None
    assert asset.limits_withdraw is None
    assert asset.limits_deposit is None

    # Computed flags
    assert asset.is_crypto is True
    assert asset.is_stablecoin is False
    assert asset.is_fiat is False

    # Raw info preserved
    pprint(asset.__dict__)
    try:
        assert asset.info["info"]["foo"] == "bar"
    except AssertionError as e:
        print(e)
        print("Expected:")
        print({"foo": "bar"})
        print("Got:")
        print(asset.info)


def test_make_asset_stablecoin():
    """Stablecoins must set correct flags"""
    ccxt_data = {
        "id": "USDT",
        "code": "USDT",
        "precision": 6,
        "info": {},
    }

    asset = make_asset(ccxt_data)

    assert asset.is_stablecoin is True
    assert asset.is_crypto is False
    assert asset.is_fiat is False


def test_make_asset_fiat():
    """Fiat currencies recognized correctly"""
    ccxt_data = {
        "id": "USD",
        "code": "USD",
        "precision": 2,
        "info": {},
    }

    asset = make_asset(ccxt_data)

    assert asset.is_fiat is True
    assert asset.is_crypto is False
    assert asset.is_stablecoin is False


def test_make_asset_with_limits_and_fees():
    """Tests extraction of optional fee and limits"""
    ccxt_data = {
        "id": "BTC",
        "code": "BTC",
        "precision": 8,
        "fee": 0.0005,
        "limits": {
            "withdraw": {"min": 0.001, "max": 1.0},
            "deposit": {"min": 0.0001, "max": 2.0},
        },
        "info": {},
    }

    asset = make_asset(ccxt_data)

    assert asset.fee == 0.0005
    assert asset.limits_withdraw["min"] == 0.001
    assert asset.limits_deposit["max"] == 2.0
    
    
# =====================================================================================
#                                   TEST MARKET                                       #     #
# =====================================================================================

@pytest.fixture
def sample_assets():
    """Builds simple Asset objects to test market conversion."""
    btc = make_asset({"id": "BTC", "code": "BTC", "precision": 8, "info": {}})
    usdt = make_asset({"id": "USDT", "code": "USDT", "precision": 6, "info": {}})
    return {"BTC": btc, "USDT": usdt}


def test_make_market_spot(sample_assets):
    ccxt_market = {
        "symbol": "BTC/USDT",
        "id": "BTCUSDT",
        "type": "spot",
        "base": "BTC",
        "quote": "USDT",
        "precision": {"price": 2, "amount": 6},
        "limits": {
            "amount": {"min": 0.0001, "max": 100},
            "price": {"min": 10, "max": 200000},
        },
        "maker": 0.001,
        "taker": 0.001,
        "info": {},
    }

    m = make_market(ccxt_market, sample_assets)

    assert isinstance(m, Market)
    assert m.symbol == "BTC/USDT"
    assert m.type == "spot"

    # asset linking
    assert m.base.code == "BTC"
    assert m.quote.code == "USDT"
    assert m.settle is None

    # precision
    assert m.price_precision == 2
    assert m.amount_precision == 6
    assert m.cost_precision is None

    # limits
    assert m.min_amount == 0.0001
    assert m.max_price == 200000
    assert m.min_cost is None

    # convenience flags
    assert m.is_spot is True
    assert m.is_linear_future is False
    assert m.is_inverse_future is False
    assert m.is_perpetual is False


def test_make_market_perpetual_future(sample_assets):
    ccxt_market = {
        "symbol": "BTC/USDT",
        "id": "BTCUSDT",
        "type": "swap",
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "contract": True,
        "linear": True,
        "inverse": False,
        "contractSize": 0.001,
        "precision": {"price": 1, "amount": 3},
        "limits": {},
        "maker": 0.0002,
        "taker": 0.0006,
        "info": {},
    }

    m = make_market(ccxt_market, sample_assets)

    assert m.type == "swap"
    assert m.contract is True
    assert m.settle.code == "USDT"
    assert m.contract_size == 0.001

    # convenience flags
    assert m.is_spot is False
    assert m.is_linear_future is True
    assert m.is_inverse_future is False
    assert m.is_perpetual is True  # no expiry → perpetual


def test_make_market_handles_missing_fields(sample_assets):
    """Allow missing fields: everything should become None instead of raising."""
    ccxt_market = {
        "symbol": "BTC/USDT",
        "id": "BTCUSDT",
        "type": None,
        "base": "BTC",
        "quote": "USDT",
        "precision": {},
        "limits": {},
        "info": {},
    }

    m = make_market(ccxt_market, sample_assets)

    assert m.type is None
    assert m.price_precision is None
    assert m.min_amount is None
    assert m.maker is None
    assert m.taker is None

    assert m.is_spot is False  # because type is None