#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 26 02:32:20 2024

TODO:   The 'future' field is set to False be default. This needs to
        adjusted if we ever want to use this for Futures symbols/
        markets.

@author dhaneor
"""

from pprint import pprint

DEFAULT_FEE = 0.001


def safe_string(dictionary, key, default=None):
    return (
        str(dictionary[key])
        if key in dictionary and dictionary[key] is not None
        else default
    )


def safe_float(dictionary, key, default=None):
    try:
        return (
            float(dictionary[key])
            if key in dictionary and dictionary[key] is not None
            else default
        )
    except (ValueError, TypeError):
        return default


def safe_integer(dictionary, key, default=None):
    try:
        return (
            int(dictionary[key])
            if key in dictionary and dictionary[key] is not None
            else default
        )
    except (ValueError, TypeError):
        return default


def safe_value(dictionary, key, default=None):
    return dictionary[key] if key in dictionary else default


def safe_number(dictionary, key, default=None):
    value = safe_string(dictionary, key, default)
    return float(value) if value is not None else default


def parse_lot_size(filters):
    for f in filters:
        if f["filterType"] == "LOT_SIZE":
            return {
                "min": safe_float(f, "minQty"),
                "max": safe_float(f, "maxQty"),
                "step": safe_float(f, "stepSize"),
            }
    return {}


def parse_price_filter(filters):
    for f in filters:
        if f["filterType"] == "PRICE_FILTER":
            return {
                "min": safe_float(f, "minPrice"),
                "max": safe_float(f, "maxPrice"),
                "tickSize": safe_float(f, "tickSize"),
            }
    return {}


def parse_min_notional(filters):
    for f in filters:
        if f["filterType"] == "MIN_NOTIONAL":
            return {
                "min": safe_float(f, "minNotional"),
            }
    return {}


def parse_permissions(market):
    """
    Parses `spot` and `margin` from Binance's permissionSets.
    """
    permissions = market.get("permissionSets", [])[0] \
        if market.get("permissionSets") else []
    spot = "SPOT" in permissions
    margin = "MARGIN" in permissions
    return spot, margin


def convert_market(market):
    """
    Converts Binance market data to CCXT unified format.
    """
    id = safe_string(market, "symbol")
    base_id = safe_string(market, "baseAsset")
    quote_id = safe_string(market, "quoteAsset")
    base = base_id.upper()
    quote = quote_id.upper()
    precision = {
        "price": safe_integer(market, "quotePrecision"),
        "amount": safe_integer(market, "baseAssetPrecision"),
    }

    filters = safe_value(market, "filters", [])
    lot_size = parse_lot_size(filters)
    price_filter = parse_price_filter(filters)
    min_notional = parse_min_notional(filters)

    # Parse spot and margin from permissions
    spot, margin = parse_permissions(market)

    return {
        "id": id,
        "symbol": f"{base}/{quote}",
        "base": base,
        "quote": quote,
        "precision": precision,
        "limits": {
            "amount": {
                "min": lot_size.get("min"),
                "max": lot_size.get("max"),
            },
            "price": {
                "min": price_filter.get("min"),
                "max": price_filter.get("max"),
            },
            "cost": {
                "min": min_notional.get("min"),
            },
        },
        "taker": DEFAULT_FEE,  # Hardcoded default taker fee
        "maker": DEFAULT_FEE,  # Hardcoded default maker fee
        "margin": margin,  # Derived from permissions
        "spot": spot,  # Derived from permissions
        "future": False,  # Assuming non-futures markets
        "type": "spot" if spot else "unknown",
        "active": market.get('status') == 'TRADING',
        "status": market.get('status'),
        "contract": False,
        "info": market,  # Raw market data for reference
    }


if __name__ == "__main__":
    info = {
        "allowTrailingStop": True,
        "allowedSelfTradePreventionModes": [
            "EXPIRE_TAKER",
            "EXPIRE_MAKER",
            "EXPIRE_BOTH",
        ],
        "baseAsset": "BTC",
        "baseAssetPrecision": 8,
        "baseCommissionPrecision": 8,
        "cancelReplaceAllowed": True,
        "defaultSelfTradePreventionMode": "EXPIRE_MAKER",
        "filters": [
            {
                "filterType": "PRICE_FILTER",
                "maxPrice": "1000000.00000000",
                "minPrice": "0.01000000",
                "tickSize": "0.01000000",
            },
            {
                "filterType": "LOT_SIZE",
                "maxQty": "9000.00000000",
                "minQty": "0.00001000",
                "stepSize": "0.00001000",
            },
            {"filterType": "ICEBERG_PARTS", "limit": 10},
            {
                "filterType": "MARKET_LOT_SIZE",
                "maxQty": "69.28542500",
                "minQty": "0.00000000",
                "stepSize": "0.00000000",
            },
            {
                "filterType": "TRAILING_DELTA",
                "maxTrailingAboveDelta": 2000,
                "maxTrailingBelowDelta": 2000,
                "minTrailingAboveDelta": 10,
                "minTrailingBelowDelta": 10,
            },
            {
                "askMultiplierDown": "0.2",
                "askMultiplierUp": "5",
                "avgPriceMins": 5,
                "bidMultiplierDown": "0.2",
                "bidMultiplierUp": "5",
                "filterType": "PERCENT_PRICE_BY_SIDE",
            },
            {
                "applyMaxToMarket": False,
                "applyMinToMarket": True,
                "avgPriceMins": 5,
                "filterType": "NOTIONAL",
                "maxNotional": "9000000.00000000",
                "minNotional": "5.00000000",
            },
            {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": 200},
            {"filterType": "MAX_NUM_ALGO_ORDERS", "maxNumAlgoOrders": 5},
        ],
        "icebergAllowed": True,
        "isMarginTradingAllowed": True,
        "isSpotTradingAllowed": True,
        "ocoAllowed": True,
        "orderTypes": [
            "LIMIT",
            "LIMIT_MAKER",
            "MARKET",
            "STOP_LOSS",
            "STOP_LOSS_LIMIT",
            "TAKE_PROFIT",
            "TAKE_PROFIT_LIMIT",
        ],
        "otoAllowed": True,
        "permissionSets": [
            [
                "SPOT",
                "MARGIN",
                "TRD_GRP_004",
                "TRD_GRP_005",
                "TRD_GRP_006",
                "TRD_GRP_009",
                "TRD_GRP_010",
                "TRD_GRP_011",
                "TRD_GRP_012",
                "TRD_GRP_013",
                "TRD_GRP_014",
                "TRD_GRP_015",
                "TRD_GRP_016",
                "TRD_GRP_017",
                "TRD_GRP_018",
                "TRD_GRP_019",
                "TRD_GRP_020",
                "TRD_GRP_021",
                "TRD_GRP_022",
                "TRD_GRP_023",
                "TRD_GRP_024",
                "TRD_GRP_025",
                "TRD_GRP_026",
                "TRD_GRP_027",
                "TRD_GRP_028",
                "TRD_GRP_029",
                "TRD_GRP_030",
                "TRD_GRP_031",
                "TRD_GRP_032",
                "TRD_GRP_033",
                "TRD_GRP_034",
                "TRD_GRP_035",
                "TRD_GRP_036",
                "TRD_GRP_037",
                "TRD_GRP_038",
                "TRD_GRP_039",
                "TRD_GRP_040",
                "TRD_GRP_041",
                "TRD_GRP_042",
                "TRD_GRP_043",
                "TRD_GRP_044",
                "TRD_GRP_045",
                "TRD_GRP_046",
                "TRD_GRP_047",
                "TRD_GRP_048",
                "TRD_GRP_049",
                "TRD_GRP_050",
                "TRD_GRP_051",
                "TRD_GRP_052",
                "TRD_GRP_053",
                "TRD_GRP_054",
                "TRD_GRP_055",
                "TRD_GRP_056",
                "TRD_GRP_057",
                "TRD_GRP_058",
                "TRD_GRP_059",
                "TRD_GRP_060",
                "TRD_GRP_061",
                "TRD_GRP_062",
                "TRD_GRP_063",
                "TRD_GRP_064",
                "TRD_GRP_065",
                "TRD_GRP_066",
                "TRD_GRP_067",
                "TRD_GRP_068",
                "TRD_GRP_069",
                "TRD_GRP_070",
                "TRD_GRP_071",
                "TRD_GRP_072",
                "TRD_GRP_073",
                "TRD_GRP_074",
                "TRD_GRP_075",
                "TRD_GRP_076",
                "TRD_GRP_077",
                "TRD_GRP_078",
                "TRD_GRP_079",
                "TRD_GRP_080",
                "TRD_GRP_081",
                "TRD_GRP_082",
                "TRD_GRP_083",
                "TRD_GRP_084",
                "TRD_GRP_085",
                "TRD_GRP_086",
                "TRD_GRP_087",
                "TRD_GRP_088",
                "TRD_GRP_089",
                "TRD_GRP_090",
                "TRD_GRP_091",
                "TRD_GRP_092",
                "TRD_GRP_093",
                "TRD_GRP_094",
                "TRD_GRP_095",
                "TRD_GRP_096",
                "TRD_GRP_097",
                "TRD_GRP_098",
                "TRD_GRP_099",
                "TRD_GRP_100",
                "TRD_GRP_101",
                "TRD_GRP_102",
                "TRD_GRP_103",
                "TRD_GRP_104",
                "TRD_GRP_105",
                "TRD_GRP_106",
                "TRD_GRP_107",
                "TRD_GRP_108",
                "TRD_GRP_109",
                "TRD_GRP_110",
                "TRD_GRP_111",
                "TRD_GRP_112",
                "TRD_GRP_113",
                "TRD_GRP_114",
                "TRD_GRP_115",
                "TRD_GRP_116",
                "TRD_GRP_117",
                "TRD_GRP_118",
                "TRD_GRP_119",
                "TRD_GRP_120",
                "TRD_GRP_121",
                "TRD_GRP_122",
                "TRD_GRP_123",
                "TRD_GRP_124",
                "TRD_GRP_125",
                "TRD_GRP_126",
                "TRD_GRP_127",
                "TRD_GRP_128",
                "TRD_GRP_129",
                "TRD_GRP_130",
                "TRD_GRP_131",
                "TRD_GRP_132",
                "TRD_GRP_133",
                "TRD_GRP_134",
                "TRD_GRP_135",
                "TRD_GRP_136",
                "TRD_GRP_137",
                "TRD_GRP_138",
                "TRD_GRP_139",
                "TRD_GRP_140",
                "TRD_GRP_141",
                "TRD_GRP_142",
                "TRD_GRP_143",
                "TRD_GRP_144",
                "TRD_GRP_145",
                "TRD_GRP_146",
                "TRD_GRP_147",
                "TRD_GRP_148",
                "TRD_GRP_149",
                "TRD_GRP_150",
                "TRD_GRP_151",
                "TRD_GRP_152",
                "TRD_GRP_153",
                "TRD_GRP_154",
                "TRD_GRP_155",
                "TRD_GRP_156",
                "TRD_GRP_157",
                "TRD_GRP_158",
                "TRD_GRP_159",
                "TRD_GRP_160",
                "TRD_GRP_161",
                "TRD_GRP_162",
                "TRD_GRP_163",
                "TRD_GRP_164",
                "TRD_GRP_165",
                "TRD_GRP_166",
                "TRD_GRP_167",
                "TRD_GRP_168",
                "TRD_GRP_169",
                "TRD_GRP_170",
                "TRD_GRP_171",
                "TRD_GRP_172",
                "TRD_GRP_173",
                "TRD_GRP_174",
                "TRD_GRP_175",
                "TRD_GRP_176",
                "TRD_GRP_177",
                "TRD_GRP_178",
                "TRD_GRP_179",
                "TRD_GRP_180",
                "TRD_GRP_181",
                "TRD_GRP_182",
                "TRD_GRP_183",
                "TRD_GRP_184",
                "TRD_GRP_185",
                "TRD_GRP_186",
                "TRD_GRP_187",
                "TRD_GRP_188",
                "TRD_GRP_189",
                "TRD_GRP_190",
                "TRD_GRP_191",
                "TRD_GRP_192",
                "TRD_GRP_193",
                "TRD_GRP_194",
                "TRD_GRP_195",
                "TRD_GRP_196",
                "TRD_GRP_197",
                "TRD_GRP_198",
                "TRD_GRP_199",
                "TRD_GRP_200",
                "TRD_GRP_201",
                "TRD_GRP_202",
                "TRD_GRP_203",
                "TRD_GRP_204",
                "TRD_GRP_205",
                "TRD_GRP_206",
                "TRD_GRP_207",
                "TRD_GRP_208",
                "TRD_GRP_209",
                "TRD_GRP_210",
                "TRD_GRP_211",
                "TRD_GRP_212",
                "TRD_GRP_213",
                "TRD_GRP_214",
                "TRD_GRP_215",
                "TRD_GRP_216",
                "TRD_GRP_217",
                "TRD_GRP_218",
                "TRD_GRP_219",
                "TRD_GRP_220",
                "TRD_GRP_221",
                "TRD_GRP_222",
                "TRD_GRP_223",
                "TRD_GRP_224",
                "TRD_GRP_225",
                "TRD_GRP_226",
                "TRD_GRP_227",
                "TRD_GRP_228",
                "TRD_GRP_229",
                "TRD_GRP_230",
                "TRD_GRP_231",
                "TRD_GRP_232",
                "TRD_GRP_233",
                "TRD_GRP_234",
            ]
        ],
        "permissions": [],
        "quoteAsset": "USDT",
        "quoteAssetPrecision": 8,
        "quoteCommissionPrecision": 8,
        "quoteOrderQtyMarketAllowed": True,
        "quotePrecision": 8,
        "status": "TRADING",
        "symbol": "BTCUSDT",
    }

    ccxt_format = convert_market(info)
    ccxt_format["info"] = None
    pprint(ccxt_format)
