#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides configuration information for components in data_sources.

Created on Mon  Sep 18 19:17:23 2023

@author_ dhaneor
"""
import json
import os
import requests
import sys
from random import randint
from typing import Sequence, Optional, TypeVar
from uuid import uuid4

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

import config as cnf  # noqa: F401, E402

from zmqbricks.sockets import SockDef  # noqa: F401, E402
from zmqbricks.fukujou.curve import generate_curve_key_pair  # noqa: F401, E402
from data_sources.util.random_names import random_elven_name as rand_name  # noqa: E402
from keys import amanya as keys  # noqa: F401, E402

ScrollT = TypeVar("ScrollT", bound=object)

# endpoints
#
cnf.collector_sub = "tcp://127.0.0.1:5582"
cnf.collector_pub = "tcp://127.0.0.1:5583"
cnf.collector_mgmt = "tcp://127.0.0.1:5570"
cnf.collector_hb = "tcp://127.0.0.1:5580"

ohlcv_repo_req = "inproc://ohlcv_repository"


class BaseConfig:
    """Base configuration for components."""

    desc: str = ""  # service description, just for printing, not essential

    encrypted: bool = True  # use encryption or not

    hb_interval: float = cnf.HB_INTERVAL  # heartbeat interval (seconds)
    hb_liveness: int = cnf.HB_LIVENESS  # heartbeat liveness (max missed heartbeats)
    rgstr_timeout: int = cnf.RGSTR_TIMEOUT  # registration timeout (seconds)
    rgstr_max_errors: int = cnf.RGSTR_MAX_ERRORS  # max no of registration errors
    rgstr_log_interval: int = cnf.RGSTR_LOG_INTERVAL  # resend request after (secs)

    def __init__(
        self,
        exchange: Optional[str] = "all",
        markets: Optional[Sequence[str]] = ["all"],
        sock_defs: Sequence[SockDef] = [],
        **kwargs
    ) -> None:
        self.uid: str = str(uuid4())

        self.exchange: str = exchange
        self.markets: Sequence[str] = markets
        self.desc: Optional[str] = kwargs.get("desc", BaseConfig.desc)

        self._sock_defs: Sequence[SockDef] = sock_defs
        self._hb_interval: float = kwargs.get("hb_interval", BaseConfig.hb_interval)
        self._hb_liveness: int = kwargs.get("hb_liveness", BaseConfig.hb_liveness)
        self._rgstr_timeout: int = kwargs.get(
            "rgstr_timeout", BaseConfig.rgstr_timeout
        )
        self._rgstr_max_errors: int = kwargs.get(
            "rgstr_max_errors", BaseConfig.rgstr_max_errors
        )

        self.public_key, self.private_key = generate_curve_key_pair()

        self._endpoints: dict[str, str] = {}

    @property
    def service_name(self) -> str:
        return (
            f"{self.service_type.capitalize()} for {self.exchange.upper()} "
            f"{[m.upper() for m in self.markets]}"
        )

    @property
    def endpoints(self) -> dict[str, str]:
        if not cnf.DEV_ENV:
            ip = self.external_ip()

            for name, endpoint in self._endpoints.items():
                self._endpoints[name] = endpoint.replace("*", ip)
                self._endpoints[name] = endpoint.replace("127.0.0.1", ip)
                self._endpoints[name] = endpoint.replace("localhost", ip)

        return self._endpoints

    @property
    def hb_addr(self) -> str:
        return self.endpoints.get("heartbeat", None)

    @property
    def rgstr_addr(self) -> str:
        return self.endpoints.get("registration", None)

    @property
    def req_addr(self) -> str:
        return self.endpoints.get("requests", None)

    @property
    def pub_addr(self) -> str:
        return self.endpoints.get("publisher", None)

    @property
    def mgmt_addr(self) -> str:
        return self.endpoints.get("management", None)

    @property
    def external_ip(self) -> str:
        return requests.get('https://api.ipify.org').text

    # ..................................................................................
    def as_dict(self) -> dict:
        """Get the dictionary representation"""
        return {
            "service_type": self.service_type,
            "service_name": self.service_name,
            "endpoints": self.endpoints
        } | {
            var: getattr(self, var) for var in vars(self)
            if not var.startswith("_") or var == "private_key"
        }

    def as_json(self) -> str:
        """Get the JSON representation"""
        return json.dumps(self.as_dict(), indent=2)

    # ..................................................................................
    @staticmethod
    def from_json(json_str: str) -> "BaseConfig":
        """Build a configuration object from a JSON string."""
        return json.loads(json_str, object_hook=BaseConfig.from_dict)

    @staticmethod
    def from_dict(d: dict) -> "BaseConfig":
        """Build a configuration object from a dictionary."""
        return BaseConfig(**d)


# --------------------------------------------------------------------------------------
class Streamer(BaseConfig):
    """Configuration for the streamer component."""

    service_type: str = "streamer"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = rand_name(gender='male')
        self.register_at = kwargs.get("register_at", cnf.collector_mgmt)

        port = randint(cnf.STREAMER_BASE_PORT, cnf.STREAMER_BASE_PORT + 50)
        self.publisher_addr = (f"tcp://127.0.0.1:{port}")

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "publisher": self.publisher_addr,
        }


class Collector(BaseConfig):
    """Configuration for the collector component."""

    service_type: str = "collector"
    no_consumer_no_subs: bool = False  # unsubscribe from upstream if no consumers
    max_cache_size: int = 1000  # for duplicate check
    kinsfolk_check_interval: int = 15  # seconds

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = rand_name(gender='male')
        self.SUBSCRIBER_ADDR = cnf.collector_sub
        self.PUBLISHER_ADDR = cnf.collector_pub
        self.RGSTR_ADDR = cnf.collector_mgmt
        self.HB_ADDR = cnf.collector_hb
        self.public_key, self.private_key = cnf.CoLLECTOR_KEYS

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "publisher": self.PUBLISHER_ADDR,
            "subscriber": self.SUBSCRIBER_ADDR,
            "registration": self.RGSTR_ADDR,
            "heartbeat": self.HB_ADDR,
        }


class OhlcvRegistry(BaseConfig):
    """Configuration for the ohlcv registry"""
    PUBLISHER_ADDR = "inproc://craeft_pond"
    REPO_ADDR = ohlcv_repo_req
    cnf.collector_PUB_ADDR = cnf.collector_pub
    CONTAINER_SIZE_LIMIT = 1000


class OhlcvRepository(BaseConfig):
    """Configuration for ohlcv_repository."""
    REQUESTS_ADDR = ohlcv_repo_req


class Amanya(BaseConfig):
    """Configuration for the Amanya component."""

    service_type: str = "amanya"
    desc: str = "Central Configuration Service"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._endpoints: dict[str, str] = {
            "registration": "tcp://*:6000",
            "requests": "tcp://*:6001",
            "publisher": "tcp://*:6002",
            "heartbeat": "tcp://*:33333"
        }

        self.public = keys.public
        self.private = keys.private


# --------------------------------------------------------------------------------------
# ConfigT = TypeVar("ConfigT", ["BaseConfig"])

valid_service_types = {
    "streamer": Streamer,
    "collector": Collector,
    "ohlcv_registry": OhlcvRegistry,
    "ohlcv_repository": OhlcvRepository,
    "amanya": Amanya,
}


def get_config(
    service_type: str,
    exchange: str,
    markets: Sequence[str],
    sock_defs: Sequence[SockDef],
    **kwargs
) -> BaseConfig:
    """Get the configuration for a component.

    NOTE: Values for service_type, exchange, and market and sock_defs
    must be specified! See the definitions above for more information
    about possible options that can be provided as keyword argument!

    Parameters
    ----------
    service_type : str
        The type of the component.
    exchange : str
        The name of the exchange.
    market : str
        The name of the market.
    sock_defs : Sequence[SockDef]
        The socket definitions for the component.

    Returns
    -------
    ConfigT
        The configuration for the component.

    Raises
    ------
    ValueError
        If the service_type is not a valid service type.
    """
    if service_type not in valid_service_types:
        raise ValueError(f"Invalid service type: {service_type}")

    return valid_service_types[service_type](
        exchange=exchange or kwargs.pop("sock_defs"),
        markets=markets or kwargs.pop("markets"),
        sock_defs=sock_defs,
        **kwargs
    )


def get_rgstr_info(service_type, exchange="kcuoin", market="spot") -> ScrollT | None:

    markets = [market] if isinstance(market, str) else market

    if service_type == "collector":
        cnf.collector_conf = Collector(exchange, markets)

        class C:
            endpoint = cnf.collector_conf.endpoints.get("registration")
            public_key = cnf.collector_conf.public_key

            def __repr__(self):
                return f"C(endpoint={self.endpoint}, public_key={self.public_key})"

        return C()


if __name__ == "__main__":
    # c = get_config("collector", "kucoin", ["spot"], [])

    # [print(f"{k} -> {v}") for k, v in vars(c).items()]

    print(get_rgstr_info("collector", "kucoin", "spot"))
