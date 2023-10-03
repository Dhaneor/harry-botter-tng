#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides configuration information for components in data_sources.

Created on Mon  Sep 18 19:17:23 2023

@author_ dhaneor
"""
import json
import os
import sys
from random import choice
from typing import Sequence, Optional
from uuid import uuid4

# --------------------------------------------------------------------------------------
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --------------------------------------------------------------------------------------

from zeromq.sockets import SockDef  # noqa: F401, E402
from data_sources.util.random_names import random_elven_name as rand_name  # noqa: E402

# some default values that are (can) be shared between components
DEFAULT_ENCODING = "utf-8"
DEFAULT_HB_INTERVAL = 1  # seconds
DEFAULT_HB_LIVENESS = 5  # heartbeat liveness
DEFAULT_RGSTR_TIMEOUT = 10  # seconds
DEFAULT_RGSTR_RESEND_AFTER = 900  # resend request after (secs)
DEFAULT_RGSTR_MAX_ERRORS = 10  # maximum number of registration errors
STREAMER_BASE_PORT = 5500

# endpoints
#
collector_sub = "tcp://127.0.0.1:5582"
collector_pub = "tcp://127.0.0.1:5583"
collector_mgmt = "tcp://127.0.0.1:5570"
collector_hb = "tcp://127.0.0.1:5580"

ohlcv_repo_req = "inproc://ohlcv_repository"


class BaseConfig:
    """Base configuration for components."""

    desc: str = ""  # service description, just for printing, not essential
    external_ip: str = "127.0.0.1"  # external ip address

    hb_interval: float = DEFAULT_HB_INTERVAL  # heartbeat interval (seconds)
    hb_liveness: int = DEFAULT_HB_LIVENESS  # heartbeat liveness (max missed heartbeats)
    rgstr_timeout: int = DEFAULT_RGSTR_TIMEOUT  # registration timeout (seconds)
    rgstr_max_errors: int = DEFAULT_RGSTR_MAX_ERRORS  # max no of registration errors
    rgstr_resend_after: int = DEFAULT_RGSTR_RESEND_AFTER  # resend request after (secs)

    def __init__(
        self,
        exchange: str,
        markets: Sequence[str],
        sock_defs: Sequence[SockDef] = [],
        **kwargs
    ) -> None:
        self.uid: str = str(uuid4())

        self.exchange: str = exchange
        self.markets: Sequence[str] = markets
        self.desc: Optional[str] = kwargs.get("desc", BaseConfig.desc)
        self.external_ip: Optional[str] = kwargs.get(
            "external_ip", BaseConfig.external_ip
        )

        self._sock_defs: Sequence[SockDef] = sock_defs
        self._hb_interval: float = kwargs.get("hb_interval", BaseConfig.hb_interval)
        self._hb_liveness: int = kwargs.get("hb_liveness", BaseConfig.hb_liveness)
        self._rgstr_timeout: int = kwargs.get(
            "rgstr_timeout", BaseConfig.rgstr_timeout
        )
        self._rgstr_max_errors: int = kwargs.get(
            "rgstr_max_errors", BaseConfig.rgstr_max_errors
        )

    @property
    def service_name(self) -> str:
        return (
            f"{self.service_type.capitalize()} for {self.exchange.upper()} "
            f"{[m.upper() for m in self.markets]}"
        )

    @property
    def endpoints(self) -> dict[str, str]:
        return {}

    # ..................................................................................
    def as_dict(self) -> dict:
        """Get the dictionary representation"""
        return {
            "service_type": self.service_type,
            "service_name": self.service_name,
            "endpoints": self.endpoints
        } | {
            var: getattr(self, var) for var in vars(self) if not var.startswith("_")
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
        super().__init__(**kwargs)
        self.name = rand_name(gender='male')
        self.register_at = kwargs.get("register_at", collector_mgmt)
        self.port = choice(range(STREAMER_BASE_PORT, STREAMER_BASE_PORT + 50))
        self.PUBLISHER_ADDR = f"tcp://127.0.0.1:{STREAMER_BASE_PORT + self.port}"
        self.REGISTER_AT = collector_mgmt

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "publisher": self.PUBLISHER_ADDR,
        }


class Collector(BaseConfig):
    """Configuration for the collector component."""

    service_type: str = "collector"
    no_consumer_no_subs: bool = False  # unsubscribe from upstream if no consumers
    max_cache_size: int = 1000  # for duplicate check
    kinsfolk_check_interval: int = 15  # seconds

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = rand_name(gender='male')
        self.SUBSCRIBER_ADDR = collector_sub
        self.PUBLISHER_ADDR = collector_pub
        self.MGMT_ADDR = collector_mgmt
        self.HB_ADDR = collector_hb

    @property
    def endpoints(self) -> dict[str, str]:
        return {
            "publisher": self.PUBLISHER_ADDR,
            "subscriber": self.SUBSCRIBER_ADDR,
            "management": self.MGMT_ADDR,
            "heartbeat": self.HB_ADDR,
        }


class OhlcvRegistry(BaseConfig):
    """Configuration for the ohlcv registry"""
    PUBLISHER_ADDR = "inproc://craeft_pond"
    REPO_ADDR = ohlcv_repo_req
    COLLECTOR_PUB_ADDR = collector_pub
    CONTAINER_SIZE_LIMIT = 1000


class OhlcvRepository(BaseConfig):
    """Configuration for ohlcv_repository."""
    REQUESTS_ADDR = ohlcv_repo_req


# --------------------------------------------------------------------------------------
# ConfigT = TypeVar("ConfigT", ["BaseConfig"])

valid_service_types = {
    "streamer": Streamer,
    "collector": Collector,
    "ohlcv_registry": OhlcvRegistry,
    "ohlcv_repository": OhlcvRepository,
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
