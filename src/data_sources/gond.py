#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a class that simplifies the creation of components for
the data sources/analysis framework.

The class combines the following parts:
- kinsfolk registry
- heartbeat sending/monitoring
- registration of new kinsmen (peers)

Created on Sat Oct 07 12:01:23 2023

@author_ dhaneor
"""
from typing import TypeVar

import zmqbricks.kinsfolk as kf
import zmqbricks.registration as rgstr
import zmqbricks.heartbeat as hb
from zmq_config import BaseConfig

configT = TypeVar("configT", bound=BaseConfig)


# ======================================================================================
class Gond:
    """Skeleton class for components in the data sources/analysis framework."""

    kinsfolk: kf.Kinsfolk  # kinsfolk registry component
    registration: object  # registration monitor component
    heartbeat: object  # heartbeat sending/monitoring component
    craeft: object  # craeft component (the main task of the component)

    def __init__(self, config: configT):
        self.config = config

    def run(self) -> None:
        ...
