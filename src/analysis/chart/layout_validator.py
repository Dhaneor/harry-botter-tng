#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 25 21:37:20 2024

@author dhaneor
"""
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class LayoutValidator:
    @staticmethod
    def validate_basic(layout):
        logger.info("basic layout validation: PASS")
        pass

    @staticmethod
    def validate_intermediate(layout):
        LayoutValidator.validate_basic(layout)
        pass

    @staticmethod
    def validate_advanced(layout):
        # Implement advanced validation logic
        pass


def validate_layout(validation_level='basic'):
    def decorator(cls):
        original_init = cls.__init__

        @wraps(cls.__init__)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            if validation_level == 'basic':
                LayoutValidator.validate_basic(self.layout)
            elif validation_level == 'intermediate':
                LayoutValidator.validate_basic(self.layout)
                LayoutValidator.validate_intermediate(self.layout)
            elif validation_level == 'advanced':
                LayoutValidator.validate_basic(self.layout)
                LayoutValidator.validate_intermediate(self.layout)
                LayoutValidator.validate_advanced(self.layout)
            else:
                raise ValueError(f"Unknown validation level: {validation_level}")

        cls.__init__ = new_init
        return cls

    return decorator
