#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import sys
import unittest

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from indicators.indicator_parameter import Parameter  # noqa: E402

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)


class TestParameter(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.param = Parameter(
            name="TestParameter",
            initial_value=10,
            _value=10,
            hard_min=1,
            hard_max=20,
            step=1
        )

    def test_parameter_value(self):
        self.assertEqual(self.param.value, 10)

    def test_parameter_value_setter(self):
        self.param.value = 15
        self.assertEqual(self.param.value, 15)

    def test_parameter_value_setter_invalid_type(self):
        with self.assertRaises(TypeError):
            self.param.value = "invalid_value"

    def test_parameter_value_setter_outside_space(self):
        with self.assertRaises(ValueError):
            self.param.value = 25

    def test_parameter_value_setter_hard_min(self):
        self.param.hard_min = 5
        with self.assertRaises(ValueError):
            self.param.value = 3

    def test_parameter_value_setter_hard_max(self):
        self.param.hard_max = 15
        with self.assertRaises(ValueError):
            self.param.value = 17

    def test_iteration(self):
        for i in self.param:
            self.assertIn(i, range(1, 21))

    def test_post_init(self):
        self.assertEqual(self.param._space, (1, 20, 1))
        self.assertFalse(self.param._enforce_int)

    def test_post_init_with_enforce_int(self):
        param = Parameter(
            name="TestParameterWithEnforceInt",
            initial_value=10,
            _value=10,
            hard_min=1,
            hard_max=20,
            step=1,
            _enforce_int=True
        )
        self.assertEqual(param._space, (1, 20, 1))
        self.assertTrue(param._enforce_int)

    def test_post_init_with_hard_limits(self):
        param = Parameter(
            name="TestParameterWithHardLimits",
            initial_value=10,
            _value=10,
            hard_min=7,
            hard_max=15,
            step=1
        )
        self.assertEqual(param.space, (7, 15, 1))
        self.assertEqual(param.hard_min, 7)
        self.assertEqual(param.hard_max, 15)

    def test_set_space(self):
        param = Parameter(
            name="TestParameterWithHardLimits",
            initial_value=10,
            _value=10,
            hard_min=7,
            hard_max=15,
            step=1
        )
        with self.assertRaises(PermissionError):
            param.space = (5, 20, 2)


if __name__ == '__main__':
    unittest.main()
