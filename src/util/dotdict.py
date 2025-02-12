#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides simple dot-notation access to nested dictionaries.

Created on Jan 09 13:20:23 2025

@author dhaneor
"""
import yaml


class DotDict(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            super().__setitem__(key, value)
        return value

    def __repr__(self):
        return self.to_yaml()

    def __str__(self):
        return self.to_yaml()

    def to_yaml(self):
        """Convert the DotDict to a YAML string, including nested values."""
        return yaml.dump(
            self.to_dict(), default_flow_style=False, sort_keys=False, indent=4
        )

    def to_dict(self):
        """Recursively convert DotDict (and nested DotDicts) to standard dictionaries"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):  # Check if the value is a DotDict
                result[key] = value.to_dict()  # Recursively convert nested DotDict
            else:
                result[key] = value if key != "info" else None
        return result