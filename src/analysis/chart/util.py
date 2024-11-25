#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24 21:30:20 2024

@author dhaneor
"""
import os
import yaml


class DotDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        return self.to_yaml()

    def __str__(self):
        return self.to_yaml()

    def to_yaml(self):
        """Convert the DotDict to a YAML string, including nested values."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_dict(self):
        """Recursively convert DotDict (and nested DotDicts) to standard dictionaries"""
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):  # Check if the value is a DotDict
                result[key] = value.to_dict()  # Recursively convert nested DotDict
            else:
                result[key] = value
        return result


def load_yaml_as_dotdict(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    if not data:
        raise ValueError('config.yaml is not available or empty')

    return DotDict(data)


# Determine the script's directory and ensure config.yaml is loaded from there
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
config_file = os.path.join(script_dir, 'config.yaml')    # Full path to config.yaml

config = load_yaml_as_dotdict(config_file)


if __name__ == '__main__':
    print(config)
