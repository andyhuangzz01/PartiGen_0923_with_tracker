import importlib
from enum import Enum


def make_enum(enum_path: str, name: str) -> Enum:
    module_path, class_name = enum_path.rsplit(".", 1)
    enum_class = getattr(importlib.import_module(module_path), class_name)
    return enum_class[name]
