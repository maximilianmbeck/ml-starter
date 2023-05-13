"""Utilities for using argparse with dataclasses."""

import argparse
from dataclasses import MISSING, fields
from typing import Any, Type, TypeVar, Union, cast, get_args, get_origin

from omegaconf import OmegaConf

from ml.core.config import BaseConfig

Config = TypeVar("Config", bound=BaseConfig)


def get_type_from_string(type_name: str) -> Type:
    """Parses a type name to a string.

    Args:
        type_name: The type name to parse.

    Returns:
        The type corresponding to the type name.

    Raises:
        ValueError: If the type name is not supported.
    """
    if type_name == "str":
        return str
    if type_name == "float":
        return float
    if type_name == "int":
        return int
    raise ValueError(type_name)


def add_args(parser: argparse.ArgumentParser, dc: Type[Config]) -> None:
    """Adds arguments to an argument parser from a dataclass.

    Args:
        parser: The argument parser to add arguments to.
        dc: The dataclass to add arguments from.

    Raises:
        NotImplementedError: If the dataclass has a field with an unsupported type.
    """
    for field in fields(dc):
        args: list[str] = []
        if field.metadata.get("short") is not None:
            args.append(f"-{field.metadata['short']}")
        args.append(f"--{field.name.replace('_', '-')}")
        kwargs: dict[str, Any] = {}
        if field.default != MISSING:
            kwargs["default"] = field.default
        elif field.default_factory != MISSING:
            kwargs["default"] = field.default_factory()
        if field.metadata.get("help") is not None:
            kwargs["help"] = field.metadata["help"]
        if field.type in (str, float, int):
            assert "default" in kwargs, f"Field {field.name} requires default"
            kwargs["type"] = field.type
        elif field.type is bool:
            assert "default" in kwargs, f"Field {field.name} requires default"
            default_val = cast(bool, kwargs["default"])
            kwargs["action"] = "store_false" if default_val else "store_true"
        elif field.type in ("str", "float", "int"):  # type: ignore[comparison-overlap]
            kwargs["type"] = get_type_from_string(cast(str, field.type))
            if "default" in kwargs and not isinstance(kwargs["default"], kwargs["type"]):
                kwargs.pop("default")
                kwargs["required"] = True
            elif "default" not in kwargs:
                kwargs["required"] = True
        elif get_origin(field.type) is Union:
            field_types = set(get_args(field.type))
            if type(None) in field_types:
                field_types.remove(type(None))
            if len(field_types) != 1:
                raise NotImplementedError(f"Field {field.name} has multiple types: {field_types}")
            field_type = list(field_types)[0]
            if field_type not in (str, float, int):
                raise NotImplementedError(f"Field {field.name} has unsupported type: {field_type}")
            kwargs["type"] = field_type
        else:
            raise NotImplementedError(f"Couldn't get type for {field.name}")
        parser.add_argument(*args, **kwargs)


def from_args(args: argparse.Namespace, dc: Type[Config]) -> Config:
    """Creates a dataclass from an argument parser namespace.

    Args:
        args: The argument parser namespace to create the dataclass from.
        dc: The dataclass to create.

    Returns:
        The dataclass created from the argument parser namespace.
    """
    values: dict[str, Any] = {}
    for field in fields(dc):
        values[field.name] = getattr(args, field.name)
    cfg = OmegaConf.structured(dc(**values))
    dc.resolve(cfg)
    return cfg
