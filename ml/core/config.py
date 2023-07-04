"""Defines the base config and object classes.

The base configuration dataclass provides some small increments over what
OmegaConf expects. The :func:``conf_field`` function exists to standardize
the metadata that each field should have.
"""

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from omegaconf import MISSING, DictConfig, OmegaConf

FieldType = Any


def conf_field(
    value: FieldType,
    *,
    help: str | None = None,  # pylint: disable=redefined-builtin
    short: str | None = None,
) -> FieldType:
    """Gets a field for a given value.

    Args:
        value: The default value for the current field
        help: An optional metadata field, which may be parsed to a command
            line argument in some CLIs
        short: An optional metadata field, which may be parsed to a command
            line argument in some CLIs

    Returns:
        The correctly constructed dataclass field
    """
    metadata: dict[str, Any] = {}
    if help is not None:
        metadata["help"] = help
    if short is not None:
        metadata["short"] = short

    if hasattr(value, "__call__"):
        return field(default_factory=value, metadata=metadata)
    if value.__class__.__hash__ is None:
        return field(default_factory=lambda: value, metadata=metadata)
    return field(default=value, metadata=metadata)


BaseConfigT = TypeVar("BaseConfigT", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Defines the base class for all configs."""

    name: str = conf_field(MISSING, short="n", help="The referenced name of the object to construct")

    @classmethod
    def get_defaults(cls: type[BaseConfigT]) -> dict[str, BaseConfigT]:
        """Returns default configurations.

        Returns:
            A dictionary of default configurations for the current config
        """
        return {}

    @classmethod
    def update(cls: type[BaseConfigT], config: DictConfig) -> DictConfig:
        """Runs post-construction config update.

        Args:
            config: The config to update
        """
        return cast(DictConfig, OmegaConf.merge(OmegaConf.structured(cls), config))

    @classmethod
    def resolve(cls: type[BaseConfigT], config: BaseConfigT) -> None:
        """Runs post-construction config resolution.

        Args:
            config: The config to resolve
        """


class BaseObject(Generic[BaseConfigT]):
    """Defines the base class for all objects."""

    __constants__ = ["config"]

    def __init__(self, config: BaseConfigT) -> None:
        self.config: BaseConfigT = config

        self._raw_config: DictConfig | None = None

    @property
    def raw_config(self) -> DictConfig:
        if self._raw_config is None:
            raise RuntimeError("Cannot access raw config yet; it has yet to be assigned")
        return self._raw_config

    def set_raw_config(self, raw_config: DictConfig) -> None:
        if self._raw_config is not None:
            raise RuntimeError("The raw config object was already written")
        self._raw_config = raw_config
